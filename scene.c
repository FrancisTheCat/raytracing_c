#include "codin/codin.h"

#include "codin/io.h"
#include "codin/os.h"
#include "codin/sort.h"
#include "codin/thread.h"

#include "stdatomic.h"

#include "raytracer.h"
#include "scene.h"

typedef struct __attribute__((aligned(32))) {
  i32    version, n_nodes, n_triangles, bvh_depth;
  Camera camera;
} Scene_File_Header;

extern void scene_save_writer(Writer const *w, Scene const *scene) {
  Scene_File_Header header = {
    .version     = 0,
    .n_nodes     = scene->bvh.nodes.len,
    .n_triangles = scene->triangles.len,
    .bvh_depth   = scene->bvh.depth,
    .camera      = scene->camera,
  };
  write_any(w, &header);
  Byte_Slice node_data = slice_to_bytes(scene->bvh.nodes);
  Byte_Slice tris_data = {
    .data = (byte *)scene->triangles.x[0],
    .len  = TRIANGLES_ALLOCATION_SIZE(scene->triangles.len),
  };
  write_bytes(w, node_data);
  write_bytes(w, tris_data);
}

extern bool scene_load_bytes(Byte_Slice data, Scene *scene) {
  assert((uintptr)data.data % 32 == 0);

  Scene_File_Header header;
  if (data.len < size_of(header)) {
    return false;
  }
  mem_copy(&header, data.data, size_of(header));

  if (data.len != size_of(header) + header.n_nodes * size_of(BVH_Node) + TRIANGLES_ALLOCATION_SIZE(header.n_triangles)) {
    return false;
  }

  scene->camera         = header.camera;
  scene->bvh.depth      = header.bvh_depth;
  scene->bvh.nodes.len  = header.n_nodes;
  scene->bvh.nodes.data = (BVH_Node *)(data.data + size_of(header));

  assert((uintptr)scene->bvh.nodes.data % 32 == 0);

  f32 *tris_data = (f32 *)(data.data + size_of(header) + header.n_nodes * size_of(BVH_Node));

  assert((uintptr)tris_data % 32 == 0);

  scene->triangles.x[0] = tris_data + header.n_triangles * 0;
  scene->triangles.x[1] = tris_data + header.n_triangles * 1;
  scene->triangles.x[2] = tris_data + header.n_triangles * 2;

  scene->triangles.y[0] = tris_data + header.n_triangles * 3;
  scene->triangles.y[1] = tris_data + header.n_triangles * 4;
  scene->triangles.y[2] = tris_data + header.n_triangles * 5;

  scene->triangles.z[0] = tris_data + header.n_triangles * 6;
  scene->triangles.z[1] = tris_data + header.n_triangles * 7;
  scene->triangles.z[2] = tris_data + header.n_triangles * 8;

  scene->triangles.aos = (Triangle_AOS *)(tris_data + header.n_triangles * 9);
  scene->triangles.len = header.n_triangles;

  return true;
}

internal void triangles_init(Triangles *triangles, isize len, Allocator allocator) {
  while (len % SIMD_WIDTH) {
    len += 1;
  }
  triangles->len = len;

  f32 *data = (f32 *)unwrap_err(mem_alloc_aligned(TRIANGLES_ALLOCATION_SIZE(len), SIMD_ALIGN, allocator));

  triangles->x[0] = data + len * 0;
  triangles->x[1] = data + len * 1;
  triangles->x[2] = data + len * 2;

  triangles->y[0] = data + len * 3;
  triangles->y[1] = data + len * 4;
  triangles->y[2] = data + len * 5;

  triangles->z[0] = data + len * 6;
  triangles->z[1] = data + len * 7;
  triangles->z[2] = data + len * 8;

  triangles->aos = (Triangle_AOS *)(data + len * 9);
}

internal void triangles_destroy(Triangles *triangles, Allocator allocator) {
  mem_free(triangles->x[0], TRIANGLES_ALLOCATION_SIZE(triangles->len), allocator);
}

internal void triangles_insert(Triangles *triangles, Triangle_Slice v, isize offset) {
  assert(v.len + offset <= triangles->len);
  assert(offset         >= 0);

  slice_iter_v(v, t, i, {
    triangles->x[0][offset + i] = t.positions[0].x;
    triangles->x[1][offset + i] = t.positions[1].x;
    triangles->x[2][offset + i] = t.positions[2].x;

    triangles->y[0][offset + i] = t.positions[0].y;
    triangles->y[1][offset + i] = t.positions[1].y;
    triangles->y[2][offset + i] = t.positions[2].y;

    triangles->z[0][offset + i] = t.positions[0].z;
    triangles->z[1][offset + i] = t.positions[1].z;
    triangles->z[2][offset + i] = t.positions[2].z;

    Vec3 edge1 = vec3_sub(vec3(t.positions[1].x, t.positions[1].y, t.positions[1].z), vec3(t.positions[0].x, t.positions[0].y, t.positions[0].z));
    Vec3 edge2 = vec3_sub(vec3(t.positions[2].x, t.positions[2].y, t.positions[2].z), vec3(t.positions[0].x, t.positions[0].y, t.positions[0].z));

    Vec2 delta_uv1 = vec2_sub(t.tex_coords[1], t.tex_coords[0]);
    Vec2 delta_uv2 = vec2_sub(t.tex_coords[2], t.tex_coords[0]);

    f32 d = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;
    if (abs_f32(d) < 0.0001f) {
      if (d < 0) {
        d = -0.0001f;
      } else {
        d =  0.0001f;
      }
    }

    f32 inv_d = 1.0f / d;

    Vec3 tangent   = vec3_normalize(vec3_scale(vec3_sub(vec3_scale(edge1, delta_uv2.y), vec3_scale(edge2, delta_uv1.y)), inv_d));
    Vec3 bitangent = vec3_normalize(vec3_scale(vec3_sub(vec3_scale(edge2, delta_uv1.x), vec3_scale(edge1, delta_uv2.x)), inv_d));

    triangles->aos[offset + i] = (Triangle_AOS) {
      .shader       = t.shader,
      .normal       = vec3_normalize(vec3_cross(edge1, edge2)),
      .normal_a     = t.normals[0],
      .normal_b     = t.normals[1],
      .normal_c     = t.normals[2],
      .tex_coords_a = t.tex_coords[0],
      .tex_coords_b = t.tex_coords[1],
      .tex_coords_c = t.tex_coords[2],
      .tangent      = tangent,
      .bitangent    = bitangent,
    };
  });
}

internal f32 aabb_surface_area(AABB *aabb) {
  f32 x = aabb->max.x - aabb->min.x;
  f32 y = aabb->max.y - aabb->min.y;
  f32 z = aabb->max.z - aabb->min.z;
  return 2.0f * (x * y + y * z + z * x);
}

internal void aabb_union(AABB *a, AABB *b, AABB *c) {
  c->min = vec3(
    min(a->min.x, b->min.x),
    min(a->min.y, b->min.y),
    min(a->min.z, b->min.z),
  );
  c->max = vec3(
    max(a->max.x, b->max.x),
    max(a->max.y, b->max.y),
    max(a->max.z, b->max.z),
  );
}

internal void aabb_triangle(Triangle *triangle, AABB *aabb) {
  aabb->min = vec3(
    min(triangle->positions[0].x, min(triangle->positions[1].x, triangle->positions[2].x)) - EPSILON,
    min(triangle->positions[0].y, min(triangle->positions[1].y, triangle->positions[2].y)) - EPSILON,
    min(triangle->positions[0].z, min(triangle->positions[1].z, triangle->positions[2].z)) - EPSILON,
  );
  aabb->max = vec3(
    max(triangle->positions[0].x, max(triangle->positions[1].x, triangle->positions[2].x)) + EPSILON,
    max(triangle->positions[0].y, max(triangle->positions[1].y, triangle->positions[2].y)) + EPSILON,
    max(triangle->positions[0].z, max(triangle->positions[1].z, triangle->positions[2].z)) + EPSILON,
  );
}

internal void aabb_triangle_slice(Triangle_Slice triangles, AABB *aabb) {
  *aabb = (AABB) {0};

  slice_iter_v(triangles, t, i, {
    AABB t_aabb;
    aabb_triangle(&t, &t_aabb);
    if (!i) {
      *aabb = t_aabb;
    }
    aabb_union(aabb, &t_aabb, aabb);
  });
}

internal void sort_triangle_slice(Triangle_Slice slice, isize axis) {
  assert(axis <  3);
  assert(axis >= 0);
  sort_slice_by(
    slice,
    i,
    j,
    ({
      Triangle *t_i = &IDX(slice, i);
      Triangle *t_j = &IDX(slice, j);
      f32 centroid_i = t_i->positions[0].data[axis] +
                       t_i->positions[1].data[axis] +
                       t_i->positions[2].data[axis];
      f32 centroid_j = t_j->positions[0].data[axis] +
                       t_j->positions[1].data[axis] +
                       t_j->positions[2].data[axis];
      centroid_i < centroid_j;
    })
  );
}

internal isize bvh_required_depth(isize n_triangles) {
  n_triangles = (n_triangles + SIMD_WIDTH - 1) / SIMD_WIDTH;

  isize n = 1, i = 0;
  while (n < n_triangles) {
    n *= SIMD_WIDTH;
    i += 1;
  }
  return i;
}

internal isize bvh_partition_triangles(isize n_triangles, isize per_child) {
  isize n = 0, left = n_triangles;
  while (n < n_triangles / 2 && left > per_child) {
    n    += per_child;
    left -= per_child;
  }
  return n;
}

typedef struct {
  isize          depth;
  BVH_Index      index;
  Triangle_Slice triangles;
  _Atomic bool   ready;
} BVH_Build_Task;

typedef struct {
  Scene                *scene;
  Slice(BVH_Build_Task) tasks;
  _Atomic isize         read;
  _Atomic isize         write;
  _Atomic isize         n_working;
  _Atomic isize         n_active;
} BVH_Construction_Context;

internal void bvh_build_thread(BVH_Construction_Context *ctx);

internal void bvh_build_threaded(Scene *scene, Triangle_Slice triangles, isize n_threads) {
  BVH_Construction_Context ctx = {
    .scene     = scene,
    .write     = 1,
    .n_active  = n_threads,
  };
  slice_init(&ctx.tasks, scene->bvh.nodes.len, context.temp_allocator);
  IDX(ctx.tasks, 0) = (BVH_Build_Task) {
    .triangles = triangles,
    .depth     = scene->bvh.depth,
    .index     = 0,
    .ready     = true,
  };

  ctx.n_working = 1;
  for_range(thread_id, 0, n_threads - 1) {
    thread_create((Thread_Proc)bvh_build_thread, &ctx, THREAD_STACK_DEFAULT, THREAD_TLS_DEFAULT);
  }
  ctx.n_working -= 1;
  bvh_build_thread(&ctx);

  while (ctx.n_active) {
    processor_yield();
  }
}

internal void bvh_build(BVH_Construction_Context *ctx, Scene *scene, Triangle_Slice triangles, isize depth, BVH_Index index);

internal void bvh_build_thread(BVH_Construction_Context *ctx) {
  for (;;) {
    isize read = atomic_fetch_add(&ctx->read, 1);
    while (read >= ctx->write) {
      if (ctx->n_working == 0) {
        ctx->n_active -= 1;
        return;
      }
      processor_yield();
    }

    BVH_Build_Task const *task = &IDX(ctx->tasks, read);
    ctx->n_working += 1;
    while (!task->ready) {
      processor_yield();
    }
    bvh_build(ctx, ctx->scene, task->triangles, task->depth, task->index);
    ctx->n_working -= 1;
  }
}

internal void bvh_build(
  BVH_Construction_Context *ctx,
  Scene                    *scene,
  Triangle_Slice            triangles,
  isize                     depth,
  BVH_Index                 index
) {
  if (triangles.len <= SIMD_WIDTH) {
    triangles_insert(&scene->triangles, triangles, (index - scene->bvh.last_row_offset) * SIMD_WIDTH);
    return;
  }

  isize per_child = bvh_n_leaf_nodes(depth);
  assert(per_child * SIMD_WIDTH >= triangles.len);
  assert(per_child >= SIMD_WIDTH);

  Triangle_Slice slices[SIMD_WIDTH] = { triangles, };
  isize        n_slices = 1;

  Triangle_Slice finished[SIMD_WIDTH] = {0};
  isize        n_finished = 0;

  while (n_slices != 0) {
    n_slices -= 1;
    Triangle_Slice slice = slices[n_slices];
    assert(slice.len > SIMD_WIDTH);

    isize split = bvh_partition_triangles(slice.len, per_child);

    Triangle_Slice left, right;
    left  = slice_end(slice,   split);
    right = slice_start(slice, split);
    
    f32 min_surface_area = F32_INFINITY;
    i32 best_axis        = 0;
    for_range(axis, 0, 3) {
      sort_triangle_slice(slice, axis);
      AABB a, b;
      aabb_triangle_slice(left,  &a);
      aabb_triangle_slice(right, &b);
      f32 surface_area = aabb_surface_area(&a) + aabb_surface_area(&b);
      if (surface_area <= min_surface_area) {
        min_surface_area = surface_area;
        best_axis        = axis;
      }
    }

    if (best_axis != 2) {
      sort_triangle_slice(slice, best_axis);
    }

    if (left.len > per_child) {
      assert(n_slices < count_of(slices));
      slices[n_slices] = left;
      n_slices += 1;
    } else if (left.len) {
      assert(n_finished < count_of(finished));
      finished[n_finished] = left;
      n_finished += 1;
    }
    if (right.len > per_child) {
      assert(n_slices < count_of(slices));
      slices[n_slices] = right;
      n_slices += 1;
    } else if (right.len) {
      assert(n_finished < count_of(finished));
      finished[n_finished] = right;
      n_finished += 1;
    }
  }

  BVH_Node node = {0};
  slice_iter_v(slice_array(Slice(Triangle_Slice), finished), tris, i, {
    if (i >= n_finished) {
      break;
    }
    AABB aabb;
    aabb_triangle_slice(tris, &aabb);

    node.mins.x[i] = aabb.min.x;
    node.mins.y[i] = aabb.min.y;
    node.mins.z[i] = aabb.min.z;

    node.maxs.x[i] = aabb.max.x;
    node.maxs.y[i] = aabb.max.y;
    node.maxs.z[i] = aabb.max.z;

    isize child_index = index * SIMD_WIDTH + 1 + i;
    if (ctx == nil || depth <= 3) {
      bvh_build(nil, scene, tris, depth - 1, child_index);
    } else {
      isize write = atomic_fetch_add(&ctx->write, 1);
      IDX(ctx->tasks, write) = (BVH_Build_Task) {
        .triangles = tris,
        .depth     = depth - 1,
        .index     = child_index,
      };
      IDX(ctx->tasks, write).ready = true;
    }
  });

  IDX(scene->bvh.nodes, index) = node;
  return;
}

extern void scene_init(Scene *scene, Triangle_Slice triangles, Allocator allocator) {
  isize depth                = bvh_required_depth(triangles.len);
  isize n_internal           = bvh_n_internal_nodes(depth);
  scene->bvh.depth           = depth;
  scene->bvh.last_row_offset = n_internal;

  slice_init(&scene->bvh.nodes, n_internal, allocator);
  triangles_init(&scene->triangles, bvh_n_leaf_nodes(depth) * SIMD_WIDTH, allocator);
  
  bvh_build_threaded(scene, triangles, 12);
}
