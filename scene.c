#include "codin/codin.h"

#include "codin/allocators.h"
#include "codin/fmt.h"
#include "codin/io.h"
#include "codin/sort.h"

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
    .n_triangles = scene->triangles.cap,
    .bvh_depth   = scene->bvh.depth,
    .camera      = scene->camera,
  };
  write_any(w, &header);
  Byte_Slice node_data = slice_to_bytes(scene->bvh.nodes);
  Byte_Slice tris_data = {
    .data = (byte *)scene->triangles.x[0],
    .len  = TRIANGLES_ALLOCATION_SIZE(scene->triangles.cap),
  };
  write_bytes(w, node_data);
  write_bytes(w, tris_data);
}

extern b8 scene_load_bytes(Byte_Slice data, Scene *scene) {
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
  scene->bvh.nodes.cap  = header.n_nodes;
  scene->bvh.nodes.data = (BVH_Node *)(data.data + size_of(header));

  assert((uintptr)scene->bvh.nodes.data % 32 == 0);

  scene->bvh.nodes.allocator = panic_allocator();

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
  scene->triangles.cap = header.n_triangles;

  scene->triangles.allocator = panic_allocator();

  return true;
}

internal void triangles_init(Triangles *triangles, isize len, isize cap, Allocator allocator) {
  if (cap % SIMD_WIDTH) {
    cap += SIMD_WIDTH - cap % SIMD_WIDTH;
  }
  triangles->len       = len;
  triangles->cap       = cap;
  triangles->allocator = allocator;

  f32 *data = (f32 *)unwrap_err(mem_alloc_aligned(TRIANGLES_ALLOCATION_SIZE(cap), SIMD_ALIGN, allocator));

  triangles->x[0] = data + cap * 0;
  triangles->x[1] = data + cap * 1;
  triangles->x[2] = data + cap * 2;

  triangles->y[0] = data + cap * 3;
  triangles->y[1] = data + cap * 4;
  triangles->y[2] = data + cap * 5;

  triangles->z[0] = data + cap * 6;
  triangles->z[1] = data + cap * 7;
  triangles->z[2] = data + cap * 8;

  triangles->aos = (Triangle_AOS *)(data + cap * 9);
}

internal void triangles_destroy(Triangles *triangles) {
  mem_free(triangles->x[0], TRIANGLES_ALLOCATION_SIZE(triangles->cap), triangles->allocator);
}

internal void triangles_append(Triangles *triangles, Triangle_Slice v) {
  if (triangles->len + v.len >= triangles->cap) {
    isize new_cap = max(triangles->cap * 2, SIMD_WIDTH);
    if (triangles->cap + v.len > new_cap) {
      new_cap  = triangles->cap * 2 + ((v.len + SIMD_WIDTH + 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }
    assert(new_cap % 8 == 0);
    f32 *new_data = (f32 *)unwrap_err(mem_alloc_aligned(TRIANGLES_ALLOCATION_SIZE(new_cap), SIMD_ALIGN, triangles->allocator));

    mem_tcopy(new_data + new_cap * 0, triangles->x[0], triangles->len);
    mem_tcopy(new_data + new_cap * 1, triangles->x[1], triangles->len);
    mem_tcopy(new_data + new_cap * 2, triangles->x[2], triangles->len);

    mem_tcopy(new_data + new_cap * 3, triangles->y[0], triangles->len);
    mem_tcopy(new_data + new_cap * 4, triangles->y[1], triangles->len);
    mem_tcopy(new_data + new_cap * 5, triangles->y[2], triangles->len);

    mem_tcopy(new_data + new_cap * 6, triangles->z[0], triangles->len);
    mem_tcopy(new_data + new_cap * 7, triangles->z[1], triangles->len);
    mem_tcopy(new_data + new_cap * 8, triangles->z[2], triangles->len);

    mem_tcopy((Triangle_AOS *)(new_data + new_cap * 9), triangles->aos, triangles->len);

    triangles->x[0] = new_data + new_cap * 0;
    triangles->x[1] = new_data + new_cap * 1;
    triangles->x[2] = new_data + new_cap * 2;

    triangles->y[0] = new_data + new_cap * 3;
    triangles->y[1] = new_data + new_cap * 4;
    triangles->y[2] = new_data + new_cap * 5;

    triangles->z[0] = new_data + new_cap * 6;
    triangles->z[1] = new_data + new_cap * 7;
    triangles->z[2] = new_data + new_cap * 8;

    triangles->aos = (Triangle_AOS *)(new_data + new_cap * 9);

    triangles->cap = new_cap;
  }

  slice_iter_v(v, t, i, {
    triangles->x[0][triangles->len + i] = t.positions[0].x;
    triangles->x[1][triangles->len + i] = t.positions[1].x;
    triangles->x[2][triangles->len + i] = t.positions[2].x;

    triangles->y[0][triangles->len + i] = t.positions[0].y;
    triangles->y[1][triangles->len + i] = t.positions[1].y;
    triangles->y[2][triangles->len + i] = t.positions[2].y;

    triangles->z[0][triangles->len + i] = t.positions[0].z;
    triangles->z[1][triangles->len + i] = t.positions[1].z;
    triangles->z[2][triangles->len + i] = t.positions[2].z;

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

    triangles->aos[triangles->len + i] = (Triangle_AOS) {
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
  triangles->len += v.len;
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
      AABB aabb_i, aabb_j;
      aabb_triangle(&IDX(slice, i), &aabb_i);
      aabb_triangle(&IDX(slice, j), &aabb_j);

      aabb_i.min.data[axis] <= aabb_j.min.data[axis];
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

internal isize bvh_n_leaf_nodes(isize depth) {
  isize n = 1;
  for_range(i, 0, depth) {
    n *= SIMD_WIDTH;
  }
  return n;
}

internal isize bvh_n_internal_nodes(isize depth) {
  isize n = 1;
  isize nodes = 0;
  for_range(i, 0, depth) {
    nodes += n;
    n     *= SIMD_WIDTH;
  }
  return nodes;
}

internal isize bvh_partition_triangles(isize n_triangles, isize per_child) {
  isize n = 0, left = n_triangles;
  while (n < n_triangles / 2 && left > per_child) {
    n    += per_child;
    left -= n_triangles;
  }
  return n;
}

internal void bvh_build(Scene *scene, Triangle_Slice triangles, isize depth) {
  isize n_leaves = bvh_n_leaf_nodes(depth);

  isize per_child = n_leaves;

  if (triangles.len <= SIMD_WIDTH) {
    triangles_append(&scene->triangles, triangles);
    while (scene->triangles.len % SIMD_WIDTH) {
      scene->triangles.len += 1;
    }
    assert(scene->triangles.len <= scene->triangles.cap);
    return;
  }

  Triangle_Slice slices[SIMD_WIDTH] = { triangles, };
  isize        n_slices = 1;

  Triangle_Slice finished[SIMD_WIDTH] = {0};
  isize        n_finished = 0;

  while (n_slices != 0) {
    n_slices -= 1;
    Triangle_Slice slice = slices[n_slices];
    assert(slice.len > per_child);

    isize split = bvh_partition_triangles(slice.len, per_child);

    Triangle_Slice left, right;
    left  = slice_end(slice,   split);
    right = slice_start(slice, split);
    
    f32 min_surface_area = F32_INFINITY;
    i32 best_axis = 0;
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

    assert(n_slices   < count_of(slices));
    assert(n_finished < count_of(finished));

    if (left.len > per_child) {
      slices[n_slices] = left;
      n_slices += 1;
    } else if (left.len) {
      finished[n_finished] = left;
      n_finished += 1;
    }
    if (right.len > per_child) {
      slices[n_slices] = right;
      n_slices += 1;
    } else if (right.len) {
      finished[n_finished] = right;
      n_finished += 1;
    }
  }

  // [ 0, 1, 2, 3, 4, 4, 3, 4, 4, 2, 3, 4, 4, 3, 4, 4, 1, 2, 3, 4, 4, 3, 4, 4, 2, 3, 4, 4, 3, 4, 4, ]
  // [ 0, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, ]

  // child i of node n:
  // n + 1 + i * internal_nodes(depth - 1)

  BVH_Node node = {0};
  isize index = scene->bvh.nodes.len;
  vector_append(&scene->bvh.nodes, node);

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

    bvh_build(scene, tris, depth - 1);
  });

  IDX(scene->bvh.nodes, index) = node;
  return;
}

extern void scene_init(Scene *scene, Triangle_Slice triangles, Allocator allocator) {
  isize depth      = bvh_required_depth(triangles.len);
  isize n_internal = bvh_n_internal_nodes(depth);
  scene->bvh.depth = depth;

  vector_init(&scene->bvh.nodes, 0, n_internal, allocator);
  triangles_init(&scene->triangles, 0, ((triangles.len + SIMD_WIDTH) / SIMD_WIDTH) * SIMD_WIDTH, allocator);

  scene->triangles.allocator = panic_allocator();
  scene->bvh.nodes.allocator = panic_allocator();
  
  bvh_build(scene, triangles, depth);
}
