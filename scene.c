#include "codin/codin.h"

#include "codin/allocators.h"
#include "codin/fmt.h"
#include "codin/io.h"
#include "codin/sort.h"

#include "raytracer.h"
#include "scene.h"

typedef struct __attribute__((aligned(32))) {
  i32       version, n_nodes, n_triangles;
  i32 root;
  u8        _padding[16];
} Scene_File_Header;

extern void scene_save_writer(Writer const *w, Scene const *scene) {
  Scene_File_Header header = {
    .version     = 0,
    .n_nodes     = scene->bvh.nodes.len,
    .n_triangles = scene->triangles.cap,
  };
  write_any(w, &header);
  Byte_Slice node_data = slice_to_bytes(scene->bvh.nodes);
  Byte_Slice tris_data = {
    .data = (byte *)scene->triangles.a_x,
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

  scene->bvh.nodes.len  = header.n_nodes;
  scene->bvh.nodes.cap  = header.n_nodes;
  scene->bvh.nodes.data = (BVH_Node *)(data.data + size_of(header));

  assert((uintptr)scene->bvh.nodes.data % 32 == 0);

  scene->bvh.nodes.allocator = panic_allocator();

  f32 *tris_data = (f32 *)(data.data + size_of(header) + header.n_nodes * size_of(BVH_Node));

  assert((uintptr)tris_data % 32 == 0);

  scene->triangles.a_x = tris_data + header.n_triangles * 0;
  scene->triangles.a_y = tris_data + header.n_triangles * 1;
  scene->triangles.a_z = tris_data + header.n_triangles * 2;

  scene->triangles.b_x = tris_data + header.n_triangles * 3;
  scene->triangles.b_y = tris_data + header.n_triangles * 4;
  scene->triangles.b_z = tris_data + header.n_triangles * 5;

  scene->triangles.c_x = tris_data + header.n_triangles * 6;
  scene->triangles.c_y = tris_data + header.n_triangles * 7;
  scene->triangles.c_z = tris_data + header.n_triangles * 8;

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

  triangles->a_x = data + cap * 0;
  triangles->a_y = data + cap * 1;
  triangles->a_z = data + cap * 2;

  triangles->b_x = data + cap * 3;
  triangles->b_y = data + cap * 4;
  triangles->b_z = data + cap * 5;

  triangles->c_x = data + cap * 6;
  triangles->c_y = data + cap * 7;
  triangles->c_z = data + cap * 8;

  triangles->aos = (Triangle_AOS *)(data + cap * 9);
}

internal void triangles_destroy(Triangles *triangles) {
  mem_free(triangles->a_x, TRIANGLES_ALLOCATION_SIZE(triangles->cap), triangles->allocator);
}

internal void triangles_append(Triangles *triangles, Triangle_Slice v) {
  if (triangles->len + v.len >= triangles->cap) {
    isize new_cap = max(triangles->cap * 2, SIMD_WIDTH);
    if (triangles->cap + v.len > new_cap) {
      new_cap  = triangles->cap * 2 + ((v.len + SIMD_WIDTH + 1) / SIMD_WIDTH) * SIMD_WIDTH;
    }
    assert(new_cap % 8 == 0);
    f32 *new_data = (f32 *)unwrap_err(mem_alloc_aligned(TRIANGLES_ALLOCATION_SIZE(new_cap), SIMD_ALIGN, triangles->allocator));

    mem_tcopy(new_data + new_cap * 0, triangles->a_x, triangles->len);
    mem_tcopy(new_data + new_cap * 1, triangles->a_y, triangles->len);
    mem_tcopy(new_data + new_cap * 2, triangles->a_z, triangles->len);

    mem_tcopy(new_data + new_cap * 3, triangles->b_x, triangles->len);
    mem_tcopy(new_data + new_cap * 4, triangles->b_y, triangles->len);
    mem_tcopy(new_data + new_cap * 5, triangles->b_z, triangles->len);

    mem_tcopy(new_data + new_cap * 6, triangles->c_x, triangles->len);
    mem_tcopy(new_data + new_cap * 7, triangles->c_y, triangles->len);
    mem_tcopy(new_data + new_cap * 8, triangles->c_z, triangles->len);

    mem_tcopy((Triangle_AOS *)(new_data + new_cap * 9), triangles->aos, triangles->len);

    triangles->a_x = new_data + new_cap * 0;
    triangles->a_y = new_data + new_cap * 1;
    triangles->a_z = new_data + new_cap * 2;

    triangles->b_x = new_data + new_cap * 3;
    triangles->b_y = new_data + new_cap * 4;
    triangles->b_z = new_data + new_cap * 5;

    triangles->c_x = new_data + new_cap * 6;
    triangles->c_y = new_data + new_cap * 7;
    triangles->c_z = new_data + new_cap * 8;

    triangles->aos = (Triangle_AOS *)(new_data + new_cap * 9);

    triangles->cap = new_cap;
  }

  slice_iter_v(v, t, i, {
    triangles->a_x[triangles->len + i] = t.a.x;
    triangles->a_y[triangles->len + i] = t.a.y;
    triangles->a_z[triangles->len + i] = t.a.z;

    triangles->b_x[triangles->len + i] = t.b.x;
    triangles->b_y[triangles->len + i] = t.b.y;
    triangles->b_z[triangles->len + i] = t.b.z;

    triangles->c_x[triangles->len + i] = t.c.x;
    triangles->c_y[triangles->len + i] = t.c.y;
    triangles->c_z[triangles->len + i] = t.c.z;

    Vec3 edge1 = vec3_sub(vec3(t.b.x, t.b.y, t.b.z), vec3(t.a.x, t.a.y, t.a.z));
    Vec3 edge2 = vec3_sub(vec3(t.c.x, t.c.y, t.c.z), vec3(t.a.x, t.a.y, t.a.z));

    Vec2 delta_uv1 = vec2_sub(t.tex_coords_b, t.tex_coords_a);
    Vec2 delta_uv2 = vec2_sub(t.tex_coords_c, t.tex_coords_a);

    f32 d = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;
    if (abs_f32(d) < 0.0001f) {
      d = 1.0f;
    }

    f32 inv_d = 1.0f / d;

    Vec3 tangent   = vec3_normalize(vec3_scale(vec3_sub(vec3_scale(edge1, delta_uv2.y), vec3_scale(edge2, delta_uv1.y)), inv_d));
    Vec3 bitangent = vec3_normalize(vec3_scale(vec3_sub(vec3_scale(edge2, delta_uv1.x), vec3_scale(edge1, delta_uv2.x)), inv_d));

    triangles->aos[triangles->len + i] = (Triangle_AOS) {
      .shader       = t.shader,
      .normal       = vec3_normalize(vec3_cross(edge1, edge2)),
      .normal_a     = t.normal_a,
      .normal_b     = t.normal_b,
      .normal_c     = t.normal_c,
      .tex_coords_a = t.tex_coords_a,
      .tex_coords_b = t.tex_coords_b,
      .tex_coords_c = t.tex_coords_c,
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
    min(triangle->a.x, min(triangle->b.x, triangle->c.x)) - EPSILON,
    min(triangle->a.y, min(triangle->b.y, triangle->c.y)) - EPSILON,
    min(triangle->a.z, min(triangle->b.z, triangle->c.z)) - EPSILON,
  );
  aabb->max = vec3(
    max(triangle->a.x, max(triangle->b.x, triangle->c.x)) + EPSILON,
    max(triangle->a.y, max(triangle->b.y, triangle->c.y)) + EPSILON,
    max(triangle->a.z, max(triangle->b.z, triangle->c.z)) + EPSILON,
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

      aabb_i.min.data[axis] < aabb_j.min.data[axis];
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

  // fmt_printflnc("triangles: %d", triangles.len);
  // fmt_printflnc("depth:     %d", depth);
  // fmt_printflnc("internal:  %d", n_internal);
  // fmt_printflnc("leaves:    %d", n_leaves);
  // process_exit(0);

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
    if (slice.len <= per_child) {
      panic("");
    }

    isize split = bvh_partition_triangles(slice.len, per_child);
    // fmt_printflnc("n: %d, split: %d, per_child: %d", slice.len, split, per_child);

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

  // slice_iter_v(slice_array(Slice(Triangle_Slice), finished), tris, i, {
  //   if (i) {
  //     fmt_printc(", ");
  //   }
  //   fmt_printfc("%d", tris.len);
  // });
  // fmt_printlnc("");

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
