#include "raytracer.h"

#define WIDTH       (1024)
#define HEIGHT      (1024)
#define SAMPLES     (16)
#define MAX_BOUNCES 8
#define USE_BVH     1
#define USE_THREADS 1
#define N_THREADS   16

#define USE_CACHED_BVH 0

#define CHUNK_SIZE 32

#define CHUNKS_X ((WIDTH  + CHUNK_SIZE - 1) / CHUNK_SIZE)
#define CHUNKS_Y ((HEIGHT + CHUNK_SIZE - 1) / CHUNK_SIZE)
#define N_CHUNKS (CHUNKS_X * CHUNKS_Y)

internal thread_local u32 random_state = 0;

u32 rand_u32() {
  u32 state = random_state * 747796405u + 2891336453u;
  u32 word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  random_state = (word >> 22u) ^ word;
  return random_state;
}

f32 rand_f32() {
  return rand_u32() / (f32)U32_MAX;
}

f32 rand_f32_range(f32 min, f32 max) {
  return rand_f32() * (max - min) + min;
}

Vec3 rand_vec3() {
  loop {
    Vec3 p = vec3(
      rand_f32_range(-1, 1),
      rand_f32_range(-1, 1),
      rand_f32_range(-1, 1),
    );
    f32 lensq = vec3_length2(p);
    if (1e-40f < lensq && lensq <= 1) {
      return vec3_scale(p, 1.0 / sqrt_f32(lensq));
    }
  }
}

internal inline f32x8 ray_spheres_hit_8(Ray *ray, Spheres *spheres, isize offset) {
  f32x8 x = _mm256_load_ps(spheres->position_x + offset * 8);
  f32x8 y = _mm256_load_ps(spheres->position_y + offset * 8);
  f32x8 z = _mm256_load_ps(spheres->position_z + offset * 8);
  f32x8 r = _mm256_load_ps(spheres->radius     + offset * 8);

  f32x8 rox = _mm256_set1_ps(ray->position.x);
  f32x8 roy = _mm256_set1_ps(ray->position.y);
  f32x8 roz = _mm256_set1_ps(ray->position.z);

  f32x8 rdx = _mm256_set1_ps(ray->direction.x);
  f32x8 rdy = _mm256_set1_ps(ray->direction.y);
  f32x8 rdz = _mm256_set1_ps(ray->direction.z);

  f32x8 ox = rox - x;
  f32x8 oy = roy - y;
  f32x8 oz = roz - z;

  f32x8 a = _mm256_set1_ps(vec3_dot(ray->direction, ray->direction));
  f32x8 b = (ox * rdx + oy * rdy + oz * rdz) * 2.0f;
  f32x8 c = (ox * ox  + oy * oy  + oz * oz) - r * r;

  f32x8 d = b * b - 4.0f * a * c;
  f32x8 d_sqrt = _mm256_sqrt_ps(d);

  f32x8 hit_mask = _mm256_cmp_ps(d, _mm256_setzero_ps(), _CMP_LE_OQ);

  f32x8 distance = (-b - d_sqrt) / (2.0f * a);

  return _mm256_blendv_ps(distance, _mm256_set1_ps(F32_INFINITY), hit_mask);
}

internal inline Vec3x8 vec3x8_cross(Vec3x8 a, Vec3x8 b) {
  return (Vec3x8) {
    .x = a.y * b.z - a.z * b.y,
    .y = a.z * b.x - a.x * b.z,
    .z = a.x * b.y - a.y * b.x,
  };
}

internal inline Vec3x8 vec3x8_sub(Vec3x8 a, Vec3x8 b) {
  return (Vec3x8) {
    .x = a.x - b.x,
    .y = a.y - b.y,
    .z = a.z - b.z,
  };
}

internal inline Vec3x8 vec3x8_mul(Vec3x8 a, Vec3x8 b) {
  return (Vec3x8) {
    .x = a.x * b.x,
    .y = a.y * b.y,
    .z = a.z * b.z,
  };
}

internal inline f32x8 vec3x8_dot(Vec3x8 a, Vec3x8 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

internal void triangles_init(Triangles *triangles, isize len, isize cap, Allocator allocator) {
  if (cap % 8) {
    cap += 8 - cap % 8;
  }
  triangles->len       = len;
  triangles->cap       = cap;
  triangles->allocator = allocator;

  f32 *data = (f32 *)unwrap_err(mem_alloc_aligned(TRIANGLES_ALLOCATION_SIZE(cap), 32, allocator));

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
    isize new_cap = max(triangles->cap * 2, 8);
    if (triangles->cap + v.len > new_cap) {
      new_cap  = triangles->cap * 2 + ((v.len + 7) / 8) * 8;
    }
    assert(new_cap % 8 == 0);
    f32 *new_data = (f32 *)unwrap_err(mem_alloc_aligned(TRIANGLES_ALLOCATION_SIZE(new_cap), 32, triangles->allocator));

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

    triangles->aos[triangles->len + i] = (Triangle_AOS) {
      .shader       = t.shader,
      .normal_a     = t.normal_a,
      .normal_b     = t.normal_b,
      .normal_c     = t.normal_c,
      .tex_coords_a = t.tex_coords_a,
      .tex_coords_b = t.tex_coords_b,
      .tex_coords_c = t.tex_coords_c,
    };
  });
  triangles->len += v.len;
}

internal inline f32 min_f32x8(f32x8 vec, f32 epsilon, i32 *index) {
  // Set elements less than epsilon to +∞
  f32x8 inf = _mm256_set1_ps(F32_INFINITY);
  f32x8 nan_mask = _mm256_cmp_ps(vec, _mm256_set1_ps(epsilon), _CMP_GT_OQ);
  f32x8 sanitized_vec = _mm256_blendv_ps(inf, vec, nan_mask);

  f32x8 min1 = _mm256_min_ps(sanitized_vec, _mm256_permute_ps(sanitized_vec, 0b10110001));
  f32x8 min2 = _mm256_min_ps(min1, _mm256_permute_ps(min1, 0b01001110));
  f32x8 min3 = _mm256_min_ps(min2, _mm256_permute2f128_ps(min2, min2, 0b00000001));
  
  f32 min_value = _mm256_cvtss_f32(min3);

  f32x8 cmp  = _mm256_cmp_ps(sanitized_vec, min3, _CMP_EQ_OQ);
  i32   mask = _mm256_movemask_ps(cmp);
  *index     = __builtin_ctz(mask);

  return min_value;
}

internal inline b8 ray_triangles_hit_8(
  Ray       *ray,
  Triangles *triangles,
  isize      offset,
  Hit       *hit
) {
  const f32x8 epsilon  = _mm256_set1_ps( EPSILON);
  const f32x8 nepsilon = _mm256_set1_ps(-EPSILON);

  Vec3x8 ray_dir, ray_origin;
  ray_dir.x = _mm256_set1_ps(ray->direction.x);
  ray_dir.y = _mm256_set1_ps(ray->direction.y);
  ray_dir.z = _mm256_set1_ps(ray->direction.z);

  ray_origin.x = _mm256_set1_ps(ray->position.x);
  ray_origin.y = _mm256_set1_ps(ray->position.y);
  ray_origin.z = _mm256_set1_ps(ray->position.z);

  Vec3x8 a, b, c;
  a.x = _mm256_load_ps(triangles->a_x + offset);
  a.y = _mm256_load_ps(triangles->a_y + offset);
  a.z = _mm256_load_ps(triangles->a_z + offset);
  
  b.x = _mm256_load_ps(triangles->b_x + offset);
  b.y = _mm256_load_ps(triangles->b_y + offset);
  b.z = _mm256_load_ps(triangles->b_z + offset);
  
  c.x = _mm256_load_ps(triangles->c_x + offset);
  c.y = _mm256_load_ps(triangles->c_y + offset);
  c.z = _mm256_load_ps(triangles->c_z + offset);

  Vec3x8 edge1, edge2;
  edge1.x = b.x - a.x;
  edge1.y = b.y - a.y;
  edge1.z = b.z - a.z;

  edge2.x = c.x - a.x;
  edge2.y = c.y - a.y;
  edge2.z = c.z - a.z;

  Vec3x8 ray_cross_e2 = vec3x8_cross(ray_dir, edge2);

  f32x8     det = vec3x8_dot(edge1, ray_cross_e2);
  f32x8 inv_det = 1.0f / det;

  Vec3x8 s = vec3x8_sub(ray_origin, a);
  
  Vec3x8 s_cross_e1 = vec3x8_cross(s, edge1);

  f32x8 u = inv_det * vec3x8_dot(s, ray_cross_e2);
  f32x8 v = inv_det * vec3x8_dot(ray_dir, s_cross_e1);
  f32x8 t = inv_det * vec3x8_dot(edge2,   s_cross_e1);

  // (det > -epsilon && det < epsilon)
  f32x8 miss_mask_1 = _mm256_and_ps(
    _mm256_cmp_ps(det, nepsilon, _CMP_GT_OQ),
    _mm256_cmp_ps(det,  epsilon, _CMP_LT_OQ)
  );

  // (u < -epsilon || u > 1 + epsilon)
  f32x8 miss_mask_2 = _mm256_or_ps(
    _mm256_cmp_ps(u, nepsilon, _CMP_LT_OQ),
    _mm256_cmp_ps(u, _mm256_set1_ps(1 + EPSILON), _CMP_GT_OQ)
  );

  // (v < -epsilon || u + v > 1 + epsilon)
  f32x8 miss_mask_3 = _mm256_or_ps(
    _mm256_cmp_ps(v, nepsilon, _CMP_LT_OQ),
    _mm256_cmp_ps(u + v, _mm256_set1_ps(1 + EPSILON), _CMP_GT_OQ)
  );

  f32x8 miss_mask_4 = _mm256_cmp_ps(t, epsilon, _CMP_LT_OQ);

  f32x8 miss_mask =
    _mm256_or_ps(_mm256_or_ps(miss_mask_1, miss_mask_2), _mm256_or_ps(miss_mask_3, miss_mask_4));

  f32x8 distances = _mm256_blendv_ps(t, _mm256_set1_ps(F32_INFINITY), miss_mask);

  i32 triangle_index;
  f32 min = min_f32x8(distances, EPSILON, &triangle_index);

  if (min < hit->distance) {
    hit->distance = min;

    i32 t = triangle_index + offset;

    f32 t1 = u[triangle_index];
    f32 t2 = v[triangle_index];
    f32 t0 = 1 - t1 - t2;

    hit->point      = vec3_add(ray->position, vec3_scale(ray->direction, min));
    hit->normal     = vec3(
      .x = triangles->aos[t].normal_a.x * t0 + triangles->aos[t].normal_b.x * t1 + triangles->aos[t].normal_c.x * t2,
      .y = triangles->aos[t].normal_a.y * t0 + triangles->aos[t].normal_b.y * t1 + triangles->aos[t].normal_c.y * t2,
      .z = triangles->aos[t].normal_a.z * t0 + triangles->aos[t].normal_b.z * t1 + triangles->aos[t].normal_c.z * t2,
    );
    hit->tex_coords = vec2(
      .x = triangles->aos[t].tex_coords_a.x * t0 + triangles->aos[t].tex_coords_b.x * t1 + triangles->aos[t].tex_coords_c.x * t2,
      .y = triangles->aos[t].tex_coords_a.y * t0 + triangles->aos[t].tex_coords_b.y * t1 + triangles->aos[t].tex_coords_c.y * t2,
    );
    hit->shader     = triangles->aos[t].shader;

    Vec3 edge1 = vec3_sub(vec3(triangles->b_x[t], triangles->b_y[t], triangles->b_z[t]), vec3(triangles->a_x[t], triangles->a_y[t], triangles->a_z[t]));
    Vec3 edge2 = vec3_sub(vec3(triangles->c_x[t], triangles->c_y[t], triangles->c_z[t]), vec3(triangles->a_x[t], triangles->a_y[t], triangles->a_z[t]));

    Vec2 delta_uv1 = vec2_sub(triangles->aos[t].tex_coords_b, triangles->aos[t].tex_coords_a);
    Vec2 delta_uv2 = vec2_sub(triangles->aos[t].tex_coords_c, triangles->aos[t].tex_coords_a);

    f32 d = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;
    if (abs_f32(d) < EPSILON) {
      d = 1.0f;
    }

    f32 inv_d = 1.0f / d;

    Vec3 tangent = vec3_scale(vec3_sub(vec3_scale(edge1, delta_uv2.y), vec3_scale(edge2, delta_uv1.y)), inv_d);
    Vec3 bitangent = vec3_scale(vec3_sub(vec3_scale(edge2, delta_uv1.x), vec3_scale(edge1, delta_uv2.x)), inv_d);
    f32 tangent_w = (vec3_dot(vec3_cross(hit->normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;

    hit->tangent   = tangent;
    hit->bitangent = vec3_scale(bitangent, tangent_w);

    return true;
  }

  return false;
}

internal inline u8 ray_aabbs_hit_8(
  Ray   *ray,
  f32    t_min,
  f32    t_max,
  Vec3x8 mins,
  Vec3x8 maxs
) {
  Vec3x8 inv_dir = {
    .x = _mm256_set1_ps(1.0f / ray->direction.x),
    .y = _mm256_set1_ps(1.0f / ray->direction.y),
    .z = _mm256_set1_ps(1.0f / ray->direction.z),
  };
  Vec3x8 origin = {
    .x = _mm256_set1_ps(ray->position.x),
    .y = _mm256_set1_ps(ray->position.y),
    .z = _mm256_set1_ps(ray->position.z),
  };

  Vec3x8 t0s = vec3x8_mul(vec3x8_sub(mins, origin), inv_dir);
  Vec3x8 t1s = vec3x8_mul(vec3x8_sub(maxs, origin), inv_dir);

  Vec3x8 t_small = {
    .x = _mm256_min_ps(t0s.x, t1s.x),
    .y = _mm256_min_ps(t0s.y, t1s.y),
    .z = _mm256_min_ps(t0s.z, t1s.z),
  };

  Vec3x8 t_big = {
    .x = _mm256_max_ps(t0s.x, t1s.x),
    .y = _mm256_max_ps(t0s.y, t1s.y),
    .z = _mm256_max_ps(t0s.z, t1s.z),
  };

  f32x8 t_minv = _mm256_max_ps(_mm256_set1_ps(t_min), _mm256_max_ps(t_small.x, _mm256_max_ps(t_small.y, t_small.z)));
  f32x8 t_maxv = _mm256_min_ps(_mm256_set1_ps(t_max), _mm256_min_ps(t_big.x,   _mm256_min_ps(t_big.y,   t_big.z)));

  return _mm256_movemask_ps(_mm256_cmp_ps(t_minv, t_maxv, _CMP_LT_OQ));
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

#if USE_BVH
  #include "bvh.c"

  internal void ray_bvh_node_hit(Ray *ray, BVH *bvh, BVH_Node *node, Hit *hit, i32 depth) {
    u8 aabb_hits = ray_aabbs_hit_8(ray, EPSILON, F32_INFINITY, node->mins, node->maxs);
    u8 mask      = 1;

    for_range(offset, 0, 8) {
      if (aabb_hits & mask) {
        BVH_Index idx = node->children[offset];
        if (idx.leaf) {
          ray_triangles_hit_8(ray, &bvh->triangles, idx.index, hit);
        } else {
          ray_bvh_node_hit(ray, bvh, &IDX(bvh->nodes, idx.index), hit, depth + 1);
        }
      }
      mask <<= 1;
    }
  }

  internal void ray_bvh_hit(Ray *ray, BVH *bvh, Hit *hit) {
    if (bvh->root.leaf) {
      ray_triangles_hit_8(ray, &bvh->triangles, bvh->root.index, hit);
    } else {
      ray_bvh_node_hit(ray, bvh, &IDX(bvh->nodes, bvh->root.index), hit, 1);
    }
  }

  internal BVH_Index bvh_build(BVH *bvh, Triangle_Slice triangles) {
    if (triangles.len <= 8) {
      BVH_Index idx = { .index = bvh->triangles.len, .leaf = true, };
      triangles_append(&bvh->triangles, triangles);
      while (bvh->triangles.len % 8) {
        bvh->triangles.len += 1;
      }
      assert(bvh->triangles.len <= bvh->triangles.cap);
      return idx;
    }

    Triangle_Slice slices[8] = { triangles, };
    isize n_slices = 1;

    while (n_slices < 8) {
      isize _n_slices = n_slices;
      for_range(slice_i, 0, _n_slices) {
        Triangle_Slice slice = slices[slice_i];
        if (slice.len <= 8) {
          n_slices += 1;
          continue;
        }
        f32 min_surface_area = F32_INFINITY;
        i32 best_axis;
        for_range(axis, 0, 3) {
          sort_triangle_slice(slice, axis);
          AABB a, b;
          aabb_triangle_slice(slice_end(slice,   slice.len / 2), &a);
          aabb_triangle_slice(slice_start(slice, slice.len / 2), &b);
          f32 surface_area = aabb_surface_area(&a) + aabb_surface_area(&b);
          if (surface_area <= min_surface_area) {
            min_surface_area = surface_area;
            best_axis        = axis;
          }
        }

        if (best_axis != 2) {
          sort_triangle_slice(slice, best_axis);
        }

        slices[slice_i ] = slice_end(slice,   slice.len / 2);
        slices[n_slices] = slice_start(slice, slice.len / 2);
        n_slices += 1;
      }
    }
  
    BVH_Node node = {0};
    BVH_Index index = (BVH_Index) { .index = bvh->nodes.len, .leaf = false, };
    vector_append(&bvh->nodes, node);

    slice_iter_v(slice_array(Slice(Triangle_Slice), slices), tris, i, {
      AABB aabb;
      aabb_triangle_slice(tris, &aabb);

      node.mins.x[i] = aabb.min.x;
      node.mins.y[i] = aabb.min.y;
      node.mins.z[i] = aabb.min.z;

      node.maxs.x[i] = aabb.max.x;
      node.maxs.y[i] = aabb.max.y;
      node.maxs.z[i] = aabb.max.z;

      node.children[i] = bvh_build(bvh, tris);
    });

    IDX(bvh->nodes, index.index) = node;
    return index;
  }

  internal void bvh_init(BVH *bvh, Triangle_Slice src_triangles, Allocator allocator) {
    vector_init(&bvh->nodes, 0, 8, allocator);
    triangles_init(&bvh->triangles, 0, src_triangles.len * 8, allocator);
    bvh->root = bvh_build(bvh, src_triangles);
  }

#else
  Triangles scene_triangles;
#endif

Image background_image = {0};

Vec3 sample_background(Vec3 dir) {
  f32 invPi    = 1.0f / PI;
  f32 invTwoPi = 1.0f / (2.0f * PI);

  f32 u = 0.5f + atan2_f32(dir.x, dir.z) * invTwoPi;
  f32 v = 0.5f - asin_f32(dir.y) * invPi;

  i32 ui = u * background_image.width;
  i32 vi = v * background_image.height;

  ui = clamp(ui, 0, background_image.width);
  vi = clamp(vi, 0, background_image.height);

  Vec3 color = vec3(
    IDX(background_image.pixels, background_image.components * (ui + background_image.width * vi) + 0) / 255.0f,
    IDX(background_image.pixels, background_image.components * (ui + background_image.width * vi) + 1) / 255.0f,
    IDX(background_image.pixels, background_image.components * (ui + background_image.width * vi) + 2) / 255.0f,
  );

  return color;
}

internal inline void ray_triangles_hit(Ray *ray, Triangles *triangles, Hit *hit) {
  for_range(offset, 0, (triangles->len + 7) / 8) {
    ray_triangles_hit_8(ray, triangles, offset * 8, hit);
  }
}

internal inline void ray_spheres_hit(Ray *ray, Spheres *spheres, Hit *hit) {
  for_range(offset, 0, (spheres->len + 7) / 8) {
    f32x8 distances = ray_spheres_hit_8(ray, spheres, offset * 8);
  
    i32 idx;
    f32 new_min = min_f32x8(distances, EPSILON, &idx);
    if (new_min < hit->distance) {
      hit->distance = new_min;

      i32 s = idx + offset * 8;

      hit->point  = vec3_add(ray->position, vec3_scale(ray->direction, hit->distance));
      hit->normal = vec3_sub(hit->point, vec3(spheres->position_x[s], spheres->position_y[s], spheres->position_z[s]));
      hit->normal = vec3_scale(hit->normal, 1.0f / spheres->radius[s]);
      hit->shader = spheres->shader[s];
    }
  }
}

internal Vec3 sample_texture(Image *texture, Vec2 tex_coords) {
  isize u = tex_coords.x * texture->width;
  if (u >= texture->width) {
    u = texture->width - 1;
  }
  isize v = (1 - tex_coords.y) * texture->height;
  if (v >= texture->height) {
    v = texture->height - 1;
  }

  return vec3(
    IDX(texture->pixels, texture->components * (u + texture->width * v) + 0) / 255.0f,
    IDX(texture->pixels, texture->components * (u + texture->width * v) + 1) / 255.0f,
    IDX(texture->pixels, texture->components * (u + texture->width * v) + 2) / 255.0f,
  );
}

typedef struct {
  Vec3 albedo;
} Diffuse_Shader_Data;

internal void diffuse_shader_proc(rawptr _data, Shader_Input const *input, Shader_Output *output) {
  Diffuse_Shader_Data const *data = (Diffuse_Shader_Data *)_data;

  output->tint      = data->albedo;
  output->direction = vec3_normalize(vec3_add(rand_vec3(), input->normal));
}

typedef struct {
  Vec3 tint;
  f32  roughness;
} Metal_Shader_Data;

internal void metal_shader_proc(rawptr _data, Shader_Input const *input, Shader_Output *output) {
  Metal_Shader_Data const *data = (Metal_Shader_Data *)_data;

  output->tint      = data->tint;
  output->direction = vec3_normalize(
    vec3_add(
      vec3_reflect(input->direction, input->normal),
      vec3_scale(rand_vec3(), data->roughness)
    )
  );
}

enum {
  PBR_Type_Metal,
  PBR_Type_Diffuse,
  // PBR_Type_Dielectric,
};

typedef struct {
  Vec3   albedo, emission;
  f32    roughness;
  u8     type;
  Image *texture_albedo;
  Image *texture_normal;
  Image *texture_metal_roughness;
  Image *texture_emission;
} PBR_Shader_Data;

internal void pbr_shader_proc(rawptr _data, Shader_Input const *input, Shader_Output *output) {
  PBR_Shader_Data const *data = (PBR_Shader_Data *)_data;

  Color3 albedo = data->albedo;
  if (data->texture_albedo) {
    albedo = vec3_mul(data->albedo, sample_texture(data->texture_albedo, input->tex_coords));
  }
  output->tint = albedo;

  Color3 emmission = data->emission;
  if (data->texture_emission) {
    emmission = vec3_mul(emmission, sample_texture(data->texture_emission, input->tex_coords));
  }
  output->emission = emmission;

  u8  type      = data->type;
  f32 roughness = data->roughness;
  if (data->texture_metal_roughness) {
    Vec3 mr = sample_texture(data->texture_metal_roughness, input->tex_coords);
    if (mr.b > 0.5f) {
      type = PBR_Type_Metal;
    } else {
      type = PBR_Type_Diffuse;
    }
    roughness *= mr.g;
  }

  Vec3 normal = input->normal;
  if (data->texture_normal) {
    Vec3 v = sample_texture(data->texture_normal, input->tex_coords);
         v = vec3_add(vec3_scale(v, 2.0), vec3(-1, -1, -1));

    Vec3 t = vec3_normalize(input->tangent);
    Vec3 b = vec3_normalize(input->bitangent);
    Vec3 n = input->normal;

    normal = vec3_normalize(
      vec3(
        .x = v.x * t.x + v.y * b.x + v.z * n.x + n.x * 0.5f,
        .y = v.x * t.y + v.y * b.y + v.z * n.y + n.y * 0.5f,
        .z = v.x * t.z + v.y * b.z + v.z * n.z + n.z * 0.5f,
      )
    );
  }

  switch (type) {
  CASE PBR_Type_Metal:
    Vec3 dir          = vec3_reflect(input->direction, normal);
         dir          = vec3_add(dir, vec3_scale(rand_vec3(), roughness));
    output->direction = vec3_normalize(dir);
  CASE PBR_Type_Diffuse:
    output->direction = vec3_normalize(vec3_add(rand_vec3(), normal));
  }
}

internal void debug_shader_proc(rawptr _data, Shader_Input const *input, Shader_Output *output) {
  PBR_Shader_Data const *data = (PBR_Shader_Data *)_data;

  Vec3 normal = input->normal;
  if (data->texture_normal) {
    Vec3 v = sample_texture(data->texture_normal, input->tex_coords);
         v = vec3_add(vec3_scale(v, 2.0), vec3(-1, -1, -1));

    Vec3 t = vec3_normalize(input->tangent);
    Vec3 b = vec3_normalize(input->bitangent);
    Vec3 n = input->normal;

    normal = vec3_normalize(
      vec3(
        .x = v.x * t.x + v.y * b.x + v.z * n.x + n.x * 0.5f,
        .y = v.x * t.y + v.y * b.y + v.z * n.y + n.y * 0.5f,
        .z = v.x * t.z + v.y * b.z + v.z * n.z + n.z * 0.5f,
      )
    );
  }

  // output->emission  = vec3_broadcast(normal.y * 0.5 + 0.5);
  // output->emission  = vec3_broadcast(vec3_dot(normal, vec3(0, 1, 0)));
  output->emission  = vec3_add(vec3_scale(normal, 0.5), vec3_broadcast(0.5f));
  output->terminate = true;
}

Color3 cast_ray(BVH *bvh, Ray ray) {
  Color3 accumulated_tint = vec3(1, 1, 1);
  Color3 emission         = {0};
  for_range(i, 0, MAX_BOUNCES) {
    Hit hit = { .distance = F32_INFINITY, };
    #if USE_BVH
      ray_bvh_hit(&ray, bvh, &hit);
    #else
      ray_triangles_hit(&ray, &scene_triangles, &hit);
    #endif
    if (hit.distance != F32_INFINITY) {
      hit.normal   = vec3_normalize(hit.normal);
      ray.position = vec3_add(hit.point, vec3_scale(hit.normal, EPSILON));

      Shader_Input shader_input = {
        .direction  = ray.direction,
        .normal     = hit.normal,
        .tangent    = hit.tangent,
        .bitangent  = hit.bitangent,
        .position   = hit.point,
        .tex_coords = hit.tex_coords,
      };
      Shader_Output shader_output = {0};

      hit.shader.proc(hit.shader.data, &shader_input, &shader_output);

      ray.direction    = shader_output.direction;
      emission         = vec3_add(emission, vec3_mul(shader_output.emission, accumulated_tint));
      accumulated_tint = vec3_mul(accumulated_tint, shader_output.tint);

      if (shader_output.terminate) {
        break;
      }
    } else {
      return vec3_add(vec3_mul(sample_background(ray.direction), accumulated_tint), emission);
    }
  }
  return emission;
}

internal inline f32 aces_f32(f32 x) {
  const f32 a = 2.51;
  const f32 b = 0.03;
  const f32 c = 2.43;
  const f32 d = 0.59;
  const f32 e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

internal inline f32x8 fract_f32x8(f32x8 x) { return _mm256_sub_ps(x, _mm256_floor_ps(x)); }

internal inline f32x8 hash12x8(Vec2x8 p) {
  Vec3x8 p3 = {
    fract_f32x8(p.x * 0.1031f),
    fract_f32x8(p.y * 0.1031f),
    fract_f32x8(p.x * 0.1031f),
  };
  f32x8 add = _mm256_set1_ps(33.33f);
  f32x8 dot = p3.x * (p3.y + add) + p3.y * (p3.z + add) + p3.z * (p3.x + add);
  
  return fract_f32x8((p3.x + p3.y + dot * 2.0f) * (p3.z + dot));
}

typedef struct {
  Matrix_3x3 matrix;
  Vec3       position;
  f32        fov, focal_length;
} Camera;

typedef struct {
  Camera      camera;
  Image       image;
  BVH        *bvh;
  _Atomic i32 threads_done, current_chunk;
} Rendering_Context;

void render_thread_proc(rawptr arg) {
  Rendering_Context *ctx = (Rendering_Context *)arg;

  Camera camera = ctx->camera;

  loop {
    isize c = atomic_fetch_add(&ctx->current_chunk, 1);
    if (c >= N_CHUNKS) {
      atomic_fetch_add(&ctx->threads_done, 1);
      return;
    }

    isize start_x = (c % CHUNKS_X) * CHUNK_SIZE;
    isize start_y = (c / CHUNKS_X) * CHUNK_SIZE;

    for_range(y, start_y, start_y + CHUNK_SIZE) {
      if (y >= HEIGHT) {
        break;
      }

      for_range(x, start_x, start_x + CHUNK_SIZE) {
        if (x >= WIDTH) {
          break;
        }

        Color3 color = vec3(0, 0, 0);

        f32x8 sample_indices = { 0, 1, 2, 3, 4, 5, 6, 7, };

        for_range(sample_batch, 0, (SAMPLES + 7) / 8) {
          f32x8 rand_a = hash12x8((Vec2x8) {
            .x = _mm256_set1_ps((f32)x * 50.0f) + sample_indices,
            .y = _mm256_set1_ps((f32)y),
          });
          f32x8 rand_b = hash12x8((Vec2x8) {
            .x = _mm256_set1_ps((f32)x * 50.0f) + sample_indices,
            .y = _mm256_set1_ps((f32)y),
          });

          Vec2x8 uvs = {
            .x = ((f32)x + rand_a - 0.5) / WIDTH  - 0.5,
            .y = ((f32)y + rand_b - 0.5) / HEIGHT - 0.5,
          };
          Vec3x8 directions = {
            .x =  uvs.x * ((f32)WIDTH / HEIGHT),
            .y = -uvs.y,
            .z = _mm256_set1_ps(-camera.focal_length),
          };

          f32x8 inv_lengths = _mm256_rsqrt_ps(
            directions.x * directions.x +
            directions.y * directions.y +
            directions.z * directions.z
          );

          directions = (Vec3x8) {
            .x = camera.matrix.rows[0][0] * directions.x + camera.matrix.rows[0][1] * directions.y + camera.matrix.rows[0][2] * directions.z,
            .y = camera.matrix.rows[1][0] * directions.x + camera.matrix.rows[1][1] * directions.y + camera.matrix.rows[1][2] * directions.z,
            .z = camera.matrix.rows[2][0] * directions.x + camera.matrix.rows[2][1] * directions.y + camera.matrix.rows[2][2] * directions.z,
          };

          directions.x *= inv_lengths;
          directions.y *= inv_lengths;
          directions.z *= inv_lengths;

          f32 directions_x[8] __attribute__((aligned(32)));
          f32 directions_y[8] __attribute__((aligned(32)));
          f32 directions_z[8] __attribute__((aligned(32)));

          _mm256_store_ps(directions_x, directions.x);
          _mm256_store_ps(directions_y, directions.y);
          _mm256_store_ps(directions_z, directions.z);

          for_range(sample, 0, 8) {
            if (sample + sample_batch * 8 >= SAMPLES) {
              break;
            }
            Ray r = {
              .position  = camera.position,
              .direction = vec3(directions_x[sample], directions_y[sample], directions_z[sample]),
            };
            color = vec3_add(color, cast_ray(ctx->bvh, r));
          }
          sample_indices += _mm256_set1_ps(8);
        }

        color = vec3_scale(color, 1.0f / SAMPLES);
        // color = vec3(
        //   aces_f32(color.r),
        //   aces_f32(color.g),
        //   aces_f32(color.b),
        // );
        color = vec3(
          clamp(color.r, 0, 1),
          clamp(color.g, 0, 1),
          clamp(color.b, 0, 1),
        );
        // color = vec3(
        //   pow_f32(color.r, 1.0f / 2.2f),
        //   pow_f32(color.g, 1.0f / 2.2f),
        //   pow_f32(color.b, 1.0f / 2.2f),
        // );
        color = vec3_scale(color, 255.999f);

        IDX(ctx->image.pixels, 3 * (x + y * WIDTH) + 0) = (u8)color.data[0];
        IDX(ctx->image.pixels, 3 * (x + y * WIDTH) + 1) = (u8)color.data[1];
        IDX(ctx->image.pixels, 3 * (x + y * WIDTH) + 2) = (u8)color.data[2];
      }
    }
  }
}

internal void load_texture(String path, Image *texture) {
  b8 ok = png_load_bytes(
    unwrap_err(read_entire_file_path(path, context.allocator)),
    texture,
    context.allocator
  );
  if (!ok) {
    fmt_eprintflnc("Failed to load texture: '%S'", path);
    process_exit(1);
  }
}

i32 main() {
  context.logger = (Logger) {0};

  Image image = {
    .components = 3,
    .pixel_type = PT_u8,
    .width      = WIDTH,
    .height     = HEIGHT,
  };
  image.pixels = slice_make_aligned(Byte_Slice, WIDTH * HEIGHT * 3, 64, context.allocator);

  Image texture_albedo          = {0};
  Image texture_metal_roughness = {0};
  Image texture_emission        = {0};
  Image texture_normal          = {0};

  load_texture(LIT("background.png"), &background_image);

  load_texture(LIT("helmet_albedo.png"),   &texture_albedo);
  load_texture(LIT("helmet_normal.png"),   &texture_normal);
  load_texture(LIT("helmet_emission.png"), &texture_emission);
  load_texture(LIT("helmet_mr.png"),       &texture_metal_roughness);

  f32 angle    = PI * 0.125f;
  f32 distance = 2;

  Camera camera;
  camera.position     = vec3(sin_f32(angle) * distance, -0.2f, cos_f32(angle) * distance);
  camera.matrix       = matrix_3x3_rotate(vec3(0, 1, 0), angle);
  camera.fov          = PI / 2.0f;
  camera.focal_length = 0.5f / atan_f32(camera.fov * 0.5f);

  isize n_triangles = 0;

  BVH bvh;

  #if USE_CACHED_BVH
    Fd bvh_file = unwrap_err(file_open(LIT("test.bvh"), FP_Read));
    File_Info fi;
    OS_Error err = file_stat(bvh_file, &fi);
    assert(err == OSE_None);
    Byte_Slice data = slice_make_aligned(Byte_Slice, fi.size, 32, context.allocator);
    isize n_read = unwrap_err(file_read(bvh_file, data));
    assert(n_read == fi.size);
    b8 ok = bvh_load_bytes(data, &bvh);
    assert(ok);

    n_triangles = bvh.triangles.len;
  #else
    Byte_Slice data = unwrap_err(read_entire_file_path(LIT("helmet.obj"), context.allocator));
    Obj_File obj = {0};
    obj_load(bytes_to_string(data), &obj, context.allocator);

    Triangle_Slice triangles = slice_make(Triangle_Slice, obj.triangles.len, context.allocator);

    Shader shader = {
      .data = &(PBR_Shader_Data) {
        .albedo                  = vec3(1, 1, 1),
        .emission                = vec3(1, 1, 1),
        .type                    = PBR_Type_Metal,
        .roughness               = 1.0f,
        .texture_albedo          = &texture_albedo,
        .texture_metal_roughness = &texture_metal_roughness,
        .texture_emission        = &texture_emission,
        .texture_normal          = &texture_normal,
      },
      .proc = pbr_shader_proc,
    };

    slice_iter_v(obj.triangles, t, i, {
      IDX(triangles, i) = (Triangle) {
        .a            = t.a.position,
        .b            = t.b.position,
        .c            = t.c.position,
        .normal_a     = t.a.normal,
        .normal_b     = t.b.normal,
        .normal_c     = t.c.normal,
        .tex_coords_a = t.a.tex_coords,
        .tex_coords_b = t.b.tex_coords,
        .tex_coords_c = t.c.tex_coords,
        .shader       = shader,
      };
    });
    #if USE_BVH
      Timestamp bvh_start_time = time_now();
      bvh_init(&bvh, triangles, context.allocator);
      fmt_printflnc("Bvh generated in %dms", time_since(bvh_start_time) / Millisecond);

      Fd bvh_file = unwrap_err(file_open(LIT("test.bvh"), FP_Truncate | FP_Write | FP_Create));
      Writer bvh_out = writer_from_handle(bvh_file);
      bvh_save_writer(&bvh_out, &bvh);
    #else
      triangles_init(&scene_triangles, 0, triangles.len, context.allocator);
      triangles_append(&scene_triangles, triangles);
    #endif

    n_triangles = triangles.len;
  #endif

  fmt_printflnc("Width:     %d", WIDTH);
  fmt_printflnc("Height:    %d", HEIGHT);
  fmt_printflnc("Samples:   %d", SAMPLES);
  fmt_printflnc("Bounces:   %d", MAX_BOUNCES);
  if (USE_THREADS) {
    fmt_printflnc("Threads:   %d", N_THREADS);
  }
#if USE_BVH
    fmt_printflnc("BVH-Nodes: %d", bvh.nodes.len);
#endif
  fmt_printflnc("Triangles: %d", n_triangles);

  fmt_printlnc("");

  Timestamp start_time = time_now();

  Rendering_Context rendering_context = {
    .bvh    = &bvh,
    .camera = camera,
    .image  = image,
  };

  #if USE_THREADS
    for_range(i, 0, N_THREADS) {
      thread_create(render_thread_proc, &rendering_context, THREAD_STACK_DEFAULT, THREAD_TLS_DEFAULT);
    }

    while (rendering_context.threads_done < N_THREADS) {
      isize  c   = rendering_context.current_chunk;
      String bar = LIT("====================");
      f32    p   = c / (f32)((i32)N_CHUNKS);
      if (p > 1.0f) {
        p = 1.0f;
      }
      fmt_printfc("\r[%-20S] %d%%", slice_end(bar, p * 20), (i32)(100 * p));
      time_sleep(Millisecond * 500);
    }

    fmt_printlnc("\r[====================] 100%");
  #else 
    render_thread_proc(nil);
  #endif

  Duration time = time_since(start_time);
  fmt_printflnc("%dms", (isize)(time / Millisecond));
  fmt_printflnc("%d samples/second", (isize)(WIDTH * HEIGHT * SAMPLES / (f64)((f64)time / Second)));

  Fd output_file = unwrap_err(file_open(LIT("output.png"), FP_Read_Write | FP_Create | FP_Truncate));
  Writer output_writer = writer_from_handle(output_file);
  assert(png_save_writer(&output_writer, &image));
  file_close(output_file);
}
