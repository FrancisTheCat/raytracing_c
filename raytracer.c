#include "raytracer.h"

#include "codin/codin.h"
#include "codin/image.h"
#include "codin/linalg.h"
#include "codin/obj.h"
#include "codin/os.h"
#include "codin/sort.h"
#include "codin/strconv.h"
#include "codin/thread.h"
#include "codin/time.h"

#include "scene.h"
#include "scene.c"

#include "stdatomic.h"

#if SIMD_WIDTH == 8

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

internal inline void ray_spheres_hit_8(
  Ray     const *ray,
  Spheres const *spheres,
  isize          offset,
  Hit           *hit
) {
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

  f32x8 distances = (-b - d_sqrt) / (2.0f * a);

  distances =  _mm256_blendv_ps(distances, _mm256_set1_ps(F32_INFINITY), hit_mask);
  
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

internal inline b8 ray_triangles_hit_8(
  Ray       const *ray,
  Triangles const *triangles,
  isize            offset,
  Hit             *hit
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

  // (u < -epsilon || u > 1 + epsilon)
  f32x8 miss_mask_1 = _mm256_or_ps(
    _mm256_cmp_ps(u, nepsilon, _CMP_LT_OQ),
    _mm256_cmp_ps(u, _mm256_set1_ps(1 + EPSILON), _CMP_GT_OQ)
  );

  // (v < -epsilon || u + v > 1 + epsilon)
  f32x8 miss_mask_2 = _mm256_or_ps(
    _mm256_cmp_ps(v, nepsilon, _CMP_LT_OQ),
    _mm256_cmp_ps(u + v, _mm256_set1_ps(1 + EPSILON), _CMP_GT_OQ)
  );

  f32x8 miss_mask_3 = _mm256_cmp_ps(t, epsilon, _CMP_LT_OQ);

  f32x8 miss_mask =
    _mm256_or_ps(miss_mask_1, _mm256_or_ps(miss_mask_2, miss_mask_3));

  f32x8 distances = _mm256_blendv_ps(t, _mm256_set1_ps(F32_INFINITY), miss_mask);

  i32 triangle_index;
  f32 min = min_f32x8(distances, 0, &triangle_index);

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

    hit->normal_geo = triangles->aos[t].normal;
    hit->tangent    = triangles->aos[t].tangent;
    hit->bitangent  = triangles->aos[t].bitangent;

    return true;
  }

  return false;
}

internal inline void ray_aabbs_hit_8(
  Ray   *ray,
  f32    t_min,
  f32    t_max,
  Vec3x8 mins,
  Vec3x8 maxs,
  f32   *distances
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

  f32x8 miss_mask = _mm256_cmp_ps(t_minv, t_maxv, _CMP_GE_OQ);
  f32x8 distances_v = _mm256_blendv_ps(t_minv, _mm256_set1_ps(F32_INFINITY), miss_mask);
  _mm256_store_ps(distances, distances_v);
}

#endif

#if SIMD_WIDTH == 16

internal inline f32 min_f32x16(f32x16 vec, f32 epsilon, i32 *index) {
  __mmask16 epsilon_mask = _mm512_cmp_ps_mask(vec, _mm512_set1_ps(epsilon), _CMP_GT_OQ);
            vec          = _mm512_mask_blend_ps(epsilon_mask, vec, _mm512_set1_ps(F32_INFINITY));
  f32       min_value    = _mm512_reduce_min_ps(vec);
  i32       mask         = _mm512_cmp_ps_mask(vec, _mm512_set1_ps(min_value), _CMP_EQ_OQ);
           *index        = __builtin_ctz(mask);

  return min_value;
}

internal inline b8 ray_triangles_hit_16(
  Ray       const *ray,
  Triangles const *triangles,
  isize            offset,
  Hit             *hit
) {
  const f32x16 epsilon  = _mm512_set1_ps( EPSILON);
  const f32x16 nepsilon = _mm512_set1_ps(-EPSILON);

  Vec3x16 ray_dir, ray_origin;
  ray_dir.x = _mm512_set1_ps(ray->direction.x);
  ray_dir.y = _mm512_set1_ps(ray->direction.y);
  ray_dir.z = _mm512_set1_ps(ray->direction.z);

  ray_origin.x = _mm512_set1_ps(ray->position.x);
  ray_origin.y = _mm512_set1_ps(ray->position.y);
  ray_origin.z = _mm512_set1_ps(ray->position.z);

  Vec3x16 a, b, c;
  a.x = _mm512_load_ps(triangles->a_x + offset);
  a.y = _mm512_load_ps(triangles->a_y + offset);
  a.z = _mm512_load_ps(triangles->a_z + offset);
  
  b.x = _mm512_load_ps(triangles->b_x + offset);
  b.y = _mm512_load_ps(triangles->b_y + offset);
  b.z = _mm512_load_ps(triangles->b_z + offset);
  
  c.x = _mm512_load_ps(triangles->c_x + offset);
  c.y = _mm512_load_ps(triangles->c_y + offset);
  c.z = _mm512_load_ps(triangles->c_z + offset);

  Vec3x16 edge1, edge2;
  edge1.x = b.x - a.x;
  edge1.y = b.y - a.y;
  edge1.z = b.z - a.z;

  edge2.x = c.x - a.x;
  edge2.y = c.y - a.y;
  edge2.z = c.z - a.z;

  Vec3x16 ray_cross_e2 = vec3x16_cross(ray_dir, edge2);

  f32x16     det = vec3x16_dot(edge1, ray_cross_e2);
  f32x16 inv_det = 1.0f / det;

  Vec3x16 s = vec3x16_sub(ray_origin, a);
  
  Vec3x16 s_cross_e1 = vec3x16_cross(s, edge1);

  f32x16 u = inv_det * vec3x16_dot(s, ray_cross_e2);
  f32x16 v = inv_det * vec3x16_dot(ray_dir, s_cross_e1);
  f32x16 t = inv_det * vec3x16_dot(edge2,   s_cross_e1);

  // (u < -epsilon || u > 1 + epsilon)
  __mmask16 miss_mask_1 = 
    _mm512_cmp_ps_mask(u, nepsilon, _CMP_LT_OQ) |
    _mm512_cmp_ps_mask(u, _mm512_set1_ps(1 + EPSILON), _CMP_GT_OQ);

  // (v < -epsilon || u + v > 1 + epsilon)
  __mmask16 miss_mask_2 = 
    _mm512_cmp_ps_mask(v, nepsilon, _CMP_LT_OQ) |
    _mm512_cmp_ps_mask(u + v, _mm512_set1_ps(1 + EPSILON), _CMP_GT_OQ);

  __mmask16 miss_mask_3 = _mm512_cmp_ps_mask(t, epsilon, _CMP_LT_OQ);

  __mmask16 miss_mask = miss_mask_1 | miss_mask_2 | miss_mask_3;

  f32x16 distances = _mm512_mask_blend_ps(miss_mask, t, _mm512_set1_ps(F32_INFINITY));

  i32 triangle_index;
  f32 min = min_f32x16(distances, 0, &triangle_index);

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

    hit->normal_geo = triangles->aos[t].normal;
    hit->tangent    = triangles->aos[t].tangent;
    hit->bitangent  = triangles->aos[t].bitangent;

    return true;
  }

  return false;
}

internal inline void ray_aabbs_hit_16(
  Ray   *ray,
  f32    t_min,
  f32    t_max,
  Vec3x16 mins,
  Vec3x16 maxs
) {
  Vec3x16 inv_dir = {
    .x = _mm512_set1_ps(1.0f / ray->direction.x),
    .y = _mm512_set1_ps(1.0f / ray->direction.y),
    .z = _mm512_set1_ps(1.0f / ray->direction.z),
  };
  Vec3x16 origin = {
    .x = _mm512_set1_ps(ray->position.x),
    .y = _mm512_set1_ps(ray->position.y),
    .z = _mm512_set1_ps(ray->position.z),
  };

  Vec3x16 t0s = vec3x16_mul(vec3x16_sub(mins, origin), inv_dir);
  Vec3x16 t1s = vec3x16_mul(vec3x16_sub(maxs, origin), inv_dir);

  Vec3x16 t_small = {
    .x = _mm512_min_ps(t0s.x, t1s.x),
    .y = _mm512_min_ps(t0s.y, t1s.y),
    .z = _mm512_min_ps(t0s.z, t1s.z),
  };

  Vec3x16 t_big = {
    .x = _mm512_max_ps(t0s.x, t1s.x),
    .y = _mm512_max_ps(t0s.y, t1s.y),
    .z = _mm512_max_ps(t0s.z, t1s.z),
  };

  f32x16 t_minv = _mm512_max_ps(_mm512_set1_ps(t_min), _mm512_max_ps(t_small.x, _mm512_max_ps(t_small.y, t_small.z)));
  f32x16 t_maxv = _mm512_min_ps(_mm512_set1_ps(t_max), _mm512_min_ps(t_big.x,   _mm512_min_ps(t_big.y,   t_big.z)));

  __mmask16 miss_mask   = _mm512_cmp_ps_mask(t_minv, t_maxv, _CMP_GE_OQ);
  f32x16    distances_v = _mm512_mask_blend_ps(miss_mask, t_minv, _mm512_set1_ps(F32_INFINITY));
  _mm512_store_ps(distances, distances_v);
}

internal inline void ray_spheres_hit_16(
  Ray     const *ray,
  Spheres const *spheres,
  isize          offset,
  Hit           *hit
) {
  f32x16 x = _mm512_load_ps(spheres->position_x + offset * 8);
  f32x16 y = _mm512_load_ps(spheres->position_y + offset * 8);
  f32x16 z = _mm512_load_ps(spheres->position_z + offset * 8);
  f32x16 r = _mm512_load_ps(spheres->radius     + offset * 8);

  f32x16 rox = _mm512_set1_ps(ray->position.x);
  f32x16 roy = _mm512_set1_ps(ray->position.y);
  f32x16 roz = _mm512_set1_ps(ray->position.z);

  f32x16 rdx = _mm512_set1_ps(ray->direction.x);
  f32x16 rdy = _mm512_set1_ps(ray->direction.y);
  f32x16 rdz = _mm512_set1_ps(ray->direction.z);

  f32x16 ox = rox - x;
  f32x16 oy = roy - y;
  f32x16 oz = roz - z;

  f32x16 a = _mm512_set1_ps(vec3_dot(ray->direction, ray->direction));
  f32x16 b = (ox * rdx + oy * rdy + oz * rdz) * 2.0f;
  f32x16 c = (ox * ox  + oy * oy  + oz * oz) - r * r;

  f32x16 d = b * b - 4.0f * a * c;
  f32x16 d_sqrt = _mm512_sqrt_ps(d);

  __mmask16 hit_mask = _mm512_cmp_ps_mask(d, _mm512_setzero_ps(), _CMP_LE_OQ);

  f32x16 distances = (-b - d_sqrt) / (2.0f * a);

  distances = _mm512_mask_blend_ps(hit_mask, distances, _mm512_set1_ps(F32_INFINITY));
  
  i32 idx;
  f32 new_min = min_f32x16(distances, EPSILON, &idx);
  if (new_min < hit->distance) {
    hit->distance = new_min;

    i32 s = idx + offset * 8;

    hit->point  = vec3_add(ray->position, vec3_scale(ray->direction, hit->distance));
    hit->normal = vec3_sub(hit->point, vec3(spheres->position_x[s], spheres->position_y[s], spheres->position_z[s]));
    hit->normal = vec3_scale(hit->normal, 1.0f / spheres->radius[s]);
    hit->shader = spheres->shader[s];
  }
}

#endif

internal void ray_bvh_node_hit(Ray *ray, Scene const *scene, BVH_Node *node, Hit *hit) {
  f32 distances[SIMD_WIDTH] __attribute__((aligned(SIMD_ALIGN)));
  ray_aabbs_hit_SIMD(ray, EPSILON, hit->distance, node->mins, node->maxs, &distances[0]);

  // NOTE(Franz): this depth sorting is, tho substantially more complicated than the naive approach
  // the idea is that we do the near collisions first, which we can then compare to the upper bound
  // for the distance given by the aabb hit check.
  for_range(i, 0, SIMD_WIDTH) {
    f32 min_distance = F32_INFINITY;
    i32 min_index    = -1;

    for_range(j, 0, SIMD_WIDTH) {
      if (distances[j] < min_distance) {
        min_distance = distances[j];
        min_index    = j;
      }
    }

    if (min_index == -1 || min_distance >= hit->distance) {
      return;
    }

    BVH_Index idx = node->children[min_index];
    if (idx.leaf) {
      ray_triangles_hit_SIMD(ray, &scene->triangles, idx.index, hit);
    } else {
      ray_bvh_node_hit(ray, scene, &IDX(scene->bvh.nodes, idx.index), hit);
    }
    
    distances[min_index] = F32_INFINITY;
  }
}

internal inline void ray_spheres_hit(Ray *ray, Spheres *spheres, Hit *hit) {
  for_range(offset, 0, (spheres->len + SIMD_WIDTH - 1) / SIMD_WIDTH) {
    ray_spheres_hit_SIMD(ray, spheres, offset * SIMD_WIDTH, hit);
  }
}

internal inline void ray_triangles_hit(Ray const *ray, Triangles const *triangles, Hit *hit) {
  for_range(offset, 0, (triangles->len + SIMD_WIDTH - 1) / SIMD_WIDTH) {
    ray_triangles_hit_SIMD(ray, triangles, offset * SIMD_WIDTH, hit);
  }
}

internal void ray_scene_hit(Ray *ray, Scene const *scene, Hit *hit) {
#if 0
  ray_triangles_hit(ray, &scene->triangles, hit);
#else
  if (scene->bvh.root.leaf) {
    ray_triangles_hit_SIMD(ray, &scene->triangles, scene->bvh.root.index, hit);
  } else {
    ray_bvh_node_hit(ray, scene, &IDX(scene->bvh.nodes, scene->bvh.root.index), hit);
  }
#endif
}

internal Color3 cast_ray(Scene const *scene, Ray ray, isize max_bounces) {
  Color3 accumulated_tint = vec3(1, 1, 1);
  Color3 emission         = {0};

  Shader_Output shader_output = {0};
  Shader_Input  shader_input  = {0};

  for_range(i, 0, max_bounces) {
    Hit hit = { .distance = F32_INFINITY, };
    ray_scene_hit(&ray, scene, &hit);
    if (hit.distance != F32_INFINITY) {
      if (
        vec3_dot(hit.normal_geo, ray.direction) > 0 ||
        vec3_dot(hit.normal,     ray.direction) > 0
      ) {
        ray.position = vec3_add(hit.point, vec3_scale(ray.direction, EPSILON));
        continue;
      }

      shader_input = (Shader_Input) {
        .direction  = ray.direction,
        .normal     = vec3_normalize(hit.normal),
        .normal_geo = hit.normal_geo,
        .tangent    = hit.tangent,
        .bitangent  = hit.bitangent,
        .position   = hit.point,
        .tex_coords = hit.tex_coords,
      };
      shader_output = (Shader_Output) {0};

      hit.shader.proc(hit.shader.data, &shader_input, &shader_output);

      emission = vec3_add(emission, vec3_mul(shader_output.emission, accumulated_tint));

      if (shader_output.terminate) {
        break;
      }

      ray.direction    = shader_output.direction;
      accumulated_tint = vec3_mul(accumulated_tint, shader_output.tint);

      // NOTE(Franz): normal interpolation/mapping shenanigans
      // essentially the idea is that if the mapped normal results in a reflection
      // going into the model we make sure it actually goes in and then ignore the
      // backface collision on the othe side. Maybe this should be the shaders
      // responsibility
      f32 position_bias = (0.5f - (vec3_dot(hit.normal_geo, shader_output.direction) < 0)) * 2.0f * EPSILON;
      ray.position = vec3_add(hit.point, vec3_scale(hit.normal_geo, position_bias));
    } else {
      return vec3_add(vec3_mul(scene->background.proc(scene->background.data, ray.direction), accumulated_tint), emission);
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
  return (x * (a * x + b)) / (x * (c * x + d) + e);
}

internal inline f32 reinhard_f32(f32 x) {
  const f32 L_white = 4;
  return clamp((x * (1.0 + x / (L_white * L_white))) / (1.0 + x), 0, 1);
}

internal inline Color3 tonemap(Color3 x) {
  return vec3(
    aces_f32(x.r),
    aces_f32(x.g),
    aces_f32(x.b),
  );
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

extern void render_thread_proc(Rendering_Context *ctx) {
  random_state = time_now();

  Camera camera = ctx->scene->camera;

  #define CHUNK_SIZE 32

  isize n_chunks, chunks_x, chunks_y, width, height, samples, max_bounces;
  samples     = ctx->samples;
  max_bounces = ctx->max_bounces;
  width       = ctx->image.width;
  height      = ctx->image.height;
  chunks_x    = ((width  + CHUNK_SIZE - 1) / CHUNK_SIZE);
  chunks_y    = ((height + CHUNK_SIZE - 1) / CHUNK_SIZE);
  n_chunks    = chunks_x * chunks_y;

  f32 inv_samples = 1.0f / samples;
  f32 inv_width   = 1.0f / width;
  f32 inv_height  = 1.0f / height;
  f32 aspect      = (f32)width / (f32)height;

  loop {
    isize c = atomic_fetch_add(&ctx->_current_chunk, 1);
    if (c >= n_chunks) {
      atomic_fetch_add(&ctx->n_threads, -1);
      return;
    }

    isize start_x = (c % chunks_x) * CHUNK_SIZE;
    isize start_y = (c / chunks_x) * CHUNK_SIZE;

    for_range(y, start_y, start_y + CHUNK_SIZE) {
      if (y >= height) {
        break;
      }

      for_range(x, start_x, start_x + CHUNK_SIZE) {
        if (x >= width) {
          break;
        }

        Color3 color = vec3(0, 0, 0);

        f32x8 sample_indices = { 0, 1, 2, 3, 4, 5, 6, 7, };

        for_range(sample_batch, 0, (samples + 7) / 8) {
          f32x8 rand_a = hash12x8((Vec2x8) {
            .x = _mm256_set1_ps((f32)x * 50.0f) + sample_indices,
            .y = _mm256_set1_ps((f32)y),
          });
          f32x8 rand_b = hash12x8((Vec2x8) {
            .x = _mm256_set1_ps((f32)x * 50.0f) + sample_indices,
            .y = _mm256_set1_ps((f32)y),
          });

          Vec2x8 uvs = {
            .x = ((f32)x + rand_a - 0.5) * 2.0f * inv_width  - 1.0f,
            .y = ((f32)y + rand_b - 0.5) * 2.0f * inv_height - 1.0f,
          };
          Vec3x8 directions = {
            .x =  uvs.x * aspect,
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
            if (sample + sample_batch * 8 >= samples) {
              break;
            }
            Ray r = {
              .position  = camera.position,
              .direction = vec3(directions_x[sample], directions_y[sample], directions_z[sample]),
            };
            color = vec3_add(color, cast_ray(ctx->scene, r, max_bounces));
          }
          sample_indices += _mm256_set1_ps(8);
        }

        color = vec3_scale(color, inv_samples);
        // color = tonemap(color);
        color = vec3(
          clamp(color.r, 0, 1),
          clamp(color.g, 0, 1),
          clamp(color.b, 0, 1),
        );
        color = vec3(
          linear_to_srgb(color.r),
          linear_to_srgb(color.g),
          linear_to_srgb(color.b),
        );
        color = vec3_scale(color, 255.999f);

        IDX(ctx->image.pixels, ctx->image.components * (x + y * ctx->image.stride) + 0) = (u8)color.data[0];
        IDX(ctx->image.pixels, ctx->image.components * (x + y * ctx->image.stride) + 1) = (u8)color.data[1];
        IDX(ctx->image.pixels, ctx->image.components * (x + y * ctx->image.stride) + 2) = (u8)color.data[2];
      }
    }
  }
}

extern void lightmap_bake(Image const *lightmap, Scene const *scene, isize samples) {
  assert(lightmap->pixel_type == PT_u8);
  assert(lightmap->components >= 3);

  for_range(i, 0, scene->triangles.len) {
    Triangle_AOS aos = scene->triangles.aos[i];
    i32 min_x = min(aos.tex_coords_a.x, min(aos.tex_coords_b.x, aos.tex_coords_c.x)) * lightmap->width;
    i32 max_x = max(aos.tex_coords_a.x, max(aos.tex_coords_b.x, aos.tex_coords_c.x)) * lightmap->width;
    i32 min_y = min(aos.tex_coords_a.y, min(aos.tex_coords_b.y, aos.tex_coords_c.y)) * lightmap->height;
    i32 max_y = max(aos.tex_coords_a.y, max(aos.tex_coords_b.y, aos.tex_coords_c.y)) * lightmap->height;

    Vec2 p0 = vec2_mul(aos.tex_coords_a, vec2(lightmap->width, lightmap->height));
    Vec2 p1 = vec2_mul(aos.tex_coords_b, vec2(lightmap->width, lightmap->height));
    Vec2 p2 = vec2_mul(aos.tex_coords_c, vec2(lightmap->width, lightmap->height));

    f32 denom = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
    
    for_range(y, min_y, max_y + 1) {
      for_range(x, min_x, max_x + 1) {
        Vec2 p = vec2(x, y);
        
        f32 w0 = ((p1.y - p2.y) * (p.x - p2.x) + (p2.x - p1.x) * (p.y - p2.y)) / denom;
        f32 w1 = ((p2.y - p0.y) * (p.x - p2.x) + (p0.x - p2.x) * (p.y - p2.y)) / denom;
        f32 w2 = 1.0f - w0 - w1;
        
        if (w0 >= -EPSILON && w1 >= -EPSILON && w2 >= -EPSILON) {
          Vec3 position = vec3(
            .x = scene->triangles.a_x[i] * w0 + scene->triangles.b_x[i] * w1 + scene->triangles.c_x[i] * w2,
            .y = scene->triangles.a_y[i] * w0 + scene->triangles.b_y[i] * w1 + scene->triangles.c_y[i] * w2,
            .z = scene->triangles.a_z[i] * w0 + scene->triangles.b_z[i] * w1 + scene->triangles.c_z[i] * w2,
          );
          Vec3 normal = vec3(
            .x = scene->triangles.aos[i].normal_a.x * w0 + scene->triangles.aos[i].normal_b.x * w1 + scene->triangles.aos[i].normal_c.x * w2,
            .y = scene->triangles.aos[i].normal_a.y * w0 + scene->triangles.aos[i].normal_b.y * w1 + scene->triangles.aos[i].normal_c.y * w2,
            .z = scene->triangles.aos[i].normal_a.z * w0 + scene->triangles.aos[i].normal_b.z * w1 + scene->triangles.aos[i].normal_c.z * w2,
          );

          Vec3 accumulated = {0};

          Ray r = {
            .position = vec3_add(position, vec3_scale(normal, EPSILON)),
          };
          for_range(sample, 0, samples) {
            f32 cos;
            loop {
              Vec3 d = rand_vec3();
              cos = vec3_dot(d, normal);
              if (cos > 0) {
                r.direction = d;
                break;
              }
            }
            accumulated = vec3_add(accumulated, vec3_scale(cast_ray(scene, r, 8), cos));
          }
          
          IDX(lightmap->pixels, (x + y * lightmap->stride) * lightmap->components + 0) = accumulated.x / samples;
          IDX(lightmap->pixels, (x + y * lightmap->stride) * lightmap->components + 1) = accumulated.y / samples;
          IDX(lightmap->pixels, (x + y * lightmap->stride) * lightmap->components + 2) = accumulated.z / samples;
        }
      }
    }
  }
}

extern b8 rendering_context_is_finished(Rendering_Context *context) {
  return context->n_threads == 0;
}

extern void rendering_context_finish(Rendering_Context *context) {
  while (context->n_threads > 0) {
    processor_yield();
  }
}
