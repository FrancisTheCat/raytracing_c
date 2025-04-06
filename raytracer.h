#pragma once

#include "codin/codin.h"
#include "codin/linalg.h"

#define SIMD_WIDTH 8

#if SIMD_WIDTH == 8
  #define SIMD_ALIGN 32
  #define ray_aabbs_hit_SIMD ray_aabbs_hit_8
  #define ray_spheres_hit_SIMD ray_spheres_hit_8
  #define ray_triangles_hit_SIMD ray_triangles_hit_8
#elif SIMD_WIDTH == 16
  #define SIMD_ALIGN 64
  #define ray_aabbs_hit_SIMD ray_aabbs_hit_16
  #define ray_spheres_hit_SIMD ray_spheres_hit_16
  #define ray_triangles_hit_SIMD ray_triangles_hit_16
#endif

#include "scene.h"
#include "common.h"

#undef  IDX
#define IDX(arr, i) (arr).data[(i)]

typedef struct {
  Vec3 position;
  Vec3 direction;
} Ray;

typedef struct {
  f32         distance;
  Vec3        normal, normal_geo, point, tangent, bitangent;
  Vec2        tex_coords;
  Shader      shader;
} Hit;

typedef struct {
  f32    *position_x;
  f32    *position_y;
  f32    *position_z;
  f32    *radius;
  Shader *shader;
  isize   len;
} Spheres;

typedef struct {
  f32  *min_x;
  f32  *min_y;
  f32  *min_z;
  f32  *max_x;
  f32  *max_y;
  f32  *max_z;
  isize len;
} AABBs;

typedef struct {
  Image       image;
  Scene      *scene;
  isize       samples, max_bounces;
  _Atomic i32 n_threads, _current_chunk;
} Rendering_Context;

extern void rendering_context_finish(Rendering_Context *context);
extern b8   rendering_context_is_finished(Rendering_Context *context);

extern void render_thread_proc(Rendering_Context *context);

extern void lightmap_bake(Image const *lightmap, Scene const *scene, isize samples);
