#pragma once

#include "codin/codin.h"
#include "codin/image.h"
#include "codin/linalg.h"

#include "common.h"

typedef struct {
  Vec3 min, max;
} AABB;

typedef struct {
  Matrix_4x4 view_matrix;
  f32        fov, focal_length;
} Camera;

typedef struct {
  Vec3 direction, normal, normal_geo, tangent, bitangent, position;
  Vec2 tex_coords;
} Shader_Input;

typedef struct {
  Vec3   direction;
  Color3 tint, emission;
  b8     terminate;
} Shader_Output;

typedef void (*Shader_Proc)(rawptr, Shader_Input const *, Shader_Output *);

typedef struct {
  rawptr      data;
  Shader_Proc proc;
} Shader;

typedef struct {
  Vec3   positions [3];
  Vec3   normals   [3];
  Vec2   tex_coords[3];
  Shader shader;
} Triangle;

typedef Slice(Triangle) Triangle_Slice;

typedef struct {
  Vec3   normal, normal_a, normal_b, normal_c;
  Vec3   tangent, bitangent;
  Vec2   tex_coords_a, tex_coords_b, tex_coords_c;
  Shader shader;
} Triangle_AOS;

// SOA Vector, allocation starts at `x[0]` and has a size of TRIANGLE_ALLOCATION_SIZE(N)
typedef struct {
  f32          *x[3];
  f32          *y[3];
  f32          *z[3];
  Triangle_AOS *aos;
  i32           len, cap;
  Allocator     allocator;
} Triangles;

#define TRIANGLES_ALLOCATION_SIZE(N) \
  ((N) * (size_of(f32) * 9 + size_of(Triangle_AOS)))

typedef Color3 (*Background_Proc)(rawptr, Vec3 direction);
  
typedef struct {
  Background_Proc proc;
  rawptr          data;
} Background;

#if SIMD_WIDTH == 8
  typedef struct {
    Vec3x8 mins;
    Vec3x8 maxs;
  } BVH_Node;
#elif SIMD_WIDTH == 16
  typedef struct {
    Vec3x16 mins;
    Vec3x16 maxs;
  } BVH_Node;
#endif

typedef struct {
  Vector(BVH_Node) nodes;
  isize            depth;
} BVH;

typedef struct {
  BVH        bvh;
  Camera     camera;
  Triangles  triangles;
  Background background;
} Scene;

extern void scene_save_writer(Writer const *w, Scene const *scene);
extern b8   scene_load_bytes(Byte_Slice data, Scene *scene);
extern void scene_init(Scene *scene, Triangle_Slice src_triangles, Allocator allocator);
