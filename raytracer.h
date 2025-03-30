#pragma once

#include "codin/codin.h"

#include "codin/allocators.h"
#include "codin/image.h"
#include "codin/linalg.h"
#include "codin/os.h"
#include "codin/fmt.h"
#include "codin/time.h"
#include "codin/thread.h"
#include "codin/sort.h"
#include "codin/math.h"

#include "codin/obj.h"

#include "stdatomic.h"
#include "immintrin.h"

#define EPSILON 0.000001f

#undef  IDX
#define IDX(arr, i) (arr).data[(i)]

typedef __m256 f32x8;

typedef struct {
  f32x8 x, y;
} Vec2x8;

typedef struct {
  f32x8 x, y, z;
} Vec3x8;

typedef union {
  struct {
    f32 rows[3][3];
  };
  f32 data[3 * 3];
} Matrix_3x3;

#define MATRIX_3X3_IDENTITY (Matrix_3x3) { \
  .rows = {                                \
    {1, 0, 0},                             \
    {0, 1, 0},                             \
    {0, 0, 1},                             \
  },                                       \
}

internal inline Matrix_3x3 matrix_3x3_rotate(Vec3 v, f32 radians) {
  f32 c = cos_f32(radians);
  f32 s = sin_f32(radians);

  Vec3 a = vec3_normalize(v);
  Vec3 t = vec3_scale(a, 1 - c);

  Matrix_3x3 rot = MATRIX_3X3_IDENTITY;

  rot.rows[0][0] = c + t.data[0] * a.data[0];
  rot.rows[1][0] = 0 + t.data[0] * a.data[1] + s * a.data[2];
  rot.rows[2][0] = 0 + t.data[0] * a.data[2] - s * a.data[1];
  rot.rows[3][0] = 0;

  rot.rows[0][1] = 0 + t.data[1] * a.data[0] - s * a.data[2];
  rot.rows[1][1] = c + t.data[1] * a.data[1];
  rot.rows[2][1] = 0 + t.data[1] * a.data[2] + s * a.data[0];
  rot.rows[3][1] = 0;

  rot.rows[0][2] = 0 + t.data[2] * a.data[0] + s * a.data[1];
  rot.rows[1][2] = 0 + t.data[2] * a.data[1] - s * a.data[0];
  rot.rows[2][2] = c + t.data[2] * a.data[2];
  rot.rows[3][2] = 0;

  return rot;
}

typedef struct {
  Vec3 position;
  Vec3 direction;
} Ray;

typedef struct {
  Vec3 direction, normal, tangent, bitangent, position;
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
  f32         distance;
  Vec3        normal, point, tangent, bitangent;
  Vec2        tex_coords;
  Shader      shader;
  b8          back_face;
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
  Vec3   a, b, c;
  Vec3   normal_a, normal_b, normal_c;
  Vec2   tex_coords_a, tex_coords_b, tex_coords_c;
  Shader shader;
} Triangle;

typedef Slice(Triangle) Triangle_Slice;

typedef struct {
  Vec3   normal_a,     normal_b,     normal_c;
  Vec2   tex_coords_a, tex_coords_b, tex_coords_c;
  Shader shader;
} Triangle_AOS;

// SOA Vector, allocation starts at `a_x` and has a size of TRIANGLE_ALLOCATION_SIZE(N)
typedef struct {
  f32          *a_x, *a_y, *a_z;
  f32          *b_x, *b_y, *b_z;
  f32          *c_x, *c_y, *c_z;
  Triangle_AOS *aos;
  i32           len, cap;
  Allocator     allocator;
} Triangles;

#define TRIANGLES_ALLOCATION_SIZE(N) \
  ((N) * (size_of(f32) * 9 + size_of(Triangle_AOS)))

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
  Vec3 min, max;
} AABB;
