#pragma once

#include "codin/codin.h"
#include "codin/linalg.h"

#include "immintrin.h"

#define EPSILON 0.0001f

internal thread_local u32 random_state = 0;

internal u32 rand_u32() {
  u32 state = random_state * 747796405u + 2891336453u;
  u32 word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  random_state = (word >> 22u) ^ word;
  return random_state;
}

internal f32 rand_f32() {
  return rand_u32() / (f32)U32_MAX;
}

internal f32 rand_f32_range(f32 min, f32 max) {
  return rand_f32() * (max - min) + min;
}

internal Vec3 rand_vec3() {
  loop {
    Vec3 p = vec3(
      rand_f32_range(-1, 1),
      rand_f32_range(-1, 1),
      rand_f32_range(-1, 1),
    );
    f32 lensq = vec3_length2(p);
    if (EPSILON < lensq && lensq <= 1) {
      return vec3_scale(p, 1.0 / sqrt_f32(lensq));
    }
  }
}

typedef __m256 f32x8;

typedef struct {
  f32x8 x, y;
} Vec2x8;

typedef struct {
  f32x8 x, y, z;
} Vec3x8;

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

internal inline Matrix_3x3 matrix_3x3_from_basis(Vec3 a, Vec3 b, Vec3 c) {
  return (Matrix_3x3) {
    .rows = {
      { a.x, b.x, c.x, },
      { a.y, b.y, c.y, },
      { a.z, b.z, c.z, },
    },
  };
}

internal inline Matrix_3x3 matrix_3x3_transpose(Matrix_3x3 m) {
  return (Matrix_3x3) {
    .rows = {
      { m.rows[0][0], m.rows[1][0], m.rows[2][0], },
      { m.rows[0][1], m.rows[1][1], m.rows[2][1], },
      { m.rows[0][2], m.rows[1][2], m.rows[2][2], },
    },
  };
}

internal inline Vec3 matrix_3x3_mul_vec3(Matrix_3x3 m, Vec3 v) {
  return vec3(
    v.x * m.rows[0][0] + v.y * m.rows[0][1] + v.z * m.rows[0][2],
    v.x * m.rows[1][0] + v.y * m.rows[1][1] + v.z * m.rows[1][2],
    v.x * m.rows[2][0] + v.y * m.rows[2][1] + v.z * m.rows[2][2],
  );
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

  rot.rows[0][1] = 0 + t.data[1] * a.data[0] - s * a.data[2];
  rot.rows[1][1] = c + t.data[1] * a.data[1];
  rot.rows[2][1] = 0 + t.data[1] * a.data[2] + s * a.data[0];

  rot.rows[0][2] = 0 + t.data[2] * a.data[0] + s * a.data[1];
  rot.rows[1][2] = 0 + t.data[2] * a.data[1] - s * a.data[0];
  rot.rows[2][2] = c + t.data[2] * a.data[2];

  return rot;
}

internal inline Color3 srgb_to_linear(Color3 x) {
  return vec3(
    pow_f32((x.r + 0.055f) / 1.055f, 2.4),
    pow_f32((x.g + 0.055f) / 1.055f, 2.4),
    pow_f32((x.b + 0.055f) / 1.055f, 2.4),
  );
}

internal inline f32 linear_to_srgb(f32 c) {
  return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * pow_f32(c, 1.0f / 2.4f) - 0.055f);
}

#if SIMD_WIDTH == 16
typedef __m512 f32x16;

typedef struct {
  f32x16 x, y, z;
} Vec3x16;

internal inline Vec3x16 vec3x16_cross(Vec3x16 a, Vec3x16 b) {
  return (Vec3x16) {
    .x = a.y * b.z - a.z * b.y,
    .y = a.z * b.x - a.x * b.z,
    .z = a.x * b.y - a.y * b.x,
  };
}

internal inline Vec3x16 vec3x16_sub(Vec3x16 a, Vec3x16 b) {
  return (Vec3x16) {
    .x = a.x - b.x,
    .y = a.y - b.y,
    .z = a.z - b.z,
  };
}

internal inline Vec3x16 vec3x16_mul(Vec3x16 a, Vec3x16 b) {
  return (Vec3x16) {
    .x = a.x * b.x,
    .y = a.y * b.y,
    .z = a.z * b.z,
  };
}

internal inline f32x16 vec3x16_dot(Vec3x16 a, Vec3x16 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
#endif
