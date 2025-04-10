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
