#include "codin/codin.h"

#include "codin/allocators.h"
#include "codin/fmt.h"
#include "codin/obj.h"
#include "codin/strconv.h"
#include "codin/thread.h"

#include "raytracer.h"
#include "gltf.h"
#include "uri.h"
#include "denoiser.h"

// #define sample_texture sample_texture_nearest
#define sample_texture sample_texture_bilinear

#define CHUNK_SIZE 32

#undef STRING_CASE

#define STRING_CASE_C(s) STRING_CASE(LIT(s))
#define STRING_CASE(s) \
  } else if (string_equal(_string_switch_string, s)) {

#define STRING_SWITCH(s) { String _string_switch_string = s;

#undef  STRING_SWITCH
#define STRING_SWITCH(s) { String _string_switch_string = (s); if (false) {
#define STRING_DEFAULT() } else {
#define STRING_SWITCH_END() }}

internal Color3 sample_texture_nearest(Image const *texture, Vec2 tex_coords) {
  if (tex_coords.x < 0) {
    tex_coords.x += -(i32)tex_coords.x + 1;
  }
  if (tex_coords.y < 0) {
    tex_coords.y += -(i32)tex_coords.y + 1;
  }
  tex_coords = vec2_fract(tex_coords);
  isize u = tex_coords.x * texture->width;
  isize v = tex_coords.y * texture->height;

  return vec3(
    IDX(texture->pixels, texture->components * (u + texture->stride * v) + 0) / 255.999f,
    IDX(texture->pixels, texture->components * (u + texture->stride * v) + 1) / 255.999f,
    IDX(texture->pixels, texture->components * (u + texture->stride * v) + 2) / 255.999f,
  );
}

internal Color3 sample_texture_bilinear(Image const *texture, Vec2 tex_coords) {
  if (tex_coords.x < 0) {
    tex_coords.x += -(i32)tex_coords.x + 1;
  }
  if (tex_coords.y < 0) {
    tex_coords.y += -(i32)tex_coords.y + 1;
  }
  tex_coords = vec2_fract(tex_coords);
  f32 px = tex_coords.x * texture->width;
  f32 py = tex_coords.y * texture->height;

  isize u = (isize)px;
  isize v = (isize)py;

  f32 a = px - u;
  f32 b = py - v;

  isize u2 = (u + 1 < texture->width)  ? u + 1 : u;
  isize v2 = (v + 1 < texture->height) ? v + 1 : v;

  Color3 c00 = vec3(
    IDX(texture->pixels, texture->components * (u  + texture->stride * v ) + 0) / 255.999f,
    IDX(texture->pixels, texture->components * (u  + texture->stride * v ) + 1) / 255.999f,
    IDX(texture->pixels, texture->components * (u  + texture->stride * v ) + 2) / 255.999f,
  );
  Color3 c10 = vec3(
    IDX(texture->pixels, texture->components * (u2 + texture->stride * v ) + 0) / 255.999f,
    IDX(texture->pixels, texture->components * (u2 + texture->stride * v ) + 1) / 255.999f,
    IDX(texture->pixels, texture->components * (u2 + texture->stride * v ) + 2) / 255.999f,
  );
  Color3 c01 = vec3(
    IDX(texture->pixels, texture->components * (u  + texture->stride * v2) + 0) / 255.999f,
    IDX(texture->pixels, texture->components * (u  + texture->stride * v2) + 1) / 255.999f,
    IDX(texture->pixels, texture->components * (u  + texture->stride * v2) + 2) / 255.999f,
  );
  Color3 c11 = vec3(
    IDX(texture->pixels, texture->components * (u2 + texture->stride * v2) + 0) / 255.999f,
    IDX(texture->pixels, texture->components * (u2 + texture->stride * v2) + 1) / 255.999f,
    IDX(texture->pixels, texture->components * (u2 + texture->stride * v2) + 2) / 255.999f,
  );

  Color3 c0 = vec3_lerp(c00, c10, a);
  Color3 c1 = vec3_lerp(c01, c11, a);
  return vec3_lerp(c0, c1, b);
}

internal Color3 sample_background(Image const *image, Vec3 dir) {
  f32 inv_pi     = 1.0f / PI;
  f32 inv_two_pi = 1.0f / (2.0f * PI);

  f32 u = 0.5f + atan2_f32(dir.z, dir.x) * inv_two_pi;
  f32 v = 0.5f - asin_f32(dir.y) * inv_pi;

  Color3 color = sample_texture(image, vec2(u, v));
  return srgb_to_linear(color);
}

internal void load_texture(String path, Image *texture) {
  b8 ok = stb_image_load_bytes(
    unwrap_err(read_entire_file_path(path, context.allocator)),
    texture,
    context.allocator
  );
  if (!ok) {
    fmt_eprintflnc("Failed to load texture: '%S'", path);
    process_exit(1);
  }
}

internal inline Vec3 sample_cosine_hemisphere() {
  f32 angle    = rand_f32() * 2 * PI;
  f32 distance = sqrt_f32(rand_f32());
  Vec3 v = vec3(
    .x = sin_f32(angle) * distance,
    .y = cos_f32(angle) * distance,
  );
  v.z = sqrt_f32(1 - distance * distance);
  return v;
}

internal inline Vec3 normal_map_apply(Image *normal_map, f32 normal_map_strength, Shader_Input const *input) {
  Vec3 normal = input->normal;
  if (normal_map) {
    Vec3 v = sample_texture(normal_map, input->tex_coords);
         v = vec3_add(vec3_scale(v, 2.0), vec3(-1, -1, -1));

    v.g *= -1;

    Vec3 t = input->tangent;
    Vec3 b = input->bitangent;
    Vec3 n = input->normal;

    f32 s = normal_map_strength;

    normal = vec3_normalize(
      vec3(
        .x = s * (v.x * t.x + v.y * b.x + v.z * n.x) + n.x * (1 - s),
        .y = s * (v.x * t.y + v.y * b.y + v.z * n.y) + n.y * (1 - s),
        .z = s * (v.x * t.z + v.y * b.z + v.z * n.z) + n.z * (1 - s),
      )
    );
  }

  return normal;
}

internal inline void basis(Vec3 view, Vec3 normal, Vec3 *tangent, Vec3 *bitangent) {
  if (abs_f32(vec3_dot(normal, view)) < 0.9999f) {
    *tangent = vec3_normalize(vec3_cross(normal, view));
  } else if (abs_f32(vec3_dot(normal, vec3(0, 1, 0))) < 0.9999f) {
    *tangent = vec3_normalize(vec3_cross(normal, vec3(0, 1, 0)));
  } else {
    *tangent = vec3_normalize(vec3_cross(normal, vec3(1, 0, 0)));
  }
  *bitangent = vec3_cross(normal, *tangent);
}

internal inline Color3 disney_calculate_sheen_tint(Color3 base_color) {
  f32 luminance = vec3_dot(vec3(0.3f, 0.6f, 1.0f), base_color);
  return (luminance > 0.0f) ? vec3_scale(base_color, (1.0f / luminance)) : vec3(1, 1, 1);
}

internal inline f32 disney_fresnel_schlick_weight(f32 cos_theta) {
  f32 m = 1 - cos_theta;
  return m * m * m * m * m;
}

internal inline Vec3 disney_evaluate_sheen(f32 sheen, Color3 base_color, f32 sheen_tint, f32 h_dot_l) {
  if (sheen <= 0.0f) {
    return vec3(0);
  }

  Vec3 tint = disney_calculate_sheen_tint(base_color);
  return vec3_scale(vec3_lerp(vec3(1, 1, 1), tint, sheen_tint), sheen * disney_fresnel_schlick_weight(h_dot_l));
}

enum {
  PBR_Type_Metal,
  PBR_Type_Diffuse,
  // PBR_Type_Dielectric,
};

typedef struct {
  Vec3   base_color, emission;
  f32    roughness, metalness, normal_map_strength, sheen, sheen_tint, anisotropic_strength;
  Image *texture_albedo;
  Image *texture_normal;
  Image *texture_metal_roughness;
  Image *texture_emission;
} PBR_Shader_Data;

internal inline f32 luminance(Color3 x) {
	return vec3_dot(x, vec3(0.2126f, 0.7152f, 0.0722f));
}

internal inline f32 fresnel_schlick_f32(f32 f0, f32 f90, f32 theta) {
  return f0 + (f90 - f0) * pow_f32(1 - theta, 5);
}

internal inline Vec3 fresnel_schlick_vec3(Color3 f0, f32 f90, f32 theta) {
  return vec3_add(f0, vec3_scale(vec3_sub(vec3_broadcast(f90), f0), pow_f32(1 - theta, 5)));
}

internal inline f32 distribution_GGX(f32 roughness, f32 NoH, f32 k) {
  f32 a2 = roughness * roughness;
  return a2 / (PI * pow_f32((NoH * NoH) * (a2 * a2 - 1) + 1, k));
}

internal inline f32 smith_G(f32 NDotV, f32 alpha2) {
  f32 a = alpha2 * alpha2;
  f32 b = NDotV  * NDotV;
  return (2.0 * NDotV) / (NDotV + sqrt_f32(a + b - a * b));
}

internal inline f32 geometry_term(f32 NoL, f32 NoV, f32 roughness) {
  f32 a2 = roughness * roughness;
  f32 G1 = smith_G(NoV, a2);
  f32 G2 = smith_G(NoL, a2);
  return G1 * G2;
}

internal inline Vec3 sample_GGX_VNDF(Vec3 V, f32 ax, f32 ay) {
  Vec3 Vh = vec3_normalize(vec3(ax * V.x, ay * V.y, V.z));

  f32 lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  Vec3 T1 = lensq > 0 ? vec3_scale(vec3(-Vh.y, Vh.x, 0), 1.0f / sqrt_f32(lensq)) : vec3(1, 0, 0);
  Vec3 T2 = vec3_cross(Vh, T1);

  f32 r   = sqrt_f32(rand_f32());
  f32 phi = 2.0 * PI * rand_f32();
  f32 t1  = r * cos_f32(phi);
  f32 t2  = r * sin_f32(phi);
  f32 s   = 0.5 * (1.0 + Vh.z);
  t2      = (1.0 - s) * sqrt_f32(1.0 - t1 * t1) + s * t2;

  Vec3 Nh = vec3_add(
    vec3_add(vec3_scale(T1, t1), vec3_scale(T2, t2)),
    vec3_scale(Vh, sqrt_f32(max(0.0, 1.0 - t1 * t1 - t2 * t2)))
  );

  return vec3_normalize(vec3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
}

internal inline f32 pdf_GGX_VNDF(f32 NoH, f32 NoV, f32 roughness) {
 	f32 D  = distribution_GGX(roughness, NoH, 2);
  f32 G1 = smith_G(NoV, roughness * roughness);
  return (D * G1) / max(0.00001f, 4.0f * NoV);
}

internal inline Color3 disney_eval_diffuse(Color3 base_color, f32 NoL, f32 NoV, f32 LoH, f32 roughness) {
  f32 FD90 = 0.5f + 2 * roughness * LoH * LoH;
  f32 a = fresnel_schlick_f32(1.0f, FD90, NoL);
  f32 b = fresnel_schlick_f32(1.0f, FD90, NoV);

  return vec3_scale(base_color, (a * b / PI));
}

internal inline Color3 disney_eval_specular(f32 roughness, Color3 F, f32 NoH, f32 NoV, f32 NoL) {
  f32 D = distribution_GGX(roughness, NoH, 2);
  f32 G = geometry_term(NoL, NoV, roughness);

  return vec3_scale(F, D * G / (4 * NoL * NoV));
}

internal inline f32 shadowed_f90(Color3 f0) {
	const f32 t = 1.0f / 0.04f;
	return min(1.0f, t * luminance(f0));
}

typedef struct {
  f32    roughness, metalness, sheen, sheen_tint, anisotropic_strength2;
  Color3 base_color;
} Disney_BRDF_Data;

f32 lerp_f32(f32 x, f32 y, f32 t) {
  return x * (1 - t) + y * t;
}

Vec4 sample_disney_BRDF(Disney_BRDF_Data const *data, Vec3 in_dir, Vec3 *out_dir) {
  f32  alpha_x      = lerp_f32(data->roughness * data->roughness, 1, data->anisotropic_strength2);
  f32  alpha_y      = data->roughness * data->roughness;
  Vec3 micro_normal = sample_GGX_VNDF(in_dir, alpha_x, alpha_y);

  Color3 f0      = vec3_lerp(vec3_broadcast(0.04f), data->base_color, data->metalness);
	Color3 fresnel = fresnel_schlick_vec3(f0, shadowed_f90(f0), vec3_dot(in_dir, micro_normal));

  f32 diffuse_weight  = 1 - data->metalness;
  f32 specular_weight = luminance(fresnel);
  f32 inverse_weight  = 1 / (diffuse_weight + specular_weight);

  diffuse_weight  *= inverse_weight;
  specular_weight *= inverse_weight;
  
  Vec4 brdf = vec4(0);
  if (rand_f32() < diffuse_weight) {
    *out_dir = sample_cosine_hemisphere();
    micro_normal = vec3_normalize(vec3_add(*out_dir, in_dir));
    
    f32 NoL = out_dir->z;
    f32 NoV = in_dir.z;
    if (NoL <= 0 || NoV <= 0) {
      return vec4(0);
    }
    f32 LoH = vec3_dot(*out_dir, micro_normal);
    f32 pdf = NoL / PI;
    
    Color3 diff = vec3_mul(disney_eval_diffuse(data->base_color, NoL, NoV, LoH, data->roughness), vec3_sub(vec3(1, 1, 1), fresnel));
           diff = vec3_add(diff, disney_evaluate_sheen(data->sheen, data->base_color, data->sheen_tint, LoH));
    brdf = vec4(
      diff.r * NoL,
      diff.g * NoL,
      diff.b * NoL,
      diffuse_weight * pdf
    );
  } else {
    *out_dir = vec3_reflect(vec3_scale(in_dir, -1), micro_normal);
    
    f32 NoL = out_dir->z;
    f32 NoV = in_dir.z;
    if (NoL <= 0 || NoV <= 0) {
      return vec4(0);
    }
    NoL = max(NoL, 0.001f);
    NoV = max(NoV, 0.001f);
    f32 NoH = min(micro_normal.z, 0.99f);
    f32 pdf = pdf_GGX_VNDF(NoH, NoV, data->roughness);
    
    Vec3 spec = disney_eval_specular(data->roughness, fresnel, NoH, NoV, NoL);
    brdf = vec4(
      spec.r * NoL,
      spec.g * NoL,
      spec.b * NoL,
      specular_weight * pdf
    );
  }

  *out_dir = vec3_normalize(*out_dir);

  return brdf;
}

internal void disney_shader_proc(rawptr _data, Shader_Input const *input, Shader_Output *output) {
  PBR_Shader_Data const *data = (PBR_Shader_Data *)_data;
  Vec3 normal = normal_map_apply(data->texture_normal, data->normal_map_strength, input);

  Color3 base_color = data->base_color;
  if (data->texture_albedo) {
    base_color = vec3_mul(base_color, srgb_to_linear(sample_texture(data->texture_albedo, input->tex_coords)));
  }

  f32 roughness = data->roughness;
  f32 metalness = data->metalness;

  if (data->texture_metal_roughness) {
    Vec3 mr = sample_texture(data->texture_metal_roughness, input->tex_coords);
    roughness *= mr.g;
    metalness *= mr.b;
  }

  roughness = clamp(roughness, 0.001f, 1);
  // uhh there's reasons for this
  if (metalness > 0.9f) {
    metalness = 0.9f;
  }
  metalness /= 0.9f;

  Color3 emmission = data->emission;
  if (data->texture_emission) {
    emmission = vec3_mul(emmission, srgb_to_linear(sample_texture(data->texture_emission, input->tex_coords)));
  }
  output->emission = emmission;

  Vec3 t, b;
  basis(input->direction, normal, &t, &b);
  Matrix_3x3 tangent_to_world = matrix_3x3_from_basis(t, b, normal);
  Matrix_3x3 world_to_tangent = matrix_3x3_transpose(tangent_to_world);

  Disney_BRDF_Data brdf_data = {
    .roughness             = roughness,
    .metalness             = metalness,
    .base_color            = base_color,
    .sheen                 = data->sheen,
    .sheen_tint            = data->sheen_tint,
    .anisotropic_strength2 = data->anisotropic_strength * data->anisotropic_strength,
  };
  
  Vec3 in_dir = matrix_3x3_mul_vec3(world_to_tangent, vec3_scale(input->direction, -1));
  Vec4 brdf   = sample_disney_BRDF(&brdf_data, in_dir, &output->direction);

  output->direction = matrix_3x3_mul_vec3(tangent_to_world, output->direction);

  if (brdf.a > 0) {
    output->tint = vec3(
      brdf.r / brdf.a,
      brdf.g / brdf.a,
      brdf.b / brdf.a,
    );
  } else {
    output->terminate = true;
  }
}

internal void debug_shader_proc(rawptr _data, Shader_Input const *input, Shader_Output *output) {
  PBR_Shader_Data const *data = (PBR_Shader_Data *)_data;

  Vec3 normal = normal_map_apply(data->texture_normal, data->normal_map_strength, input);

  output->emission  = vec3_add(vec3_scale(normal, 0.5), vec3_broadcast(0.5f));
  output->terminate = true;
}

void print_usage() {
  fmt_eprintflnc("%S -W <width> -H <height> -S <samples> -T <threads> -B <max_bounces> <model.(obj|glb|gltf)>", IDX(os_args, 0));
}

typedef struct {
  String model;
  isize  width, height, samples, max_bounces, n_threads;
  b8     verbose, denoise;
} Config;

internal b8 parse_command_line_args(Config *config) {
  for (isize i = 1; i < os_args.len; ) {
    String arg = IDX(os_args, i);
    if (IDX(arg, 0) == '-') {
      if (arg.len != 2) {
        print_usage();
        return false;
      }
      if (IDX(arg, 1) == 'V') {
        config->verbose = true;
        i += 1;
        continue;
      }
      if (IDX(arg, 1) == 'D') {
        config->denoise = true;
        i += 1;
        continue;
      }
      if (i == os_args.len - 1) {
        print_usage();
        return false;
      }

      String arg2   = IDX(os_args, i + 1);
      isize  number = parse_int(&arg2);
      switch (IDX(arg, 1)) {
      CASE 'W':
        config->width = number;
      CASE 'H':
        config->height = number;
      CASE 'S':
        config->samples = number;
      CASE 'T':
        config->n_threads = number;
      CASE 'B':
        config->max_bounces = number;
      DEFAULT:
        print_usage();
        return false;
      }

      i += 2;
    } else {
      if (config->model.len) {
        print_usage();
        return false;
      } else {
        config->model = arg;
      }
      i += 1;
    }
  }

  if (config->model.len == 0) {
    print_usage();
    return false;
  }

  return true;
}

internal b8 load_model_obj(String path, Byte_Slice data, Triangle_Slice *triangles, Camera *camera) {
  Obj_File obj = {0};
  b8 ok = obj_load(bytes_to_string(data), &obj, true, context.allocator);
  if (!ok) {
    fmt_eprintlnc("Failed to load model file");
    return 1;
  }

  Hash_Map(String, Image) obj_images;
  hash_map_init(&obj_images, 64, string_equal, string_hash, context.allocator);

  #define obj_load_texture(path) {                       \
    if (path.len && !hash_map_get(obj_images, (path))) { \
      Image image;                                       \
      load_texture((path), &image);                      \
      hash_map_insert(&obj_images, (path), image);       \
    }                                                    \
  }

  slice_iter_v(obj.materials, material, i, {
    obj_load_texture(material.texture_diffuse.path);
    obj_load_texture(material.texture_emissive.path);
    if (material.is_pbr) {
      obj_load_texture(material.pbr.texture_roughness.path);
      obj_load_texture(material.pbr.texture_metallic.path);
      obj_load_texture(material.pbr.texture_normal.path);
      obj_load_texture(material.pbr.texture_sheen.path);
    } else {
      obj_load_texture(material.simple.texture_specular_color.path);
      obj_load_texture(material.simple.texture_specular.path);
      obj_load_texture(material.simple.texture_ambient.path);
      obj_load_texture(material.simple.texture_alpha.path);
      obj_load_texture(material.simple.texture_bump.path);
    }
  });

  Vector(PBR_Shader_Data) obj_shader_data;
  vector_init(&obj_shader_data, 0, 8, context.allocator);

  slice_iter_v(obj.materials, material, i, {
    PBR_Shader_Data d = {
      .base_color       = material.diffuse,
      .emission         = material.emissive,
      .roughness        = 0.5f,
      .texture_albedo   = hash_map_get(obj_images, material.texture_diffuse.path),
      .texture_emission = hash_map_get(obj_images, material.texture_emissive.path),
    };
    if (material.is_pbr) {
      d.anisotropic_strength    = material.pbr.anisotropic;
      d.metalness               = material.pbr.metallic;
      d.roughness               = material.pbr.roughness;
      d.sheen                   = material.pbr.sheen;
      d.texture_normal          = hash_map_get(obj_images, material.pbr.texture_normal.path);
      d.texture_metal_roughness = hash_map_get(obj_images, material.pbr.texture_metallic.path);
    } else {
      fmt_eprintflnc("material %d is not a pbr material", i);
    }
    vector_append(&obj_shader_data, d)
  });

  slice_init(triangles, obj.triangles.len, context.allocator);

  slice_iter_v(obj.triangles, t, i, {
    Triangle triangle = {
      .shader = {
        .data = &IDX(obj_shader_data, t.material),
        .proc = disney_shader_proc,
      },
    };
    for_range(j, 0, 3) {
      triangle.tex_coords[j] = t.vertices[j].tex_coords;
      triangle.positions[j]  = t.vertices[j].position;
      triangle.normals[j]    = t.vertices[j].normal;
    }
    IDX(*triangles, i) = triangle;
  });
  return true;
}

internal b8 load_model_gltf(String path, Byte_Slice data, Triangle_Slice *triangles, Camera *camera) {
  Growing_Arena_Allocator gltf_arena;
  Allocator gltf_allocator = growing_arena_allocator_init(&gltf_arena, 1 << 16, context.allocator);

  Gltf_File gltf;
  b8 gltf_parse_ok = gltf_parse(data, path, &gltf, gltf_allocator);
  assert(gltf_parse_ok);
  b8 gltf_load_buffers_ok = gltf_load_buffers(path, &gltf, gltf_allocator);
  assert(gltf_load_buffers_ok);

  slice_iter_v(gltf.nodes, node, i, {
    if (node.camera != -1) {
      Gltf_Camera cam = IDX(gltf.cameras, node.camera);
      if (cam.is_orthographic) {
        continue;
      }
      *camera = (Camera) {
        .fov          = cam.perspective.y_fov,
        .focal_length = 1.0f / tan_f32(cam.perspective.y_fov * 0.5f),
        .view_matrix  = node.matrix,
      };
      break;
    }
  });

  Slice(PBR_Shader_Data) gltf_shader_data;
  slice_init(&gltf_shader_data, gltf.materials.len, gltf_allocator);

  Slice(Image) gltf_images;
  slice_init(&gltf_images, gltf.images.len, gltf_allocator);

  slice_iter_v(gltf.images, image, i, {
    b8 ok = stb_image_load_bytes(image.data, &IDX(gltf_images, i), gltf_allocator);
    if (!ok) {
      fmt_eprintflnc("Failed to load image: type: '%S', uri: '%S', len: %d", image.mime_type, image.uri, image.data.len);
      return false;
    }
  });

  slice_iter_v(gltf.materials, material, i, {
    PBR_Shader_Data data = {
      .base_color = vec3(
        material.base_color.r,
        material.base_color.g,
        material.base_color.b,
      ),
      .roughness = material.roughness,
      .metalness = material.metallic,
      .sheen     = luminance(material.sheen_color),
      .emission  = material.emissive,
    };
    if (material.texture_normal.index != -1) {
      Gltf_Texture *texture    = &IDX(gltf.textures, material.texture_normal.index);
      data.texture_normal      = &IDX(gltf_images, texture->source);
      data.normal_map_strength = material.texture_normal_scale;

      // TODO: Gltf_Sampler *sampler = &IDX(gltf.samplers, texture->sampler);
    }
    if (material.texture_emissive.index != -1) {
      Gltf_Texture *texture = &IDX(gltf.textures, material.texture_emissive.index);
      data.texture_emission = &IDX(gltf_images, texture->source);
    }
    if (material.texture_base_color.index != -1) {
      Gltf_Texture *texture = &IDX(gltf.textures, material.texture_base_color.index);
      data.texture_albedo   = &IDX(gltf_images, texture->source);
    }
    if (material.texture_metallic_roughness.index != -1) {
      Gltf_Texture *texture        = &IDX(gltf.textures, material.texture_metallic_roughness.index);
      data.texture_metal_roughness = &IDX(gltf_images, texture->source);
    }
    IDX(gltf_shader_data, i) = data;
  });

  Gltf_Triangle_Vector gltf_triangles;
  vector_init(&gltf_triangles, 0, 8, context.allocator);
  gltf_to_triangles(&gltf, &gltf_triangles);

  slice_init(triangles, gltf_triangles.len, context.allocator);

  slice_iter_v(gltf_triangles, t, i, {
    Triangle triangle = {
      .shader = (Shader) {
        .data = &IDX(gltf_shader_data, t.material),
        .proc = disney_shader_proc,
      },
    };
    for_range(j, 0, 3) {
      triangle.tex_coords[j] = t.vertices[j].tex_coords;
      triangle.positions[j]  = t.vertices[j].position;
      triangle.normals[j]    = t.vertices[j].normal;
    }
    IDX(*triangles, i) = triangle;
  });
  return true;
}

internal b8 load_model_file(String path, Triangle_Slice *triangles, Camera *camera) {
  isize extension_offset = -1;
  for (isize i = path.len - 1; i >= 0; i -= 1) {
    if (IDX(path, i) == '.') {
      extension_offset = i;
      break;
    }
  }
  if (extension_offset == -1) {
    fmt_eprintflnc("Unrecognized file type: '%S'", path);
    print_usage();
    return 1;
  }
  String extension = slice_start(path, extension_offset);

  b8 is_obj = false;

  STRING_SWITCH(extension)
  STRING_CASE_C(".obj")
    is_obj = true;
  STRING_CASE_C(".glb")
    is_obj = false;
  STRING_CASE_C(".gltf")
    is_obj = false;
  STRING_DEFAULT()
    fmt_eprintflnc("Unrecognized file type: '%S'", path);
    print_usage();
    return 1;
  STRING_SWITCH_END()

  Byte_Slice data = or_do_err(read_entire_file_path(path, context.allocator), _err, {
    fmt_eprintlnc("Failed to read model file");
    return false;
  });

  b8 ok;
  if (is_obj) {
    ok = load_model_obj(path, data, triangles, camera);
  } else {
    ok = load_model_gltf(path, data, triangles, camera);
  }

  return ok;
}

i32 main() {
  context.logger = (Logger) {0};

  Config config = {
    .width       = 1024,
    .height      = 1024,
    .samples     = 16,
    .max_bounces = 8,
    .n_threads   = 1,
    .verbose     = false,
    .denoise     = false,
  };
  if (!parse_command_line_args(&config)) {
    return 1;
  }

  Image image = {
    .components = 3,
    .pixel_type = PT_u8,
    .width      = config.width,
    .stride     = config.width,
    .height     = config.height,
  };
  image.pixels = slice_make_aligned(Byte_Slice, config.width * config.height * 3, 64, context.allocator);

  Scene scene = {0};

  Image texture_background = {0};
  load_texture(LIT("background.png"), &texture_background);
  scene.background = (Background) {
    .data = &texture_background,
    .proc = (Background_Proc)sample_background,
  };

  scene.camera.view_matrix = matrix_4x4_translation_rotation_scale(vec3(0, 0, 3), vec4(0, 0, 0, 1), vec3(1, 1, 1));
  scene.camera.fov          = (70.0f / 360.0f) * PI * 2.0;
  scene.camera.focal_length = 1.0f / tan_f32(scene.camera.fov * 0.5f);

  Triangle_Slice triangles;
  if (!load_model_file(config.model, &triangles, &scene.camera)) {
    return 1;
  }

  Timestamp bvh_start_time = time_now();
  scene_init(&scene, triangles, context.allocator);
  if (config.verbose) {
    fmt_printflnc("Bvh generated in %dms", time_since(bvh_start_time) / Millisecond);

    fmt_printflnc("Width:     %d", config.width);
    fmt_printflnc("Height:    %d", config.height);
    fmt_printflnc("Samples:   %d", config.samples);
    fmt_printflnc("Bounces:   %d", config.max_bounces);
    fmt_printflnc("Threads:   %d", config.n_threads);
    fmt_printflnc("BVH-Nodes: %d", scene.bvh.nodes.len);
    fmt_printflnc("Triangles: %d", triangles.len);

    fmt_printlnc("");
  }

  Timestamp start_time = time_now();

  Rendering_Context rendering_context = {
    .image       = image,
    .scene       = &scene,
    .max_bounces = config.max_bounces,
    .n_threads   = config.n_threads,
    .samples     = config.samples,
  };

  for_range(i, 0, config.n_threads) {
    thread_create((Thread_Proc)render_thread_proc, &rendering_context, THREAD_STACK_DEFAULT, THREAD_TLS_DEFAULT);
  }

  isize chunks_x = ((image.width  + CHUNK_SIZE - 1) / CHUNK_SIZE);
  isize chunks_y = ((image.height + CHUNK_SIZE - 1) / CHUNK_SIZE);
  isize n_chunks = chunks_x * chunks_y;

  String bar = LIT("====================");
  while (!rendering_context_is_finished(&rendering_context)) {
    isize  c = rendering_context._current_chunk;
    f32    p = c / (f32)((i32)n_chunks);
    if (p > 1.0f) {
      p = 1.0f;
    }
    fmt_printfc("\r[%-20S] %d%%", slice_end(bar, p * bar.len), (i32)(100 * p));
    time_sleep(Millisecond * 500);
  }
  fmt_printflnc("\r[%S] 100%%", bar);

  Duration time = time_since(start_time);
  fmt_printflnc("%dms", (isize)(time / Millisecond));
  if (config.verbose) {
    fmt_printflnc("%d samples/second", (isize)((u64)config.width * config.height * config.samples / (f64)((f64)time / Second)));
  }

  if (config.denoise) {
    Timestamp start_time = time_now();

    Image denoised = image;
    slice_init(&denoised.pixels, image.pixels.len, context.allocator);
    denoise_image(&image, &denoised, config.n_threads);
    image = denoised;

    Duration time = time_since(start_time);
    fmt_printflnc("Denoising: %dms", (isize)(time / Millisecond));
  }

  Fd     output_file   = unwrap_err(file_open(LIT("output.png"), FP_Read_Write | FP_Create | FP_Truncate));
  Writer output_writer = writer_from_handle(output_file);
  assert(png_save_writer(&output_writer, &image));
  file_close(output_file);
}
