#include "gltf.h"

#include "codin/json.h"
#include "codin/os.h"
#include "codin/strings.h"

#include "uri.h"

#define JSON_PARSE_BLOCK(STATUS, PARSER)   \
  for (                                    \
    status =  json_parser_advance(PARSER); \
    status == Json_Status_Continue;        \
    status =  json_parser_advance(PARSER)  \
  )

#undef STRING_CASE

#define STRING_CASE_C(s) STRING_CASE(LIT(s))
#define STRING_CASE(s) \
  } else if (string_equal(_string_switch_string, s)) {

#define STRING_SWITCH(s) { String _string_switch_string = s;

#undef  STRING_SWITCH
#define STRING_SWITCH(s) { String _string_switch_string = (s); if (false) {
#define STRING_DEFAULT() } else {
#define STRING_SWITCH_END() }}

#define JSON_MATCH(STRING, LHS, TYPE)            \
  STRING_CASE_C(STRING)                          \
  if (parser->value.kind != JSON_VALUE_##TYPE) { \
    goto fail;                                   \
  }                                              \
  LHS = parser->value.TYPE;

#define JSON_VALUE_string  Json_Value_String
#define JSON_VALUE_number  Json_Value_Number
#define JSON_VALUE_integer Json_Value_Number
#define JSON_VALUE_boolean Json_Value_Bool

internal b8 gltf_parse_asset(Gltf_File *gltf, Json_Parser *parser) {
  if (parser->value.kind != Json_Value_Object) {
    return false;
  }
  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    STRING_SWITCH(parser->name)
    JSON_MATCH("version",    gltf->asset.version,     string);
    JSON_MATCH("minVersion", gltf->asset.min_version, string);
    JSON_MATCH("generator",  gltf->asset.generator,   string);
    JSON_MATCH("copyright",  gltf->asset.copyright,   string);
    STRING_DEFAULT()
      json_parser_skip(parser);
    STRING_SWITCH_END()
  }

  return status == Json_Status_End;

fail:
  return false;
}

internal b8 gltf_parse_scenes(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  Vector(Gltf_Scene) scenes;
  vector_init(&scenes, 0, 8, allocator);
  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }
    Gltf_Scene scene = {0};
    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name", scene.name, string);
      STRING_CASE_C("nodes")
        Vector(isize) nodes;
        vector_init(&nodes, 0, 8, allocator);
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            goto fail;
          }
          vector_append(&nodes, parser->value.integer);
        }
        scene.nodes = vector_to_slice(type_of(scene.nodes), nodes);
        vector_append(&scenes, scene);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }
  }

  gltf->scenes = vector_to_slice(type_of(gltf->scenes), scenes);

  return status == Json_Status_End;

fail:
  vector_delete(scenes);
  return false;
}

internal b8 gltf_parse_float_slice(Json_Parser *parser, f32 *data, isize count) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }

  isize i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Number || i > count) {
      return false;
    }
    data[i] = parser->value.number;
    i += 1;
  }

  return true;
}

internal b8 gltf_parse_nodes(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  Vector(Gltf_Node) nodes;
  vector_init(&nodes, 0, 8, allocator);
  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }
    Gltf_Node node = {
      .mesh     = -1,
      .camera   = -1,
      .skin     = -1,
    };
    Vec3 translation = {0};
    Vec3 scale       = vec3(1, 1, 1);
    Vec4 rotation    = vec4(0, 0, 0, 1);
    b8   explicit_matrix = false;
    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",   node.name,   string);
      JSON_MATCH("mesh",   node.mesh,   integer);
      JSON_MATCH("camera", node.camera, integer);
      JSON_MATCH("skin",   node.skin,   integer);
      STRING_CASE_C("rotation")
        if (!gltf_parse_float_slice(parser, rotation.data, count_of(rotation.data))) {
          goto fail;
        }
      STRING_CASE_C("translation")
        if (!gltf_parse_float_slice(parser, translation.data, count_of(translation.data))) {
          goto fail;
        }
      STRING_CASE_C("scale")
        if (!gltf_parse_float_slice(parser, scale.data, count_of(scale.data))) {
          goto fail;
        }
      STRING_CASE_C("matrix")
        if (!gltf_parse_float_slice(parser, node.matrix.data, count_of(node.matrix.data))) {
          goto fail;
        }
        explicit_matrix = true;
      STRING_CASE_C("weights")
        if (parser->value.kind != Json_Value_Array) {
          goto fail;
        }
        Vector(f32) weights;
        vector_init(&weights, 0, 8, allocator);
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            goto fail;
          }

          vector_append(&weights, parser->value.number);
        }
        node.weights = vector_to_slice(type_of(node.weights), weights);
      STRING_CASE_C("children")
        if (parser->value.kind != Json_Value_Array) {
          goto fail;
        }
        Vector(isize) children;
        vector_init(&children, 0, 8, allocator);
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            goto fail;
          }

          vector_append(&children, parser->value.integer);
        }
        node.children = vector_to_slice(type_of(node.children), children);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    if (!explicit_matrix) {
      node.matrix = matrix_4x4_translation_rotation_scale(translation, rotation, scale);
    }

    vector_append(&nodes, node);
  }

  gltf->nodes = vector_to_slice(type_of(gltf->nodes), nodes);

  return status == Json_Status_End;

fail:
  vector_delete(nodes);
  return false;
}

internal b8 gltf_parse_pbr(Gltf_PBR_Material *pbr, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Object) {
    return false;
  }

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    STRING_SWITCH(parser->name)
    JSON_MATCH("metallicFactor",  pbr->metallic_factor,  number);
    JSON_MATCH("roughnessFactor", pbr->roughness_factor, number);
    STRING_CASE_C("baseColorFactor")
      if (parser->value.kind != Json_Value_Array) {
        goto fail;
      }
      isize i = 0;
      JSON_PARSE_BLOCK(status, parser) {
        if (parser->value.kind != Json_Value_Number || i > 4) {
          goto fail;
        }
        pbr->base_color_factor.data[i] = parser->value.number;
                
        i += 1;
      }

    STRING_CASE_C("baseColorTexture")
      if (parser->value.kind != Json_Value_Object) {
        goto fail;
      }
      JSON_PARSE_BLOCK(status, parser) {
        STRING_SWITCH(parser->name)
        JSON_MATCH("index",    pbr->base_color_texture.index,     integer);
        JSON_MATCH("texCoord", pbr->base_color_texture.tex_coord, integer);
        STRING_DEFAULT()
          json_parser_skip(parser);
        STRING_SWITCH_END()
      }
      
    STRING_CASE_C("metallicRoughnessTexture")
      if (parser->value.kind != Json_Value_Object) {
        goto fail;
      }
      JSON_PARSE_BLOCK(status, parser) {
        STRING_SWITCH(parser->name)
        JSON_MATCH("index",    pbr->metallic_roughness_texture.index,     integer);
        JSON_MATCH("texCoord", pbr->metallic_roughness_texture.tex_coord, integer);
        STRING_DEFAULT()
          json_parser_skip(parser);
        STRING_SWITCH_END()
      }
      
    STRING_DEFAULT()
      json_parser_skip(parser);
    STRING_SWITCH_END()
  }

  return status == Json_Status_End;
fail:
  return false;
}

internal b8 gltf_parse_materials(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  Vector(Gltf_Material) materials;
  vector_init(&materials, 0, 8, allocator);
  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }
    Gltf_Material material = {
      .alpha_cutoff = 0.5f,
      .alpha_mode   = LIT("OPAQUE"),
      .pbr_metallic_roughness = {
        .metallic_factor   = 1,
        .roughness_factor  = 1,
        .base_color_factor = vec4(1, 1, 1, 1),
      },
    };
    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",        material.name,         string);
      JSON_MATCH("doubleSided", material.double_sided, boolean);
      JSON_MATCH("alphaMode",   material.alpha_mode,   string);
      JSON_MATCH("alphaCutoff", material.alpha_cutoff, number);
      STRING_CASE_C("emissiveFactor")
        if (parser->value.kind != Json_Value_Array) {
          goto fail;
        }
        isize i = 0;
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number || i > 3) {
            goto fail;
          }
          material.emissive_factor.data[i] = parser->value.number;
          
          i += 1;
        }

      STRING_CASE_C("emissiveTexture")
        if (parser->value.kind != Json_Value_Object) {
          goto fail;
        }
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("index",    material.emissive_texture.index,     integer);
          JSON_MATCH("texCoord", material.emissive_texture.tex_coord, integer);
          STRING_DEFAULT()
            json_parser_skip(parser);
          STRING_SWITCH_END()
        }
        
      STRING_CASE_C("normalTexture")
        if (parser->value.kind != Json_Value_Object) {
          goto fail;
        }
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("index",    material.normal_texture.index,     integer);
          JSON_MATCH("texCoord", material.normal_texture.tex_coord, integer);
          JSON_MATCH("scale",    material.normal_texture_scale,     integer);
          STRING_DEFAULT()
            json_parser_skip(parser);
          STRING_SWITCH_END()
        }
        
      STRING_CASE_C("occlusionTexture")
        if (parser->value.kind != Json_Value_Object) {
          goto fail;
        }
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("index",    material.occlusion_texture.index,     integer);
          JSON_MATCH("texCoord", material.occlusion_texture.tex_coord, integer);
          JSON_MATCH("strength", material.occlusion_texture_strength,  integer);
          STRING_DEFAULT()
            json_parser_skip(parser);
          STRING_SWITCH_END()
        }
        
      STRING_CASE_C("pbrMetallicRoughness")
        if (!gltf_parse_pbr(&material.pbr_metallic_roughness, parser, allocator)) {
          goto fail;
        }
        
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&materials, material);
  }

  gltf->materials = vector_to_slice(type_of(gltf->materials), materials);

  return status == Json_Status_End;

fail:
  vector_delete(materials);
  return false;
}

internal b8 gltf_parse_accessors(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Accessor) accessors;
  vector_init(&accessors, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Accessor accessor = {0};

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("bufferView",    accessor.buffer_view,    integer);
      JSON_MATCH("byteOffset",    accessor.byte_offset,    integer);
      JSON_MATCH("normalized",    accessor.normalized,     boolean);
      JSON_MATCH("count",         accessor.count,          integer);
      JSON_MATCH("name",          accessor.name,           string);
      STRING_CASE_C("min")
        accessor.has_min = true;
        if (!gltf_parse_float_slice(parser, &accessor.min[0], count_of(accessor.min))) {
          return false;
        }
      STRING_CASE_C("max")
        accessor.has_max = true;
        if (!gltf_parse_float_slice(parser, &accessor.max[0], count_of(accessor.max))) {
          return false;
        }
      STRING_CASE_C("componentType")
        accessor.component_type = parser->value.integer;
        if (
          accessor.component_type < 5120 ||
          accessor.component_type > 5126 ||
          accessor.component_type == 5124
        ) {
          goto fail;
        }
      STRING_CASE_C("type")
        b8 found = false;
        for_range(i, 0, count_of(gltf_accessor_type_strings)) {
          if (string_equal(parser->value.string, gltf_accessor_type_strings[i])) {
            accessor.type = i;
            found = true;
            break;
          }
        }
        if (!found) {
          goto fail;
        }
        
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&accessors, accessor);
  }

  gltf->accessors = vector_to_slice(type_of(gltf->accessors), accessors);

  return true;

fail:
  vector_delete(accessors);
  return false;
}

internal b8 gltf_parse_buffer_views(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Buffer_View) buffer_views;
  vector_init(&buffer_views, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Buffer_View buffer_view = {0};

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("byteOffset", buffer_view.byte_offset, integer);
      JSON_MATCH("byteLength", buffer_view.byte_length, integer);
      JSON_MATCH("byteStride", buffer_view.byte_stride, integer);
      JSON_MATCH("buffer",     buffer_view.buffer,      integer);
      JSON_MATCH("name",       buffer_view.name,        string);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&buffer_views, buffer_view);
  }

  gltf->buffer_views = vector_to_slice(type_of(gltf->buffer_views), buffer_views);

  return true;

fail:
  vector_delete(buffer_views);
  return false;
}

internal b8 gltf_parse_buffers(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Buffer) buffers;
  vector_init(&buffers, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Buffer buffer = {0};

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("byteLength", buffer.byte_length, integer);
      JSON_MATCH("name",       buffer.name,        string);
      JSON_MATCH("uri",        buffer.uri,         string);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&buffers, buffer);
  }

  gltf->buffers = vector_to_slice(type_of(gltf->buffers), buffers);

  return true;

fail:
  vector_delete(buffers);
  return false;
}

internal b8 gltf_parse_primitive(Json_Parser *parser, Gltf_Primitive *primitive, Allocator allocator) {
  if (parser->value.kind != Json_Value_Object) {
    return false;
  }

  *primitive = (Gltf_Primitive) {
    .mode = Gltf_Primitive_Mode_Triangles,
  };

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    STRING_SWITCH(parser->name)
    STRING_CASE_C("indices")
      if (parser->value.kind != Json_Value_Number) {
        goto fail;
      }
      primitive->indices     = parser->value.integer;
      primitive->has_indices = true;
    JSON_MATCH("material", primitive->material, integer);
    JSON_MATCH("mode",     primitive->mode,     integer);
    STRING_CASE_C("attributes")
      Vector(Gltf_Attribute) attributes;
      vector_init(&attributes, 0, 8, allocator);
      JSON_PARSE_BLOCK(status, parser) {
        if (parser->value.kind != Json_Value_Number) {
          return false;
        }
        String name = parser->name;
        Gltf_Attribute attribute = {
          .accessor = parser->value.integer,
        };
        const static struct { String name; b8 numbered; } attribute_strings[] = {
          [Gltf_Attribute_Type_Position]  = {
            .name = LIT("POSITION"),
          },
          [Gltf_Attribute_Type_Normal]    = {
            .name = LIT("NORMAL"),
          },
          [Gltf_Attribute_Type_Tangent]   = {
            .name = LIT("TANGENT"),
          },
          [Gltf_Attribute_Type_Tex_Coord] = {
            .name     = LIT("TEXCOORD_"),
            .numbered = true,
          },
          [Gltf_Attribute_Type_Color]     = {
            .name     = LIT("COLOR_"),
            .numbered = true,
          },
          [Gltf_Attribute_Type_Joints]    = {
            .name     = LIT("JOINTS_"),
            .numbered = true,
          },
          [Gltf_Attribute_Type_Weights]   = {
            .name     = LIT("WEIGHTS_"),
            .numbered = true,
          },
        };
        b8 found = false;
        for_range(i, 0, count_of(attribute_strings)) {
          if (attribute_strings[i].numbered) {
            if (string_has_prefix(name, attribute_strings[i].name)) {
              attribute.type = i;
              found = true;

              name = slice_start(name, attribute_strings[i].name.len);
              attribute.index = or_goto(parse_isize(name), fail);
              break;
            }
          } else {
            if (string_equal(attribute_strings[i].name, name)) {
              attribute.type = i;
              found = true;
              break;
            }
          }
        }
        if (!found) {
          return false;
        }
        vector_append(&attributes, attribute);
      }
      primitive->attributes = vector_to_slice(type_of(primitive->attributes), attributes);
    STRING_DEFAULT()
      json_parser_skip(parser);
    STRING_SWITCH_END()
  }

  if (primitive->mode > 6 || primitive->mode < 0) {
    return false;
  }

  return true;

fail:
  return false;
}

internal b8 gltf_parse_meshes(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Mesh) meshes;
  vector_init(&meshes, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Mesh mesh = {0};

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name", mesh.name, string);
      STRING_CASE_C("primitives")
        if (parser->value.kind != Json_Value_Array) {
          goto fail;
        }
        Vector(Gltf_Primitive) primitives;
        vector_init(&primitives, 0, 8, allocator);
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Object) {
            goto fail;
          }
          Gltf_Primitive primitive = {0};
          if (!gltf_parse_primitive(parser, &primitive, allocator)) {
            goto fail;
          }
          vector_append(&primitives, primitive);
        }
        mesh.primitives = vector_to_slice(type_of(mesh.primitives), primitives);
      STRING_CASE_C("weights")
        if (parser->value.kind != Json_Value_Array) {
          goto fail;
        }
        Vector(f32) weights;
        vector_init(&weights, 0, 8, allocator);
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            goto fail;
          }

          vector_append(&weights, parser->value.number);
        }
        mesh.weights = vector_to_slice(type_of(mesh.weights), weights);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&meshes, mesh);
  }

  gltf->meshes = vector_to_slice(type_of(gltf->meshes), meshes);

  return true;

fail:
  vector_delete(meshes);
  return false;
}

internal b8 gltf_parse_cameras(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Camera) cameras;
  vector_init(&cameras, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Camera camera = {0};

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name", camera.name, string);
      STRING_CASE_C("type")
        if (string_equal(parser->value.string, LIT("perspective"))) {
          camera.is_orthographic = false;
        } else if (string_equal(parser->value.string, LIT("orthographic"))){
          camera.is_orthographic = true;
        } else {
          return false;
        }
      STRING_CASE_C("perspective")
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("znear",       camera.perspective.z_near,      number);
          JSON_MATCH("zfar",        camera.perspective.z_far,       number);
          JSON_MATCH("aspectRatio", camera.perspective.aspect_ratio,number);
          JSON_MATCH("yfov",        camera.perspective.y_fov,       number);
          STRING_DEFAULT()
            json_parser_advance(parser);
          STRING_SWITCH_END();
        }
      STRING_CASE_C("orthographic")
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("znear", camera.orthographic.z_near, number);
          JSON_MATCH("zfar",  camera.orthographic.z_far,  number);
          JSON_MATCH("xmag",  camera.orthographic.x_mag,  number);
          JSON_MATCH("ymag",  camera.orthographic.y_mag,  number);
          STRING_DEFAULT()
            json_parser_advance(parser);
          STRING_SWITCH_END();
        }
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&cameras, camera);
  }

  gltf->cameras = vector_to_slice(type_of(gltf->cameras), cameras);

  return true;

fail:
  vector_delete(cameras);
  return false;
}

internal b8 gltf_parse_samplers(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Sampler) samplers;
  vector_init(&samplers, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Sampler sampler = {
      .mag_filter = Gltf_Mag_Filter_Linear,
      .min_filter = Gltf_Min_Filter_Nearest_Mipmap_Linear,
      .wrap_s     = Gltf_Texture_Wrap_Repeat,
      .wrap_t     = Gltf_Texture_Wrap_Repeat,
    };

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",      sampler.name,       string);
      JSON_MATCH("magFilter", sampler.mag_filter, integer);
      JSON_MATCH("minFilter", sampler.min_filter, integer);
      JSON_MATCH("wrapS",     sampler.wrap_s,     integer);
      JSON_MATCH("wrapT",     sampler.wrap_t,     integer);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    if (
      !enum_is_valid(Gltf_Mag_Filter,   sampler.mag_filter) ||
      !enum_is_valid(Gltf_Min_Filter,   sampler.min_filter) ||
      !enum_is_valid(Gltf_Texture_Wrap, sampler.wrap_s)     ||
      !enum_is_valid(Gltf_Texture_Wrap, sampler.wrap_t)
    ) {
      return false;
    }

    vector_append(&samplers, sampler);
  }

  gltf->samplers = vector_to_slice(type_of(gltf->samplers), samplers);

  return true;

fail:
  vector_delete(samplers);
  return false;
}

internal b8 gltf_parse_textures(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Texture) textures;
  vector_init(&textures, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Texture texture = {0};

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",    texture.name,    string);
      JSON_MATCH("sampler", texture.sampler, integer);
      JSON_MATCH("source",  texture.source,  integer);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&textures, texture);
  }

  gltf->textures = vector_to_slice(type_of(gltf->textures), textures);

  return true;

fail:
  vector_delete(textures);
  return false;
}

internal b8 gltf_parse_images(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  Vector(Gltf_Image) images;
  vector_init(&images, 0, 8, allocator);

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      goto fail;
    }

    Gltf_Image image = {
      .buffer_view = -1,
    };

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",       image.name,        string);
      JSON_MATCH("mimeType",   image.mime_type,   string);
      JSON_MATCH("uri",        image.uri,         string);
      JSON_MATCH("bufferView", image.buffer_view, integer);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    vector_append(&images, image);
  }

  gltf->images = vector_to_slice(type_of(gltf->images), images);

  return true;

fail:
  vector_delete(images);
  return false;
}

extern b8 gltf_parse_file(String data, Gltf_File *gltf, Allocator allocator) {
  Json_Parser parser = {0};
  Json_Status status = json_parser_init(&parser, data, allocator);

  *gltf = (Gltf_File) {0};

  JSON_PARSE_BLOCK(status, &parser) {
    STRING_SWITCH(parser.name)
    STRING_CASE_C("asset")
      if (!gltf_parse_asset(gltf, &parser)) {
        return false;
      }
    STRING_CASE_C("scene")
      if (parser.value.kind != Json_Value_Number) {
        return false;
      }
      gltf->scene = parser.value.integer;
    STRING_CASE_C("scenes")
      if (!gltf_parse_scenes(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("nodes")
      if (!gltf_parse_nodes(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("materials")
      if (!gltf_parse_materials(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("meshes")
      if (!gltf_parse_meshes(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("accessors")
      if (!gltf_parse_accessors(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("bufferViews")
      if (!gltf_parse_buffer_views(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("buffers")
      if (!gltf_parse_buffers(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("samplers")
      if (!gltf_parse_samplers(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("cameras")
      if (!gltf_parse_cameras(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("textures")
      if (!gltf_parse_textures(gltf, &parser, allocator)) {
        return false;
      }
    STRING_CASE_C("images")
      if (!gltf_parse_images(gltf, &parser, allocator)) {
        return false;
      }
    STRING_DEFAULT()
      json_parser_skip(&parser);
    STRING_SWITCH_END()
  }

  return status == Json_Status_End;
}

typedef struct {
  u32 magic, version, length;
} Glb_File_Header;

typedef struct {
  u32 length, type;
  u8  data[0];
} Glb_Chunk_Header;

typedef enum {
  Glb_Chunk_Type_Json,
  Glb_Chunk_Type_Binary,
} Glb_Chunk_Type;

internal b8 gltf_parse_glb(Byte_Slice data, String path, Gltf_File *file, Allocator allocator) {
  file->path = path;

  Glb_File_Header file_header;
  if (data.len < size_of(file_header)) {
    return false;
  }
  mem_copy(&file_header, data.data, size_of(file_header));
  if (file_header.magic != 0x46546C67) {
    return false;
  }

  if (file_header.length != data.len) {
    return false;
  }

  isize cursor = size_of(file_header);
  Glb_Chunk_Header chunk_header;
  if (cursor + size_of(chunk_header) > data.len) {
    return false;
  }
  mem_copy(&chunk_header, data.data + cursor, size_of(chunk_header));
  if (chunk_header.type != 0x4E4F534A) {
    return false;
  }
  if (cursor + chunk_header.length > data.len) {
    return false;
  }

  String json_data = {
    .data = (char *)chunk_header.data,
    .len  = chunk_header.length,
  };

  b8 json_ok = gltf_parse_file(json_data, file, allocator);
  if (!json_ok) {
    return false;
  }

  cursor += size_of(chunk_header) + chunk_header.length;

  if (cursor == data.len) {
    return true;
  }

  if (cursor + size_of(chunk_header) < data.len) {
    return false;
  }
  mem_copy(&chunk_header, data.data + cursor, size_of(chunk_header));
  if (chunk_header.type != 0x004E4942) {
    return false;
  }
  if (cursor + chunk_header.length > data.len) {
    return false;
  }
  file->glb_data = (Byte_Slice) {
    .data = chunk_header.data,
    .len  = chunk_header.length,
  };

  return true;
}

internal b8 gltf_parse(Byte_Slice data, String path, Gltf_File *gltf, Allocator allocator) {
  if (data.len < 4) {
    return false;
  }
  u32 magic;
  mem_copy(&magic, data.data, size_of(magic));

  if (magic == 0x46546C67) {
    return gltf_parse_glb(data, path, gltf, allocator);
  } else {
    return gltf_parse_file(bytes_to_string(data), gltf, allocator);
  }
}

#define GLTF_ACCESSOR_TYPE_u8  Gltf_Component_Type_Unsigned_Byte
#define GLTF_ACCESSOR_TYPE_i8  Gltf_Component_Type_Byte
#define GLTF_ACCESSOR_TYPE_u16 Gltf_Component_Type_Unsigned_Short
#define GLTF_ACCESSOR_TYPE_i16 Gltf_Component_Type_Short
#define GLTF_ACCESSOR_TYPE_u32 Gltf_Component_Type_Unsigned_Int
#define GLTF_ACCESSOR_TYPE_f32 Gltf_Component_Type_Float

#define GLTF_ACCESSOR_READ(TYPE)                                               \
  internal void gltf_accessor_read_##TYPE##s(                                  \
    Gltf_File     const *file,                                                 \
    Gltf_Accessor const *accessor,                                             \
    isize                offset,                                               \
    isize                count,                                                \
    TYPE                  *data                                                \
  ) {                                                                          \
    assert(GLTF_ACCESSOR_TYPE_##TYPE == accessor->component_type);             \
    Gltf_Buffer_View view   = IDX(file->buffer_views, accessor->buffer_view);  \
    Gltf_Buffer      buffer = IDX(file->buffers, view.buffer);                 \
    offset = offset * size_of(*data);                                          \
    assert(offset >= 0);                                                       \
    assert(accessor->byte_offset + offset <= view.byte_length);                \
    offset = offset + accessor->byte_offset + view.byte_offset;                \
    rawptr src = (rawptr)(                                                     \
      (uintptr)buffer.data.data +                                              \
      offset                                                                   \
    );                                                                         \
    mem_copy(data, src, count * size_of(data[0]));                             \
  }                                                                            \

GLTF_ACCESSOR_READ(u8);
GLTF_ACCESSOR_READ(u16);
GLTF_ACCESSOR_READ(u32);
GLTF_ACCESSOR_READ(i8);
GLTF_ACCESSOR_READ(i16);
GLTF_ACCESSOR_READ(f32);

internal b8 load_uri_data(String path, String uri, Byte_Slice *data, Allocator allocator) {
  Uri_Scheme scheme = uri_get_scheme(uri);

  b8 uri_ok;

  String file_path;
  switch (scheme) {
  CASE Uri_Scheme_Data:
    Uri_Data uri_data;
    uri_ok = uri_data_parse(uri, &uri_data, allocator);
    if (!uri_ok) {
      return false;
    }
    *data = uri_data.data;
    return true;
  CASE Uri_Scheme_File:
    Uri_File uri_file;
    uri_ok = uri_file_parse(uri, &uri_file);
    if (!uri_ok) {
      return false;
    }
    file_path = uri_file.path;
  CASE Uri_Scheme_Unknown:
    file_path = uri;
  }

  OS_Result_Bytes file = read_entire_file_path(file_path, allocator);
  if (file.err) {
    return false;
  }

  *data = file.value;

  return true;
}

extern b8 gltf_load_buffers(String path, Gltf_File *file, Allocator allocator) {
  slice_iter(file->buffers, buffer, i, {
    if (buffer->uri.len == 0 && i == 0 && file->glb_data.len != 0) {
      buffer->data = file->glb_data;
      continue;
    }
    if (!load_uri_data(path, buffer->uri, &buffer->data, allocator)) {
      return false;
    }
    if (buffer->data.len != buffer->byte_length) {
      return false;
    }
  });

  slice_iter(file->images, image, i, {
    if (image->buffer_view != -1) {
      return false;
    }
    if (!load_uri_data(path, image->uri, &image->data, allocator)) {
      return false;
    }
  });

  return true;
}

internal void node_to_triangles(
  Gltf_File const      *file,
  Gltf_Triangle_Vector *triangles,
  Gltf_Node const      *node,
  Matrix_4x4            transform
) {
  transform = matrix_4x4_mul(transform, node->matrix);

  Matrix_3x3 normal_matrix = matrix_3x3_transpose(matrix_3x3_inverse(matrix_4x4_to_3x3(transform)));

  if (node->mesh != -1) {
    Gltf_Mesh const *mesh = &IDX(file->meshes, node->mesh);
    for_range(primitive_i, 0, mesh->primitives.len) {
      Gltf_Primitive primitive = IDX(mesh->primitives, primitive_i);
    // slice_iter_v(mesh->primitives, primitive, j, {
      Gltf_Accessor const *tex_coords = nil;
      Gltf_Accessor const *positions  = nil;
      Gltf_Accessor const *normals    = nil;
      slice_iter_v(primitive.attributes, attribute, k, {
        if (attribute.type == Gltf_Attribute_Type_Tex_Coord && attribute.index == 0) {
          tex_coords = &IDX(file->accessors, attribute.accessor);
        }
        if (attribute.type == Gltf_Attribute_Type_Position) {
          positions  = &IDX(file->accessors, attribute.accessor);
        }
        if (attribute.type == Gltf_Attribute_Type_Normal) {
          normals    = &IDX(file->accessors, attribute.accessor);
        }
      });

      if (tex_coords == nil || positions == nil || normals == nil) {
        continue;
      }

      Gltf_Triangle triangle = { .material = primitive.material };

      if (primitive.has_indices) {
        Gltf_Accessor const *indices = &IDX(file->accessors, primitive.indices);

        for_range(i, 0, indices->count /  3) {
          u16 index_buf[3];
          gltf_accessor_read_u16s(file, indices, (i * 3), 3, &index_buf[0]);

          for_range(vi, 0, 3) {
            Gltf_Vertex *v = &triangle.vertices[vi];
            gltf_accessor_read_f32s(file, tex_coords, index_buf[vi] * 2, 2, &v->tex_coords.data[0]);
            gltf_accessor_read_f32s(file, positions,  index_buf[vi] * 3, 3, &v->position.data[0]);
            gltf_accessor_read_f32s(file, normals,    index_buf[vi] * 3, 3, &v->normal.data[0]);

            Vec4 pos = (Vec4) { .xyz = v->position }; pos.w = 1;
            v->position = matrix_4x4_mul_vec4(transform, pos).xyz;
            v->normal   = matrix_3x3_mul_vec3(normal_matrix, v->normal);
          }

          vector_append(triangles, triangle);
        }
        continue;
      }

      for_range(i, 0, positions->count / 3) {
        for_range(vi, 0, 3) {
          Gltf_Vertex *v = &triangle.vertices[vi];
          gltf_accessor_read_f32s(file, tex_coords, (i * 3 + vi) * 2, 2, &v->tex_coords.data[0]);
          gltf_accessor_read_f32s(file, positions,  (i * 3 + vi) * 3, 3, &v->position.data[0]);
          gltf_accessor_read_f32s(file, normals,    (i * 3 + vi) * 3, 3, &v->normal.data[0]);

          Vec4 pos = (Vec4) { .xyz = v->position }; pos.w = 1;
          v->position = matrix_4x4_mul_vec4(transform, pos).xyz;
          v->normal   = matrix_3x3_mul_vec3(normal_matrix, v->normal);
        }
        vector_append(triangles, triangle);
      }
    }
  }
  slice_iter_v(node->children, child, i, {
    node_to_triangles(file, triangles, &IDX(file->nodes, child), transform);
  });
}

extern b8 gltf_to_triangles(Gltf_File const *file, Gltf_Triangle_Vector *triangles) {
  Gltf_Scene const *scene = &IDX(file->scenes, file->scene);
  slice_iter_v(scene->nodes, node_index, i, {
    node_to_triangles(file, triangles, &IDX(file->nodes, node_index), MATRIX_4X4_IDENTITY);
  });

  return true;
}
