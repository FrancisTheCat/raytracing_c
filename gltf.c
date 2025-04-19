#include "gltf.h"

#include "codin/fmt.h"
#include "codin/json.h"
#include "codin/os.h"
#include "codin/fmt.h"
#include "codin/strings.h"

#include "uri.h"

#define JSON_PARSE_BLOCK(STATUS, PARSER)   \
  for (                                    \
    STATUS =  json_parser_advance(PARSER); \
    STATUS == Json_Status_Continue;        \
    STATUS =  json_parser_advance(PARSER)  \
  )

internal isize json_n_array_elems(Json_Parser const *parser) {
  assert(parser->value.kind == Json_Value_Object || parser->value.kind == Json_Value_Array);
  isize n = 0;

  Json_Parser parser_copy = *parser;
  
  Json_Status status;
  JSON_PARSE_BLOCK(status, &parser_copy) {
    json_parser_skip(&parser_copy);
    n += 1;
  }

  return n;
}

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
    return false;                                \
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
}

internal b8 gltf_parse_scenes(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }

  isize n_scenes = json_n_array_elems(parser);
  slice_init(&gltf->scenes, n_scenes, allocator);
  isize scene_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Scene *scene = &IDX(gltf->scenes, scene_i);
    scene_i += 1;

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name", scene->name, string);
      STRING_CASE_C("nodes")
        isize n_nodes = json_n_array_elems(parser);
        slice_init(&scene->nodes, n_nodes, allocator);
        isize node_i = 0;
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            return false;
          }
          IDX(scene->nodes, node_i) = parser->value.integer;
          node_i += 1;
        }
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }
  }

  return status == Json_Status_End;
}

internal b8 gltf_parse_string_slice(
  Gltf_File    *gltf,
  Json_Parser  *parser,
  String_Slice *s,
  Allocator     allocator
) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }

  isize n = json_n_array_elems(parser);
  slice_init(s, n, allocator);
  isize i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_String) {
      return false;
    }

    IDX(*s, i) = parser->value.string;
  }

  return true;
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

  isize n_nodes = json_n_array_elems(parser);
  slice_init(&gltf->nodes, n_nodes, allocator);
  isize node_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }
    Gltf_Node *node = &IDX(gltf->nodes, node_i);
    node_i += 1;
    *node = (Gltf_Node) {
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
      JSON_MATCH("name",   node->name,   string);
      JSON_MATCH("mesh",   node->mesh,   integer);
      JSON_MATCH("camera", node->camera, integer);
      JSON_MATCH("skin",   node->skin,   integer);
      STRING_CASE_C("rotation")
        if (!gltf_parse_float_slice(parser, rotation.data, count_of(rotation.data))) {
          return false;
        }
      STRING_CASE_C("translation")
        if (!gltf_parse_float_slice(parser, translation.data, count_of(translation.data))) {
          return false;
        }
      STRING_CASE_C("scale")
        if (!gltf_parse_float_slice(parser, scale.data, count_of(scale.data))) {
          return false;
        }
      STRING_CASE_C("matrix")
        if (!gltf_parse_float_slice(parser, node->matrix.data, count_of(node->matrix.data))) {
          return false;
        }
        explicit_matrix = true;
      STRING_CASE_C("weights")
        if (parser->value.kind != Json_Value_Array) {
          return false;
        }
        isize n_weights = json_n_array_elems(parser);
        slice_init(&node->weights, n_weights, allocator);
        isize weight_i = 0;
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            slice_delete(node->weights, allocator);
            return false;
          }

          IDX(node->weights, weight_i) = parser->value.number;
          weight_i += 1;
        }
      STRING_CASE_C("children")
        if (parser->value.kind != Json_Value_Array) {
          return false;
        }
        isize n_children = json_n_array_elems(parser);
        slice_init(&node->children, n_children, allocator);
        isize child_i = 0;
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            return false;
          }

          IDX(node->children, child_i) = parser->value.integer;
          child_i += 1;
        }
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    if (!explicit_matrix) {
      node->matrix = matrix_4x4_translation_rotation_scale(translation, rotation, scale);
    }
  }

  return status == Json_Status_End;
}

internal b8 gltf_parse_texture(Gltf_Texture_Info *texture, Json_Parser *parser) {
  if (parser->value.kind != Json_Value_Object) {
    return false;
  }
  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    STRING_SWITCH(parser->name)
    JSON_MATCH("index",    texture->index,     integer);
    JSON_MATCH("texCoord", texture->tex_coord, integer);
    STRING_DEFAULT()
      json_parser_skip(parser);
    STRING_SWITCH_END()
  }

  return true;
}

internal b8 gltf_parse_pbr(Gltf_Material *mat, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Object) {
    return false;
  }

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    STRING_SWITCH(parser->name)
    JSON_MATCH("metallicFactor",  mat->metallic,  number);
    JSON_MATCH("roughnessFactor", mat->roughness, number);
    STRING_CASE_C("baseColorFactor")
      if (parser->value.kind != Json_Value_Array) {
        return false;
      }
      isize i = 0;
      JSON_PARSE_BLOCK(status, parser) {
        if (parser->value.kind != Json_Value_Number || i > 4) {
          return false;
        }
        mat->base_color.data[i] = parser->value.number;
                
        i += 1;
      }

    STRING_CASE_C("baseColorTexture")
      if (!gltf_parse_texture(&mat->texture_base_color, parser)) {
        return false;
      }
      
    STRING_CASE_C("metallicRoughnessTexture")
      if (!gltf_parse_texture(&mat->texture_metallic_roughness, parser)) {
        return false;
      }
      
    STRING_DEFAULT()
      json_parser_skip(parser);
    STRING_SWITCH_END()
  }

  return status == Json_Status_End;
}

internal b8 gltf_parse_materials(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }

  isize n_materials = json_n_array_elems(parser);
  slice_init(&gltf->materials, n_materials, allocator);
  isize material_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }
    Gltf_Material material = GLTF_MATERIAL_DEFAULT;
    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",        material.name,         string);
      JSON_MATCH("doubleSided", material.double_sided, boolean);
      JSON_MATCH("alphaMode",   material.alpha_mode,   string);
      JSON_MATCH("alphaCutoff", material.alpha_cutoff, number);
      STRING_CASE_C("emissiveFactor")
        if (parser->value.kind != Json_Value_Array) {
          return false;
        }
        isize i = 0;
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number || i > 3) {
            return false;
          }
          material.emissive.data[i] = parser->value.number;
          
          i += 1;
        }

      STRING_CASE_C("emissiveTexture")
        if (!gltf_parse_texture(&material.texture_emissive, parser)) {
          return false;
        }
        
      STRING_CASE_C("normalTexture")
        if (parser->value.kind != Json_Value_Object) {
          return false;
        }
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("index",    material.texture_normal.index,     integer);
          JSON_MATCH("texCoord", material.texture_normal.tex_coord, integer);
          JSON_MATCH("scale",    material.texture_normal_scale,     integer);
          STRING_DEFAULT()
            json_parser_skip(parser);
          STRING_SWITCH_END()
        }
        
      STRING_CASE_C("occlusionTexture")
        if (parser->value.kind != Json_Value_Object) {
          return false;
        }
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("index",    material.texture_occlusion.index,     integer);
          JSON_MATCH("texCoord", material.texture_occlusion.tex_coord, integer);
          JSON_MATCH("strength", material.texture_occlusion_strength,  integer);
          STRING_DEFAULT()
            json_parser_skip(parser);
          STRING_SWITCH_END()
        }
        
      STRING_CASE_C("pbrMetallicRoughness")
        if (!gltf_parse_pbr(&material, parser, allocator)) {
          return false;
        }
        
      STRING_CASE_C("extensions")
        if (parser->value.kind != Json_Value_Object) {
          return false;
        }
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Object) {
            return false;
          }
          STRING_SWITCH(parser->name)
          STRING_CASE_C("KHR_materials_sheen")
            JSON_PARSE_BLOCK(status, parser) {
              STRING_SWITCH(parser->name)
              JSON_MATCH("sheenRoughnessFactor", material.sheen_roughness, number);
              STRING_CASE_C("sheenColorFactor")
                if (!gltf_parse_float_slice(parser, &material.sheen_color.data[0], 3)) {
                  return false;
                }
              STRING_CASE_C("sheenColorTexture")
                if (!gltf_parse_texture(&material.texture_sheen, parser)) {
                  return false;
                }
              STRING_CASE_C("sheenRoughnessTexture")
                if (!gltf_parse_texture(&material.texture_sheen_roughness, parser)) {
                  return false;
                }
              STRING_DEFAULT()
                json_parser_skip(parser);
              STRING_SWITCH_END()
            }
          STRING_CASE_C("KHR_materials_anisotropy")
            JSON_PARSE_BLOCK(status, parser) {
              STRING_SWITCH(parser->name)
              JSON_MATCH("anisotropyStrength", material.anisotropy_strength, number);
              JSON_MATCH("anisotropyRotation", material.anisotropy_rotation, number);
              STRING_CASE_C("anisotropyTexture")
                if (!gltf_parse_texture(&material.texture_anisotropy, parser)) {
                  return false;
                }
              STRING_DEFAULT()
                json_parser_skip(parser);
              STRING_SWITCH_END()
            }
          STRING_CASE_C("KHR_materials_clearcoat")
            JSON_PARSE_BLOCK(status, parser) {
              STRING_SWITCH(parser->name)
              JSON_MATCH("clearcoatFactor",          material.clearcoat,           number);
              JSON_MATCH("clearcoatRoughnessFactor", material.clearcoat_roughness, number);
              STRING_CASE_C("clearcoatTexture")
                if (!gltf_parse_texture(&material.texture_clearcoat, parser)) {
                  return false;
                }
              STRING_CASE_C("clearcoatTextureRoughness")
                if (!gltf_parse_texture(&material.texture_clearcoat_roughness, parser)) {
                  return false;
                }
              STRING_DEFAULT()
                json_parser_skip(parser);
              STRING_SWITCH_END()
            }
          STRING_CASE_C("KHR_materials_emissive_strength")
            JSON_PARSE_BLOCK(status, parser) {
              STRING_SWITCH(parser->name)
              JSON_MATCH("emissiveStrength", material.emissive_strength, number);
              STRING_DEFAULT()
                json_parser_skip(parser);
              STRING_SWITCH_END()
            }
          STRING_CASE_C("KHR_materials_ior")
            JSON_PARSE_BLOCK(status, parser) {
              STRING_SWITCH(parser->name)
              JSON_MATCH("number", material.ior, number);
              STRING_DEFAULT()
                json_parser_skip(parser);
              STRING_SWITCH_END()
            }
          STRING_CASE_C("KHR_materials_specular")
            JSON_PARSE_BLOCK(status, parser) {
              STRING_SWITCH(parser->name)
              JSON_MATCH("specularFactor", material.specular, number);
              STRING_CASE_C("specularColorFactor")
                if (!gltf_parse_float_slice(parser, &material.specular_color.data[0], 3)) {
                  return false;
                }
              STRING_CASE_C("specularTexture")
                if (!gltf_parse_texture(&material.texture_specular, parser)) {
                  return false;
                }
              STRING_CASE_C("specularTextureRoughness")
                if (!gltf_parse_texture(&material.texture_specular_roughness, parser)) {
                  return false;
                }
              STRING_DEFAULT()
                json_parser_skip(parser);
              STRING_SWITCH_END()
            }
          STRING_DEFAULT()
            json_parser_skip(parser);
          STRING_SWITCH_END()
        }
        
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    IDX(gltf->materials, material_i) = material;
    material_i += 1;
  }

  return status == Json_Status_End;
}

internal b8 gltf_parse_accessors(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }

  isize n_accessors = json_n_array_elems(parser);
  slice_init(&gltf->accessors, n_accessors, allocator);
  isize accessor_i = 0;
  
  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Accessor accessor = {
      .buffer_view = -1,
    };

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
          return false;
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
          return false;
        }
        
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    IDX(gltf->accessors, accessor_i) = accessor;
    accessor_i += 1;
  }

  return true;
}

internal b8 gltf_parse_buffer_views(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }

  isize n_buffer_views = json_n_array_elems(parser);
  slice_init(&gltf->buffer_views, n_buffer_views, allocator);
  isize buffer_view_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
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

    IDX(gltf->buffer_views, buffer_view_i) = buffer_view;
    buffer_view_i += 1;
  }

  return true;
}

internal b8 gltf_parse_buffers(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  isize n_buffers = json_n_array_elems(parser);
  slice_init(&gltf->buffers, n_buffers, allocator);
  isize buffer_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
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

    IDX(gltf->buffers, buffer_i) = buffer;
    buffer_i += 1;
  }

  return true;
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
        return false;
      }
      primitive->indices     = parser->value.integer;
      primitive->has_indices = true;
    JSON_MATCH("material", primitive->material, integer);
    JSON_MATCH("mode",     primitive->mode,     integer);
    STRING_CASE_C("attributes")
      if (parser->value.kind != Json_Value_Object) {
        return false;
      }
      isize n_attributes = json_n_array_elems(parser);
      slice_init(&primitive->attributes, n_attributes, allocator);
      isize attribute_i = 0;

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
              attribute.index = or_return(parse_isize(name), false);
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
        IDX(primitive->attributes, attribute_i) = attribute;
        attribute_i += 1;
      }
    STRING_DEFAULT()
      json_parser_skip(parser);
    STRING_SWITCH_END()
  }

  if (primitive->mode > 6 || primitive->mode < 0) {
    return false;
  }

  return true;
}

internal b8 gltf_parse_meshes(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  isize n_meshes = json_n_array_elems(parser);
  slice_init(&gltf->meshes, n_meshes, allocator);
  isize mesh_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Mesh *mesh = &IDX(gltf->meshes, mesh_i);
    mesh_i += 1;

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name", mesh->name, string);
      STRING_CASE_C("primitives")
        if (parser->value.kind != Json_Value_Array) {
          return false;
        }
        isize n_primitives = json_n_array_elems(parser);
        slice_init(&mesh->primitives, n_primitives, allocator);
        isize primitive_i = 0;
        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Object) {
            return false;
          }

          Gltf_Primitive *primitive = &IDX(mesh->primitives, primitive_i);
          primitive_i += 1;

          if (!gltf_parse_primitive(parser, primitive, allocator)) {
            return false;
          }
        }
      STRING_CASE_C("weights")
        if (parser->value.kind != Json_Value_Array) {
          return false;
        }

        isize n_weights = json_n_array_elems(parser);
        slice_init(&mesh->weights, n_weights, allocator);
        isize weight_i = 0;

        JSON_PARSE_BLOCK(status, parser) {
          if (parser->value.kind != Json_Value_Number) {
            return false;
          }

          IDX(mesh->weights, weight_i) = parser->value.number;
          weight_i += 1;
        }
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }
  }

  return true;
}

internal b8 gltf_parse_cameras(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  isize n_cameras = json_n_array_elems(parser);
  slice_init(&gltf->cameras, n_cameras, allocator);
  isize camera_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Camera *camera = &IDX(gltf->cameras, camera_i);
    camera_i += 1;

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name", camera->name, string);
      STRING_CASE_C("type")
        if (string_equal(parser->value.string, LIT("perspective"))) {
          camera->is_orthographic = false;
        } else if (string_equal(parser->value.string, LIT("orthographic"))){
          camera->is_orthographic = true;
        } else {
          return false;
        }
      STRING_CASE_C("perspective")
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("znear",       camera->perspective.z_near,      number);
          JSON_MATCH("zfar",        camera->perspective.z_far,       number);
          JSON_MATCH("aspectRatio", camera->perspective.aspect_ratio,number);
          JSON_MATCH("yfov",        camera->perspective.y_fov,       number);
          STRING_DEFAULT()
            json_parser_advance(parser);
          STRING_SWITCH_END();
        }
      STRING_CASE_C("orthographic")
        JSON_PARSE_BLOCK(status, parser) {
          STRING_SWITCH(parser->name)
          JSON_MATCH("znear", camera->orthographic.z_near, number);
          JSON_MATCH("zfar",  camera->orthographic.z_far,  number);
          JSON_MATCH("xmag",  camera->orthographic.x_mag,  number);
          JSON_MATCH("ymag",  camera->orthographic.y_mag,  number);
          STRING_DEFAULT()
            json_parser_advance(parser);
          STRING_SWITCH_END();
        }
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }
  }

  return true;
}

internal b8 gltf_parse_samplers(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  isize n_samplers = json_n_array_elems(parser);
  slice_init(&gltf->samplers, n_samplers, allocator);
  isize sampler_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Sampler *sampler = &IDX(gltf->samplers, sampler_i);
    sampler_i += 1;

    *sampler = (Gltf_Sampler) {
      .mag_filter = Gltf_Mag_Filter_Linear,
      .min_filter = Gltf_Min_Filter_Nearest_Mipmap_Linear,
      .wrap_s     = Gltf_Texture_Wrap_Repeat,
      .wrap_t     = Gltf_Texture_Wrap_Repeat,
    };

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",      sampler->name,       string);
      JSON_MATCH("magFilter", sampler->mag_filter, integer);
      JSON_MATCH("minFilter", sampler->min_filter, integer);
      JSON_MATCH("wrapS",     sampler->wrap_s,     integer);
      JSON_MATCH("wrapT",     sampler->wrap_t,     integer);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }

    if (
      !enum_is_valid(Gltf_Mag_Filter,   sampler->mag_filter) ||
      !enum_is_valid(Gltf_Min_Filter,   sampler->min_filter) ||
      !enum_is_valid(Gltf_Texture_Wrap, sampler->wrap_s)     ||
      !enum_is_valid(Gltf_Texture_Wrap, sampler->wrap_t)
    ) {
      return false;
    }
  }

  return true;
}

internal b8 gltf_parse_textures(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  isize n_textures = json_n_array_elems(parser);
  slice_init(&gltf->textures, n_textures, allocator);
  isize texture_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Texture *texture = &IDX(gltf->textures, texture_i);
    texture_i += 1;

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",    texture->name,    string);
      JSON_MATCH("sampler", texture->sampler, integer);
      JSON_MATCH("source",  texture->source,  integer);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }
  }

  return true;
}

internal b8 gltf_parse_images(Gltf_File *gltf, Json_Parser *parser, Allocator allocator) {
  if (parser->value.kind != Json_Value_Array) {
    return false;
  }
  
  isize n_images = json_n_array_elems(parser);
  slice_init(&gltf->images, n_images, allocator);
  isize image_i = 0;

  Json_Status status;
  JSON_PARSE_BLOCK(status, parser) {
    if (parser->value.kind != Json_Value_Object) {
      return false;
    }

    Gltf_Image *image = &IDX(gltf->images, image_i);
    image_i += 1;

    *image = (Gltf_Image) {
      .buffer_view = -1,
    };

    JSON_PARSE_BLOCK(status, parser) {
      STRING_SWITCH(parser->name)
      JSON_MATCH("name",       image->name,        string);
      JSON_MATCH("mimeType",   image->mime_type,   string);
      JSON_MATCH("uri",        image->uri,         string);
      JSON_MATCH("bufferView", image->buffer_view, integer);
      STRING_DEFAULT()
        json_parser_skip(parser);
      STRING_SWITCH_END()
    }
  }

  return true;
}

extern void gltf_file_destroy(Gltf_File *gltf, Allocator allocator) {
  slice_iter_v(gltf->scenes, scene, i, slice_delete(scene.nodes,     allocator));
  slice_iter_v(gltf->nodes,  node,  i, slice_delete(node.children,   allocator); slice_delete(node.weights, allocator));
  slice_iter_v(gltf->meshes, mesh,  i, {
    slice_iter_v(mesh.primitives, primitive,  j, {
      slice_delete(primitive.attributes, allocator);
    });
    slice_delete(mesh.primitives, allocator);
    slice_delete(mesh.weights,    allocator)
  });

  slice_delete(gltf->scenes,              allocator);
  slice_delete(gltf->nodes,               allocator);
  slice_delete(gltf->materials,           allocator);
  slice_delete(gltf->meshes,              allocator);
  slice_delete(gltf->accessors,           allocator);
  slice_delete(gltf->buffer_views,        allocator);
  slice_delete(gltf->buffers,             allocator);
  slice_delete(gltf->samplers,            allocator);
  slice_delete(gltf->cameras,             allocator);
  slice_delete(gltf->textures,            allocator);
  slice_delete(gltf->images,              allocator);
  slice_delete(gltf->extensions_used,     allocator);
  slice_delete(gltf->extensions_required, allocator);
}

extern b8 gltf_parse_file(String data, Gltf_File *gltf, Allocator allocator) {
  Json_Parser parser = {0};
  Json_Status status = json_parser_init(&parser, data, allocator);

  *gltf = (Gltf_File) {0};

  JSON_PARSE_BLOCK(status, &parser) {
    STRING_SWITCH(parser.name)
    STRING_CASE_C("asset")
      if (!gltf_parse_asset(gltf, &parser)) {
        goto fail;
      }
    STRING_CASE_C("extensionsRequired")
      if (!gltf_parse_string_slice(gltf, &parser, &gltf->extensions_required, allocator)) {
        return false;
      }
    STRING_CASE_C("extensionsUsed")
      if (!gltf_parse_string_slice(gltf, &parser, &gltf->extensions_used, allocator)) {
        return false;
      }
    STRING_CASE_C("scene")
      if (parser.value.kind != Json_Value_Number) {
        goto fail;
      }
      gltf->scene = parser.value.integer;
    STRING_CASE_C("scenes")
      if (!gltf_parse_scenes(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("nodes")
      if (!gltf_parse_nodes(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("materials")
      if (!gltf_parse_materials(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("meshes")
      if (!gltf_parse_meshes(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("accessors")
      if (!gltf_parse_accessors(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("bufferViews")
      if (!gltf_parse_buffer_views(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("buffers")
      if (!gltf_parse_buffers(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("samplers")
      if (!gltf_parse_samplers(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("cameras")
      if (!gltf_parse_cameras(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("textures")
      if (!gltf_parse_textures(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_CASE_C("images")
      if (!gltf_parse_images(gltf, &parser, allocator)) {
        goto fail;
      }
    STRING_DEFAULT()
      json_parser_skip(&parser);
    STRING_SWITCH_END()
  }

  return status == Json_Status_End;

fail:
  gltf_file_destroy(gltf, allocator);
  return false;
}

typedef struct {
  u32 magic, version, length;
} Glb_File_Header;

typedef struct {
  u32 length, type;
} Glb_Chunk_Header;

typedef enum {
  Glb_Chunk_Type_Json,
  Glb_Chunk_Type_Binary,
} Glb_Chunk_Type;

extern b8 gltf_parse_glb(Byte_Slice data, String path, Gltf_File *file, Allocator allocator) {
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
  cursor += size_of(chunk_header);
  if (cursor + chunk_header.length > data.len) {
    return false;
  }

  String json_data = {
    .data = (char *)data.data + cursor,
    .len  = chunk_header.length,
  };

  b8 json_ok = gltf_parse_file(json_data, file, allocator);
  if (!json_ok) {
    return false;
  }

  cursor += chunk_header.length;

  if (cursor == data.len) {
    return true;
  }

  if (cursor + size_of(chunk_header) > data.len) {
    return false;
  }
  mem_copy(&chunk_header, data.data + cursor, size_of(chunk_header));
  if (chunk_header.type != 0x004E4942) {
    return false;
  }
  cursor += size_of(chunk_header);
  if (cursor + chunk_header.length > data.len) {
    return false;
  }
  file->glb_data = (Byte_Slice) {
    .data = data.data + cursor,
    .len  = chunk_header.length,
  };

  return true;
}

extern b8 gltf_parse(Byte_Slice data, String path, Gltf_File *gltf, Allocator allocator) {
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
    offset = offset + accessor->byte_offset + view.byte_offset;                \
    assert(offset <= buffer.byte_length);                                      \
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
      Gltf_Buffer_View *view   = &IDX(file->buffer_views, image->buffer_view);
      Gltf_Buffer      *buffer = &IDX(file->buffers, view->buffer);
      image->data = (Byte_Slice) {
        .data = buffer->data.data + view->byte_offset,
        .len  = view->byte_length,
      };
      continue;
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

        for_range(i, 0, indices->count / 3) {
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
