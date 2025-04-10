#include "codin/codin.h"

#include "codin/linalg.h"

typedef enum {
  Gltf_Component_Type_Byte           = 5120,
  Gltf_Component_Type_Unsigned_Byte  = 5121,
  Gltf_Component_Type_Short          = 5122,
  Gltf_Component_Type_Unsigned_Short = 5123,
  Gltf_Component_Type_Unsigned_Int   = 5125,
  Gltf_Component_Type_Float          = 5126,
} Gltf_Component_Type;

typedef enum {
  Gltf_Accessor_Type_Scalar,
  Gltf_Accessor_Type_Vec2,
  Gltf_Accessor_Type_Vec3,
  Gltf_Accessor_Type_Vec4,
  Gltf_Accessor_Type_Mat2,
  Gltf_Accessor_Type_Mat3,
} Gltf_Accessor_Type;

internal String gltf_accessor_type_strings[] = {
  [Gltf_Accessor_Type_Scalar]  = LIT("SCALAR"),
  [Gltf_Accessor_Type_Vec2]    = LIT("VEC2"),
  [Gltf_Accessor_Type_Vec3]    = LIT("VEC3"),
  [Gltf_Accessor_Type_Vec4]    = LIT("VEC4"),
  [Gltf_Accessor_Type_Mat2]    = LIT("MAT2"),
  [Gltf_Accessor_Type_Mat3]    = LIT("MAT3"),
};

typedef struct {
  String       name;
  isize        mesh, camera, skin;
  Slice(isize) children;
  Matrix_4x4   matrix;
  Slice(f32)   weights;
} Gltf_Node;

typedef struct {
  isize index;
  isize tex_coord;
} Gltf_Texture_Info;

typedef struct {
  Color4            base_color_factor;
  f32               metallic_factor;
  f32               roughness_factor;
  Gltf_Texture_Info base_color_texture;
  Gltf_Texture_Info metallic_roughness_texture;
} Gltf_PBR_Material;

typedef struct {
  String            name;
  Gltf_PBR_Material pbr_metallic_roughness;
  Gltf_Texture_Info emissive_texture;
  Gltf_Texture_Info normal_texture;
  Gltf_Texture_Info occlusion_texture;
  Color3            emissive_factor;
  String            alpha_mode;
  f32               alpha_cutoff;
  f32               occlusion_texture_strength;
  f32               normal_texture_scale;
  b8                double_sided;
} Gltf_Material;

typedef enum {
  Gltf_Attribute_Type_Position,
  Gltf_Attribute_Type_Normal,
  Gltf_Attribute_Type_Tangent,
  Gltf_Attribute_Type_Tex_Coord,
  Gltf_Attribute_Type_Color,
  Gltf_Attribute_Type_Joints,
  Gltf_Attribute_Type_Weights,
} Gltf_Attribute_Type;

typedef struct {
  Gltf_Attribute_Type type;
  isize               index;
  isize               accessor;
} Gltf_Attribute;

#define GLTF_PRIMITVE_MODES(X)             \
  X(Gltf_Primitive_Mode_Points,         0) \
  X(Gltf_Primitive_Mode_Lines,          1) \
  X(Gltf_Primitive_Mode_Line_Loop,      2) \
  X(Gltf_Primitive_Mode_Line_Strip,     3) \
  X(Gltf_Primitive_Mode_Triangles,      4) \
  X(Gltf_Primitive_Mode_Triangle_Strip, 5) \
  X(Gltf_Primitive_Mode_Triangle_Fan,   6)

X_ENUM_EXPLICIT(Gltf_Primitive_Mode, GLTF_PRIMITVE_MODES);

typedef struct {
  Slice(Gltf_Attribute) attributes;
  isize                 indices, material;
  Gltf_Primitive_Mode   mode;
  b8                    has_indices;
} Gltf_Primitive;

typedef struct {
  String                name;
  Slice(Gltf_Primitive) primitives;
  Slice(f32)            weights;
} Gltf_Mesh;

typedef struct {
  String              name;
  isize               buffer_view, byte_offset;
  Gltf_Component_Type component_type;
  Gltf_Accessor_Type  type;
  isize               count;
  f32                 min[16], max[16];
  b8                  has_min, has_max, normalized;
} Gltf_Accessor;

typedef struct {
  String name;
  isize  buffer, byte_offset, byte_length, byte_stride;
} Gltf_Buffer_View;

typedef struct {
  String     name, uri;
  isize      byte_length;
  Byte_Slice data;
} Gltf_Buffer;

typedef struct {
  String       name;
  Slice(isize) nodes;
} Gltf_Scene;

typedef struct {
  String name;
  String type;
  union {
    struct {
      f32 x_mag, y_mag;
      f32 z_far, z_near;
    } orthographic;
    struct {
      f32 aspect_ratio;
      f32 y_fov;
      f32 z_far, z_near;
    } perspective;
  };
  b8 is_orthographic;
} Gltf_Camera;

typedef struct {
  String version;
  String min_version;
  String generator;
  String copyright;
} Gltf_Asset;

#define GLTF_MAG_FILTERS(X)        \
  X(Gltf_Mag_Filter_Nearest, 9728) \
  X(Gltf_Mag_Filter_Linear,  9729) \

X_ENUM_EXPLICIT(Gltf_Mag_Filter, GLTF_MAG_FILTERS);

#define GLTF_MIN_FILTERS(X)                       \
  X(Gltf_Min_Filter_Nearest,                9728) \
  X(Gltf_Min_Filter_Linear,                 9729) \
  X(Gltf_Min_Filter_Nearest_Mipmap_Nearest, 9984) \
  X(Gltf_Min_Filter_Linear_Mipmap_Nearest,  9985) \
  X(Gltf_Min_Filter_Nearest_Mipmap_Linear,  9986) \
  X(Gltf_Min_Filter_Linear_Mipmap_Linear,   9987) \

X_ENUM_EXPLICIT(Gltf_Min_Filter, GLTF_MIN_FILTERS);

#define GLTF_TEXTURE_WRAPS(X)                 \
  X(Gltf_Texture_Wrap_Clamp_To_Edge,   33071) \
  X(Gltf_Texture_Wrap_Mirrored_Repeat, 33648) \
  X(Gltf_Texture_Wrap_Repeat,          10497) \

X_ENUM_EXPLICIT(Gltf_Texture_Wrap, GLTF_TEXTURE_WRAPS);

typedef struct {
  String            name;
  Gltf_Mag_Filter   mag_filter;
  Gltf_Min_Filter   min_filter;
  Gltf_Texture_Wrap wrap_s, wrap_t;
} Gltf_Sampler;

typedef struct {
  String     uri, name, mime_type;
  isize      buffer_view;
  Byte_Slice data;
} Gltf_Image;

typedef struct {
  String name;
  isize  sampler, source;
} Gltf_Texture;

typedef struct {
  String                  path;
  isize                   scene;
  Gltf_Asset              asset;
  Slice(Gltf_Node)        nodes;
  Slice(Gltf_Scene)       scenes;
  Slice(Gltf_Mesh)        meshes;
  Slice(Gltf_Image)       images;
  Slice(Gltf_Buffer)      buffers;
  Slice(Gltf_Camera)      cameras;
  Slice(Gltf_Texture)     textures;
  Slice(Gltf_Sampler)     samplers;
  Slice(Gltf_Material)    materials;
  Slice(Gltf_Accessor)    accessors;
  Slice(Gltf_Buffer_View) buffer_views;
  Byte_Slice              glb_data;
} Gltf_File;

typedef struct {
  Vec3 position, normal;
  Vec2 tex_coords;
} Gltf_Vertex;

typedef struct {
  Gltf_Vertex vertices[3];
  isize       material;
} Gltf_Triangle;

typedef Vector(Gltf_Triangle) Gltf_Triangle_Vector;
typedef Slice(Gltf_Triangle) Gltf_Triangle_Slice;
  
extern b8 gltf_parse_file(String data, Gltf_File *gltf, Allocator allocator);
extern b8 gltf_load_buffers(String path, Gltf_File *file, Allocator allocator);
extern b8 gltf_to_triangles(Gltf_File const *file, Gltf_Triangle_Vector *triangles);
