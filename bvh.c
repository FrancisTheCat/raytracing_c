#include "raytracer.h"

typedef struct {
  i32 index: 31;
  b8  leaf:   1;
} BVH_Index;

STATIC_ASSERT(size_of(BVH_Index) == size_of(i32));

typedef struct {
  Vec3x8    mins;
  Vec3x8    maxs;
  BVH_Index children[8];
} BVH_Node;

typedef struct {
  Vector(BVH_Node) nodes;
  Triangles        triangles;
  BVH_Index        root;
} BVH;

typedef struct __attribute__((aligned(32))) {
  i32       version, n_nodes, n_triangles;
  BVH_Index root;
  u8        _padding[16];
} BVH_File_Header;

internal void bvh_save_writer(Writer *w, BVH *bvh) {
  BVH_File_Header header = {
    .version     = 0,
    .n_nodes     = bvh->nodes.len,
    .n_triangles = bvh->triangles.cap,
    .root        = bvh->root,
  };
  write_any(w, &header);
  Byte_Slice node_data = slice_to_bytes(bvh->nodes);
  Byte_Slice tris_data = {
    .data = (byte *)bvh->triangles.a_x,
    .len  = TRIANGLES_ALLOCATION_SIZE(bvh->triangles.cap),
  };
  write_bytes(w, node_data);
  write_bytes(w, tris_data);
}

internal b8 bvh_load_bytes(Byte_Slice data, BVH *bvh) {
  assert((uintptr)data.data % 32 == 0);

  BVH_File_Header header;
  if (data.len < size_of(header)) {
    return false;
  }
  mem_copy(&header, data.data, size_of(header));

  if (data.len != size_of(header) + header.n_nodes * size_of(BVH_Node) + TRIANGLES_ALLOCATION_SIZE(header.n_triangles)) {
    return false;
  }

  bvh->root = header.root;

  bvh->nodes.len  = header.n_nodes;
  bvh->nodes.cap  = header.n_nodes;
  bvh->nodes.data = (BVH_Node *)(data.data + size_of(header));

  assert((uintptr)bvh->nodes.data % 32 == 0);

  bvh->nodes.allocator = panic_allocator();

  f32 *tris_data = (f32 *)(data.data + size_of(header) + header.n_nodes * size_of(BVH_Node));

  assert((uintptr)tris_data % 32 == 0);

  bvh->triangles.a_x = tris_data + header.n_triangles * 0;
  bvh->triangles.a_y = tris_data + header.n_triangles * 1;
  bvh->triangles.a_z = tris_data + header.n_triangles * 2;

  bvh->triangles.b_x = tris_data + header.n_triangles * 3;
  bvh->triangles.b_y = tris_data + header.n_triangles * 4;
  bvh->triangles.b_z = tris_data + header.n_triangles * 5;

  bvh->triangles.c_x = tris_data + header.n_triangles * 6;
  bvh->triangles.c_y = tris_data + header.n_triangles * 7;
  bvh->triangles.c_z = tris_data + header.n_triangles * 8;

  bvh->triangles.aos = (Triangle_AOS *)(tris_data + header.n_triangles * 9);

  bvh->triangles.len = header.n_triangles;
  bvh->triangles.cap = header.n_triangles;

  bvh->triangles.allocator = panic_allocator();

  return true;
}
