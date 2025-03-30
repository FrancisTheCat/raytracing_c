#include "codin/codin.h"
#include "codin/os.h"

#define Image    rl_Image
#define Material rl_Material
#define Ray      rl_Ray
#define Shader   rl_Shader

#include "raylib.h"

#undef Image
#undef Ray
#undef Material
#undef Shader

#include "bvh.c"

internal Color colors[] = {
  RAYWHITE,
  YELLOW,
  GREEN,
  BLUE,
  PURPLE,
  RED,
  ORANGE,
};

internal void draw_bvh_node(BVH *bvh, BVH_Node *node, i32 depth) {
  if (depth < 0) {
    return;
  }
  Color color = ColorAlpha(colors[depth % count_of(colors)], 0.25f);
  for_range(i, 0, 8) {
    Vector3 position;
    float width, height, depth;

    position.x = (node->maxs.x[i] + node->mins.x[i]) / 2.0f;
    position.y = (node->maxs.y[i] + node->mins.y[i]) / 2.0f;
    position.z = (node->maxs.z[i] + node->mins.z[i]) / 2.0f;

    width  = node->maxs.x[i] - node->mins.x[i];
    height = node->maxs.y[i] - node->mins.y[i];
    depth  = node->maxs.z[i] - node->mins.z[i];

    DrawCubeWires(position, width, height, depth, color);
  }

  for_range(i, 0, 8) {
    if (!node->children[i].leaf) {
      draw_bvh_node(bvh, &IDX(bvh->nodes, node->children[i].index), depth - 1);
    }
  }
}

internal void draw_bvh(BVH bvh) {
  if (bvh.root.leaf) {
    return;
  }
  BVH_Node node = IDX(bvh.nodes, bvh.root.index);
  draw_bvh_node(&bvh, &node, 10);
}

i32 main() {
  BVH bvh = {0};

  Fd bvh_file = unwrap_err(file_open(LIT("test.bvh"), FP_Read));
  File_Info fi;
  OS_Error err = file_stat(bvh_file, &fi);
  assert(err == OSE_None);
  Byte_Slice data = slice_make_aligned(Byte_Slice, fi.size, 32, context.allocator);
  isize n_read = unwrap_err(file_read(bvh_file, data));
  assert(n_read == fi.size);
  b8 ok = bvh_load_bytes(data, &bvh);
  assert(ok);

  InitWindow(900, 600, "BVH");

  Camera camera     = { 0 };
  camera.position   = (Vector3){ 10.0f, 10.0f, 10.0f };
  camera.target     = (Vector3){ 0.0f, 0.0f, 0.0f };
  camera.up         = (Vector3){ 0.0f, 1.0f, 0.0f };
  camera.fovy       = 45.0f;
  camera.projection = CAMERA_PERSPECTIVE;

  DisableCursor();

  SetTargetFPS(60);
  
  while (!WindowShouldClose()) {
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    BeginDrawing();
      ClearBackground(BLACK);

      BeginMode3D(camera);
        draw_bvh(bvh);
      EndMode3D();
    EndDrawing();
  }
}
