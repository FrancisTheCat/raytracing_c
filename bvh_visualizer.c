#include "codin/codin.h"
#include "codin/os.h"

#define Image    rl_Image
#define Material rl_Material
#define Ray      rl_Ray
#define Shader   rl_Shader
#define Camera   rl_Camera

#include "raylib.h"

#undef Image
#undef Ray
#undef Material
#undef Shader
#undef Camera

#define SIMD_WIDTH 8

#include "scene.c"

internal void draw_bvh_node(BVH *bvh, BVH_Node *node, isize depth, isize show) {
  if (depth < 0) {
    return;
  }
  Color color = ColorAlpha(ColorFromHSV(-360 * (f32)depth / bvh->depth, 0.7, 1), 0.125f);
  if (depth == show) {
    for_range(i, 0, 8) {
      Vector3 position = {
        .x = (node->maxs.x[i] + node->mins.x[i]) / 2.0f,
        .y = (node->maxs.y[i] + node->mins.y[i]) / 2.0f,
        .z = (node->maxs.z[i] + node->mins.z[i]) / 2.0f,
      };

      f32 x = node->maxs.x[i] - node->mins.x[i];
      f32 y = node->maxs.y[i] - node->mins.y[i];
      f32 z = node->maxs.z[i] - node->mins.z[i];

      DrawCubeWires(position, x, y, z, color);
    }
  }

  for_range(i, 0, 8) {
    if (
      node->mins.x[i] >= node->maxs.x[i] || 
      node->mins.y[i] >= node->maxs.y[i] || 
      node->mins.z[i] >= node->maxs.z[i]
    ) {
      continue;
    }
    BVH_Node *child = node + 1 + i * bvh_n_internal_nodes(depth - 1);
    draw_bvh_node(bvh, child, depth - 1, show);
  }
}

internal void draw_bvh(BVH bvh, isize depth) {
  draw_bvh_node(&bvh, &IDX(bvh.nodes, 0), bvh.depth, depth);
}

i32 main() {
  Scene scene;

  Fd bvh_file = unwrap_err(file_open(LIT("test.scene"), FP_Read));
  File_Info fi;
  OS_Error err = file_stat(bvh_file, &fi);
  assert(err == OSE_None);
  Byte_Slice data = slice_make_aligned(Byte_Slice, fi.size, SIMD_ALIGN, context.allocator);
  isize n_read = unwrap_err(file_read(bvh_file, data));
  assert(n_read == fi.size);
  b8 ok = scene_load_bytes(data, &scene);
  assert(ok);

  InitWindow(900, 600, "BVH");

  rl_Camera camera     = { 0 };
  camera.position   = (Vector3){ 10.0f, 10.0f, 10.0f };
  camera.target     = (Vector3){ 0.0f,  0.0f,  0.0f  };
  camera.up         = (Vector3){ 0.0f,  1.0f,  0.0f  };
  camera.fovy       = 45.0f;
  camera.projection = CAMERA_PERSPECTIVE;

  DisableCursor();

  SetTargetFPS(60);

  isize depth = scene.bvh.depth;
  
  while (!WindowShouldClose()) {
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    if (IsKeyPressed(KEY_UP)) {
      depth += 1;
    } else if (IsKeyPressed(KEY_DOWN)) {
      depth -= 1;
    }

    BeginDrawing();
      ClearBackground(BLACK);

      BeginMode3D(camera);
        draw_bvh(scene.bvh, depth);
      EndMode3D();
    EndDrawing();
  }

  CloseWindow();
}
