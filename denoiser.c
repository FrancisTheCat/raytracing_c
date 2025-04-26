#include "denoiser.h"

#include "codin/linalg.h"
#include "codin/sort.h"
#include "codin/thread.h"
#include "codin/os.h"
#include <stdatomic.h>

#define DENOISING_THRESHOLD  0.0125f
#define NEIGHBOURHOOD_WEIGHT 5

internal inline f32 luminance(Color3 x) {
	return vec3_dot(x, vec3(0.2126f, 0.7152f, 0.0722f));
}

internal inline Color3 sample_image(Image const *image, isize x, isize y) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= image->width)  x = image->width  - 1;
  if (y >= image->height) y = image->height - 1;

  Color3 ret = {0};
  for_range(c, 0, min(image->components, 3)) {
    ret.data[c] = IDX(image->pixels, (x + y * image->stride) * image->components + c) / 255.999f;
  }
  return ret;
}

internal inline void store_pixel(Image const *image, isize x, isize y, Color3 color) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= image->width)  x = image->width  - 1;
  if (y >= image->height) y = image->height - 1;

  for_range(c, 0, min(image->components, 3)) {
    IDX(image->pixels, (x + y * image->stride) * image->components + c) = color.data[c] * 255.999f;
  }
}

typedef struct {
  Image         const *src, *dst;
  _Atomic isize n_threads, current_chunk;
} Denoising_Context;

#define CHUNK_SIZE 32

internal void denoiser_thread_proc(Denoising_Context *ctx) {
  Color4 colors[3 * 3];
  Slice(Color4) color_slice = slice_array(type_of(color_slice), colors);

  isize width, height, chunks_x, chunks_y, n_chunks;
  width    = ctx->src->width;
  height   = ctx->src->height;
  chunks_x = (width  + CHUNK_SIZE - 1) / CHUNK_SIZE;
  chunks_y = (height + CHUNK_SIZE - 1) / CHUNK_SIZE;
  n_chunks = chunks_x * chunks_y;

  loop {
    isize c = atomic_fetch_add(&ctx->current_chunk, 1);
    if (c >= n_chunks) {
      break;
    }

    isize start_x = (c % chunks_x) * CHUNK_SIZE;
    isize start_y = (c / chunks_x) * CHUNK_SIZE;

    for_range(y, start_y, start_y + CHUNK_SIZE) {
      if (y >= height) {
        break;
      }

      for_range(x, start_x, start_x + CHUNK_SIZE) {
        if (x >= width) {
          break;
        }

        isize n_colors = 0;

        Color4 original;
        for_range(yo, -1, 2) {
          for_range(xo, -1, 2) {
            Color4 color;
            color.xyz = sample_image(ctx->src, x + xo, y + yo);
            color.a   = luminance(color.xyz);
            if (xo == 0 && yo == 0) {
              original = color;
            }

            bool found = false;
            for_range(i, 0, n_colors) {
              if (colors[i].a > color.a) {
                found = true;

                for (isize j = n_colors; j > i; j -= 1) {
                  colors[j] = colors[j - 1];
                }

                colors[i] = color;
                break;
              }
            }
            if (!found) {
              colors[n_colors] = color;
            }
            n_colors += 1;
          }
        }

        Color4 median = colors[count_of(colors) / 2];
        f32    mean   = 0;
        slice_iter_v(color_slice, c, i, {
          if (i == 0 || i == color_slice.len - 1) { continue; }
          mean += c.a;
        });
        mean /= color_slice.len - 2;

        f32 neighbourhood_noisiness = abs_f32(median.a - mean);

        f32 luminance_diff = abs_f32(median.a - original.a) - neighbourhood_noisiness * NEIGHBOURHOOD_WEIGHT;
            luminance_diff = clamp(luminance_diff, 0, DENOISING_THRESHOLD) / DENOISING_THRESHOLD;
        store_pixel(ctx->dst, x, y, vec3_lerp(original.xyz, median.xyz, luminance_diff));
      }
    }
  }

  ctx->n_threads -= 1;
}

extern void denoise_image(Image const *src, Image const *dst, isize n_threads) {
  assert(src->pixels.data != dst->pixels.data);
  assert(src->width  == dst->width);
  assert(src->height == dst->height);

  Denoising_Context ctx = {
    .src       = src,
    .dst       = dst,
    .n_threads = n_threads,
  };

  for_range(thread_i, 0, n_threads - 1) {
    thread_create((Thread_Proc)denoiser_thread_proc, &ctx, THREAD_STACK_DEFAULT, THREAD_TLS_DEFAULT);
  }

  denoiser_thread_proc(&ctx);

  while (ctx.n_threads > 0) {
    processor_yield();
  }
}
