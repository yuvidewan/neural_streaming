/* Full-search SAD block matching for nvc.video.motion - the native backend of
 * `estimate_block_motion`. See src/nvc/video/_native/__init__.py for how it is built
 * and loaded, and C_REWRITE_REPORT.md for the measurements behind it.
 *
 * THE CONTRACT: for the same inputs this returns bit-for-bit the same motion field as
 * the PyTorch reference (`estimate_block_motion_torch`). Motion is only ever ESTIMATED
 * by the encoder and TRANSMITTED, so decodability never depends on this file - but
 * byte-identical streams do, and that is what the tests pin. Three properties make it
 * hold, and each one is load-bearing:
 *
 *   1. SUMMATION ORDER. A candidate's cost is a float32 sum over the BxB window of
 *      (|c0-s0| + |c1-s1| + |c2-s2|). The reference adds the three channels as
 *      ((a0+a1)+a2) and then sums the window in row-major order (avg_pool2d), then
 *      applies g(s) = (s / (B*B)) * (B*B) in float32 (avg_pool2d's division, and the
 *      multiplication that follows it in Python). g is the identity for a power-of-two
 *      B*B but NOT otherwise, and it is not injective, so it is applied here too and
 *      candidates are compared on g(sum), never on the raw sum. g is monotone
 *      non-decreasing (IEEE division and multiplication round monotonically), which is
 *      what keeps the early exit below valid.
 *      Float addition is not associative, so BOTH code paths below add in exactly that
 *      order. In the AVX2 path each SIMD lane is one candidate and accumulates its own
 *      window in that order; nothing is ever summed "across" lanes.
 *   2. TIE-BREAK. Candidates are visited dy-major, dx-minor from -R to +R. A candidate
 *      replaces the best when cost < best, or cost == best and |dy|+|dx| is smaller.
 *   3. EARLY EXIT. Once g(partial sum) is strictly greater than the best cost so far
 *      the candidate cannot win or tie (partial sums of non-negative floats never
 *      decrease and g is monotone), so the rest of its window is skipped. Ties are
 *      never skipped.
 *
 * Build flags that matter: -ffp-contract=off (never let the compiler fuse operations),
 * and NEVER -ffast-math / -fassociative-math (either would reorder the sums and silently
 * break property 1). There is no multiply in the cost, so FMA cannot change it anyway;
 * the flag is belt and braces for anyone who later adds one.
 *
 * The AVX2 path is compiled with a per-function target attribute and selected at RUN
 * time, so a library built with generic flags never executes an instruction the CPU
 * does not have. Non-x86 builds get the scalar path only.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#define NVC_X86 1
#include <immintrin.h>
#else
#define NVC_X86 0
#endif

#ifdef _WIN32
#define NVC_EXPORT __declspec(dllexport)
#else
#define NVC_EXPORT __attribute__((visibility("default")))
#endif

#define NVC_ABI_VERSION 1
#define NVC_LANES 8            /* candidates evaluated together by the AVX2 path */
#define NVC_PAD_FILL 1.0e6f    /* fills columns no valid candidate reads; see nvc_bs_pad */

#define NVC_OK 0
#define NVC_ERR_ARGS (-1)
#define NVC_ERR_NO_AVX2 (-2)

NVC_EXPORT int32_t nvc_bs_abi_version(void) { return NVC_ABI_VERSION; }

static int cpu_has_avx2(void)
{
#if NVC_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

NVC_EXPORT int32_t nvc_bs_has_avx2(void) { return cpu_has_avx2(); }

/* ---- padding ------------------------------------------------------------------- */

/* Shape of the padded reference the search reads: [3][padded_height][padded_width].
 * The AVX2 path loads 8 candidates at a time, so the last group can read up to 7
 * columns past the replicate-padded edge; they are extra columns filled with a large
 * value so lanes that fall beyond +R can never win or stall the early exit. */
NVC_EXPORT int32_t nvc_bs_padded_height(int32_t height, int32_t range) { return height + 2 * range; }
NVC_EXPORT int32_t nvc_bs_padded_width(int32_t width, int32_t range) { return width + 2 * range + NVC_LANES; }

/* Replicate-pad `ref` ([3][H][W]) by R on every side into `pad`
 * ([3][H+2R][W+2R+NVC_LANES]) - the same padding F.pad(mode="replicate") gives. */
NVC_EXPORT int32_t nvc_bs_pad(const float *ref, int32_t H, int32_t W, int32_t R, float *pad)
{
    if (ref == NULL || pad == NULL || H <= 0 || W <= 0 || R < 0) return NVC_ERR_ARGS;
    const int32_t PH = nvc_bs_padded_height(H, R), WP = nvc_bs_padded_width(W, R);
    const size_t rplane = (size_t)H * W, pplane = (size_t)PH * WP;
    for (int c = 0; c < 3; c++) {
        const float *src = ref + c * rplane;
        float *dst = pad + c * pplane;
        for (int32_t py = 0; py < PH; py++) {
            int32_t sy = py - R;
            sy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy);
            const float *row = src + (size_t)sy * W;
            float *out = dst + (size_t)py * WP;
            for (int32_t px = 0; px < W + 2 * R; px++) {
                int32_t sx = px - R;
                sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
                out[px] = row[sx];
            }
            for (int32_t px = W + 2 * R; px < WP; px++) out[px] = NVC_PAD_FILL;
        }
    }
    return NVC_OK;
}

/* ---- search: scalar reference path ---------------------------------------------- */

static void search_blocks_scalar(const float *cur, const float *pad, int H, int W, int R, int B,
                                 int blk_begin, int blk_end, int32_t *out_dy, int32_t *out_dx,
                                 int early_exit)
{
    const int WP = nvc_bs_padded_width(W, R), bx_n = W / B;
    const size_t cplane = (size_t)H * W, pplane = (size_t)nvc_bs_padded_height(H, R) * WP;
    const float bb = (float)(B * B);

    for (int blk = blk_begin; blk < blk_end; blk++) {
        const int y0 = (blk / bx_n) * B, x0 = (blk % bx_n) * B;
        float best = INFINITY;
        int best_key = 0x7fffffff, best_dy = 0, best_dx = 0;

        for (int dy = -R; dy <= R; dy++) {
            for (int dx = -R; dx <= R; dx++) {
                const int top = R + dy + y0, left = R + dx + x0;
                float acc = 0.0f;
                int y;
                for (y = 0; y < B; y++) {
                    const float *c0 = cur + (size_t)(y0 + y) * W + x0;
                    const float *c1 = c0 + cplane, *c2 = c1 + cplane;
                    const float *p0 = pad + (size_t)(top + y) * WP + left;
                    const float *p1 = p0 + pplane, *p2 = p1 + pplane;
                    for (int x = 0; x < B; x++) {
                        const float a0 = fabsf(c0[x] - p0[x]);
                        const float a1 = fabsf(c1[x] - p1[x]);
                        const float a2 = fabsf(c2[x] - p2[x]);
                        acc += (a0 + a1) + a2;
                    }
                    if (early_exit && best != INFINITY && (acc / bb) * bb > best) break;
                }
                if (y < B) continue;                   /* early-exited: cannot win or tie */
                const float cost = (acc / bb) * bb;
                const int key = abs(dy) + abs(dx);
                if (cost < best || (cost == best && key < best_key)) {
                    best = cost; best_key = key; best_dy = dy; best_dx = dx;
                }
            }
        }
        out_dy[blk] = best_dy;
        out_dx[blk] = best_dx;
    }
}

/* ---- search: AVX2 path (8 candidates per SIMD register) -------------------------- */

#if NVC_X86
__attribute__((target("avx2")))
static void search_blocks_avx2(const float *cur, const float *pad, int H, int W, int R, int B,
                               int blk_begin, int blk_end, int32_t *out_dy, int32_t *out_dx,
                               int early_exit)
{
    const int WP = nvc_bs_padded_width(W, R), bx_n = W / B;
    const size_t cplane = (size_t)H * W, pplane = (size_t)nvc_bs_padded_height(H, R) * WP;
    const int groups = (2 * R + 1 + NVC_LANES - 1) / NVC_LANES;
    const __m256 signmask = _mm256_set1_ps(-0.0f);
    const __m256 bbv = _mm256_set1_ps((float)(B * B));

    for (int blk = blk_begin; blk < blk_end; blk++) {
        const int y0 = (blk / bx_n) * B, x0 = (blk % bx_n) * B;
        float best = INFINITY;
        int best_key = 0x7fffffff, best_dy = 0, best_dx = 0;

        for (int dy = -R; dy <= R; dy++) {
            for (int g = 0; g < groups; g++) {
                const int dx0 = -R + NVC_LANES * g;
                const int top = R + dy + y0, left = R + dx0 + x0;
                __m256 acc = _mm256_setzero_ps();
                int y;
                for (y = 0; y < B; y++) {
                    const float *c0 = cur + (size_t)(y0 + y) * W + x0;
                    const float *c1 = c0 + cplane, *c2 = c1 + cplane;
                    const float *p0 = pad + (size_t)(top + y) * WP + left;
                    const float *p1 = p0 + pplane, *p2 = p1 + pplane;
                    for (int x = 0; x < B; x++) {
                        __m256 d0 = _mm256_sub_ps(_mm256_set1_ps(c0[x]), _mm256_loadu_ps(p0 + x));
                        __m256 d1 = _mm256_sub_ps(_mm256_set1_ps(c1[x]), _mm256_loadu_ps(p1 + x));
                        __m256 d2 = _mm256_sub_ps(_mm256_set1_ps(c2[x]), _mm256_loadu_ps(p2 + x));
                        d0 = _mm256_andnot_ps(signmask, d0);
                        d1 = _mm256_andnot_ps(signmask, d1);
                        d2 = _mm256_andnot_ps(signmask, d2);
                        acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_add_ps(d0, d1), d2));
                    }
                    if (early_exit && best != INFINITY) {
                        const __m256 scaled = _mm256_mul_ps(_mm256_div_ps(acc, bbv), bbv);
                        if (_mm256_movemask_ps(_mm256_cmp_ps(scaled, _mm256_set1_ps(best),
                                                             _CMP_LE_OQ)) == 0)
                            break;                     /* every lane strictly worse than best */
                    }
                }
                if (y < B) continue;

                float lane[NVC_LANES];
                _mm256_storeu_ps(lane, _mm256_mul_ps(_mm256_div_ps(acc, bbv), bbv));
                /* storeu: 32-byte stack alignment is not honoured by every Win64 toolchain */
                for (int l = 0; l < NVC_LANES; l++) {
                    const int dx = dx0 + l;
                    if (dx > R) break;                 /* unused lanes of the last group */
                    const int key = abs(dy) + abs(dx);
                    if (lane[l] < best || (lane[l] == best && key < best_key)) {
                        best = lane[l]; best_key = key; best_dy = dy; best_dx = dx;
                    }
                }
            }
        }
        out_dy[blk] = best_dy;
        out_dx[blk] = best_dx;
    }
}
#endif

/* Search blocks [blk_begin, blk_end) of the block grid, writing only those entries of
 * out_dy/out_dx (each [(H/B)*(W/B)] int32), so disjoint ranges can run on separate
 * threads against the same `pad`.
 *   cur   [3][H][W] float32          pad  the output of nvc_bs_pad for the reference
 *   mode  0 = best available, 1 = scalar, 2 = AVX2 (NVC_ERR_NO_AVX2 if unsupported)
 * Returns NVC_OK, NVC_ERR_ARGS or NVC_ERR_NO_AVX2. */
NVC_EXPORT int32_t nvc_bs_search(const float *cur, const float *pad, int32_t H, int32_t W,
                                 int32_t R, int32_t B, int32_t blk_begin, int32_t blk_end,
                                 int32_t *out_dy, int32_t *out_dx, int32_t early_exit,
                                 int32_t mode)
{
    if (cur == NULL || pad == NULL || out_dy == NULL || out_dx == NULL) return NVC_ERR_ARGS;
    if (H <= 0 || W <= 0 || R < 0 || B <= 0 || H % B != 0 || W % B != 0) return NVC_ERR_ARGS;
    const int32_t blocks = (H / B) * (W / B);
    if (blk_begin < 0 || blk_end > blocks || blk_begin > blk_end) return NVC_ERR_ARGS;
    if (mode < 0 || mode > 2) return NVC_ERR_ARGS;

    int use_avx2 = 0;
    if (mode == 2) {
        if (!cpu_has_avx2()) return NVC_ERR_NO_AVX2;
        use_avx2 = 1;
    } else if (mode == 0) {
        use_avx2 = cpu_has_avx2();
    }
#if NVC_X86
    if (use_avx2) {
        search_blocks_avx2(cur, pad, H, W, R, B, blk_begin, blk_end, out_dy, out_dx, early_exit);
        return NVC_OK;
    }
#else
    if (use_avx2) return NVC_ERR_NO_AVX2;
#endif
    search_blocks_scalar(cur, pad, H, W, R, B, blk_begin, blk_end, out_dy, out_dx, early_exit);
    return NVC_OK;
}
