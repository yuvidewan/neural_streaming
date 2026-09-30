/* PROTOTYPE kernels for C_REWRITE_REPORT.md - reference code, not part of the package.
 *
 * Two hot spots measured in nvc.video, rewritten as AVX2 C and verified against the
 * current PyTorch implementations by verify_kernels.py:
 *
 *   nvc_block_search  <->  nvc.video.motion.estimate_block_motion   (85% of encode time)
 *   nvc_assign_argmin <->  nvc.video.entropy.SharedCodebook.assign_tensor  (52% of decode)
 *
 * Build (the harness does this for you):
 *   gcc -O3 -mavx2 -mfma -ffp-contract=off -shared -o kernels.dll kernels.c      (Windows)
 *   gcc -O3 -fPIC -mavx2 -mfma -ffp-contract=off -shared -o kernels.so kernels.c (Linux)
 *
 * -ffp-contract=off matters: it stops the compiler from silently fusing a multiply and
 * an add, which would change float results. Never build these with -ffast-math.
 *
 * NOT production-ready as written: AVX2 only (no scalar fallback, no runtime CPU check),
 * single-threaded, no input validation. C_REWRITE_REPORT.md section 6 lists what the
 * production version needs.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* ---------------------------------------------------------------------------------
 * Full-search SAD block matching, bit-identical to estimate_block_motion.
 *
 * What "identical" requires, and how it is met:
 *   - cost = sum over the BxB window of (|c0-s0| + |c1-s1| + |c2-s2|), in float32.
 *     torch adds the 3 channels as ((a0+a1)+a2), then avg_pool2d sums the window
 *     sequentially in row-major order (then /256 and *256, both exact). Every AVX2
 *     LANE below is one candidate and accumulates its window in exactly that order,
 *     so each candidate's float32 result is bit-equal to torch's.
 *   - candidates are visited dy-major, dx-minor from -R to +R, and a candidate replaces
 *     the best when cost < best, or cost == best and |dy|+|dx| is smaller. The lanes are
 *     applied to that rule in ascending dx, so the sequential semantics are preserved.
 *   - early exit: once EVERY lane's partial sum is strictly greater than the best cost
 *     so far, no lane can win or tie (partial sums of non-negative floats never
 *     decrease), so the candidate group is skipped. Ties are never skipped.
 *
 * cur:  [3][H][W] float32
 * pad:  [3][H+2R][WP] float32, the reference replicate-padded by R on every side, with
 *       WP >= W+2R+8 and the extra columns filled with a huge value (1e6) so lanes past
 *       dx=+R (the last group has 7 unused lanes) can never win or block the early exit.
 * out_dy/out_dx: [(H/B)*(W/B)] int32
 * --------------------------------------------------------------------------------- */
EXPORT void nvc_block_search(const float *cur, const float *pad, int H, int W, int WP, int R,
                             int B, int32_t *out_dy, int32_t *out_dx, int early_exit)
{
    const int bx_n = W / B, blocks = (H / B) * bx_n;
    const int PH = H + 2 * R;
    const size_t cplane = (size_t)H * W, pplane = (size_t)PH * WP;
    const int groups = (2 * R + 1 + 7) / 8;
    const __m256 signmask = _mm256_set1_ps(-0.0f);

    for (int blk = 0; blk < blocks; blk++) {
        const int y0 = (blk / bx_n) * B, x0 = (blk % bx_n) * B;
        float best = INFINITY;
        int best_key = 0x7fffffff, best_dy = 0, best_dx = 0;

        for (int dy = -R; dy <= R; dy++) {
            for (int g = 0; g < groups; g++) {
                const int dx0 = -R + 8 * g;
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
                    if (early_exit && best != INFINITY &&
                        _mm256_movemask_ps(_mm256_cmp_ps(acc, _mm256_set1_ps(best), _CMP_LE_OQ)) == 0)
                        break;
                }
                if (y < B) continue;             /* early-exited: no lane can win or tie */

                float lane[8] __attribute__((aligned(32)));
                _mm256_store_ps(lane, acc);
                for (int l = 0; l < 8; l++) {
                    const int dx = dx0 + l;
                    if (dx > R) break;
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

/* ---------------------------------------------------------------------------------
 * Fused prototype assignment: argmin_k  sum_a probs[n][a] * log2cost[k][a], with ties
 * (exactly equal float costs) resolved to the LOWEST k, like _torch_argmin_lowest_index.
 * Never materialises the [N, K] cost matrix (33 MB at N=16384, K=512).
 *
 * probs: [N][A] float32.   lt: [A][K] float32 (the log2 costs TRANSPOSED).   K % 32 == 0.
 * Each SIMD lane is one prototype and accumulates its A products with a sequential
 * fused multiply-add chain; four independent chains are interleaved to hide FMA latency.
 * The chains are separate lanes, so per-prototype arithmetic order is unchanged.
 * --------------------------------------------------------------------------------- */
static inline void merge_min(__m256 acc, __m256i idx, __m256 *bestv, __m256i *besti)
{
    const __m256 lt = _mm256_cmp_ps(acc, *bestv, _CMP_LT_OQ);      /* strict: keeps lower k */
    *bestv = _mm256_blendv_ps(*bestv, acc, lt);
    *besti = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(*besti),
                                                  _mm256_castsi256_ps(idx), lt));
}

EXPORT void nvc_assign_argmin(const float *probs, const float *lt, int N, int A, int K,
                              int64_t *out)
{
    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (int n = 0; n < N; n++) {
        const float *p = probs + (size_t)n * A;
        __m256 bestv = _mm256_set1_ps(INFINITY);
        __m256i besti = _mm256_setzero_si256();
        for (int k0 = 0; k0 < K; k0 += 32) {
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (int a = 0; a < A; a++) {
                const __m256 pv = _mm256_set1_ps(p[a]);
                const float *l = lt + (size_t)a * K + k0;
                a0 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(l), a0);
                a1 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(l + 8), a1);
                a2 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(l + 16), a2);
                a3 = _mm256_fmadd_ps(pv, _mm256_loadu_ps(l + 24), a3);
            }
            const __m256i base = _mm256_add_epi32(_mm256_set1_epi32(k0), lane_ids);
            merge_min(a0, base, &bestv, &besti);
            merge_min(a1, _mm256_add_epi32(base, _mm256_set1_epi32(8)), &bestv, &besti);
            merge_min(a2, _mm256_add_epi32(base, _mm256_set1_epi32(16)), &bestv, &besti);
            merge_min(a3, _mm256_add_epi32(base, _mm256_set1_epi32(24)), &bestv, &besti);
        }
        float v[8] __attribute__((aligned(32)));
        int32_t ix[8] __attribute__((aligned(32)));
        _mm256_store_ps(v, bestv);
        _mm256_store_si256((__m256i *)ix, besti);
        float b = v[0];
        int bi = ix[0];
        for (int l = 1; l < 8; l++)                                  /* cross-lane: lowest k on ties */
            if (v[l] < b || (v[l] == b && ix[l] < bi)) { b = v[l]; bi = ix[l]; }
        out[n] = bi;
    }
}

/* Raw cost matrix [N][K], for the bit-exactness experiment only (K % 8 == 0).
 * use_fma=1: fused multiply-add chain.  use_fma=0: separate multiply then add. */
EXPORT void nvc_cost_rows(const float *probs, const float *lt, int N, int A, int K, float *out,
                          int use_fma)
{
    for (int n = 0; n < N; n++) {
        const float *p = probs + (size_t)n * A;
        for (int k0 = 0; k0 < K; k0 += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int a = 0; a < A; a++) {
                const __m256 pv = _mm256_set1_ps(p[a]);
                const __m256 lv = _mm256_loadu_ps(lt + (size_t)a * K + k0);
                acc = use_fma ? _mm256_fmadd_ps(pv, lv, acc)
                              : _mm256_add_ps(acc, _mm256_mul_ps(pv, lv));
            }
            _mm256_storeu_ps(out + (size_t)n * K + k0, acc);
        }
    }
}
