/* C port of nvc/compression/range_coder.py's encode_symbols/decode_symbols.
 *
 * Kept structurally identical to the Python reference on purpose - same
 * variable names, same control flow, same order of operations - so the two
 * can be read side by side and checked for equivalence line by line. See
 * range_coder.py's module docstring for the algorithm explanation; this
 * file intentionally does not repeat it.
 *
 * All interval/frequency arithmetic uses int64_t. This is safe (not an
 * approximation): span is bounded by WHOLE (2**32), frequency totals are
 * bounded by MAX_TOTAL_FREQUENCY (2**30), so every product computed here is
 * bounded by ~2**62, well inside int64_t's range (max ~2**63 - 1). All
 * operands in every division are non-negative by construction, so C's
 * truncating `/` and Python's flooring `//` agree exactly - there is no
 * signed-division edge case to account for.
 *
 * No Python.h / CPython C-API dependency on purpose: this compiles to a
 * plain shared library, loaded from Python via ctypes (see
 * range_coder.py's _native module). That avoids needing to match a
 * specific CPython ABI/version - the same .dll/.so works across Python
 * versions, and the build step is a single `gcc -shared` call, not a full
 * setuptools extension build.
 */
#include <stdint.h>
#include <stdlib.h>

#define PRECISION 32
#define WHOLE (((int64_t)1) << PRECISION)
#define HALF (WHOLE >> 1)
#define QUARTER (WHOLE >> 2)
#define THREE_QUARTERS (3 * QUARTER)

#if defined(_WIN32)
#define NVC_EXPORT __declspec(dllexport)
#else
#define NVC_EXPORT __attribute__((visibility("default")))
#endif

/* ---- Bit writer: dynamic buffer, MSB-first, doubling growth ---- */
typedef struct {
    uint8_t *data;
    int64_t len;       /* bytes fully written */
    int64_t cap;
    uint32_t current;  /* bits accumulating for the in-progress byte */
    int32_t filled;    /* how many bits are in `current` */
} BitWriter;

/* All allocation can fail (OOM). `ok` tracks that across every bw_* call so
 * rc_encode can check it once at the end and report failure to Python
 * cleanly, instead of dereferencing a NULL `data` pointer. */
static int bw_init(BitWriter *w) {
    w->cap = 4096;
    w->data = (uint8_t *)malloc((size_t)w->cap);
    w->len = 0;
    w->current = 0;
    w->filled = 0;
    return w->data != NULL;
}

static int bw_ensure(BitWriter *w, int64_t extra) {
    if (w->len + extra > w->cap) {
        int64_t new_cap = w->cap;
        while (w->len + extra > new_cap) {
            new_cap *= 2;
        }
        uint8_t *grown = (uint8_t *)realloc(w->data, (size_t)new_cap);
        if (grown == NULL) {
            return 0; /* w->data is still valid (unchanged) but too small - caller must abort */
        }
        w->data = grown;
        w->cap = new_cap;
    }
    return 1;
}

static void bw_write(BitWriter *w, int bit, int *ok) {
    if (!*ok) return;
    w->current = (w->current << 1) | (uint32_t)(bit & 1);
    w->filled += 1;
    if (w->filled == 8) {
        if (!bw_ensure(w, 1)) { *ok = 0; return; }
        w->data[w->len++] = (uint8_t)w->current;
        w->current = 0;
        w->filled = 0;
    }
}

static void bw_finish(BitWriter *w, int *ok) {
    if (!*ok) return;
    if (w->filled) {
        if (!bw_ensure(w, 1)) { *ok = 0; return; }
        w->data[w->len++] = (uint8_t)(w->current << (8 - w->filled));
        w->current = 0;
        w->filled = 0;
    }
}

/* ---- Bit reader: MSB-first, reads 0 forever past the end (deliberate -
 * see range_coder.py's _BitReader docstring for why that's correct) ---- */
typedef struct {
    const uint8_t *data;
    int64_t length;
    int64_t position;
    int32_t bit;
} BitReader;

static void br_init(BitReader *r, const uint8_t *data, int64_t length) {
    r->data = data;
    r->length = length;
    r->position = 0;
    r->bit = 0;
}

static int br_read(BitReader *r) {
    if (r->position >= r->length) {
        return 0;
    }
    int b = (r->data[r->position] >> (7 - r->bit)) & 1;
    r->bit += 1;
    if (r->bit == 8) {
        r->bit = 0;
        r->position += 1;
    }
    return b;
}

/* bisect_right over a non-decreasing int64 array (mirrors Python's
 * bisect.bisect_right exactly: first index i such that cum[i] > value). */
static int64_t bisect_right_i64(const int64_t *cum, int64_t table_width, int64_t value) {
    int64_t lo = 0, hi = table_width;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (value < cum[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

/* symbols, table_index: length n, int64.
 * cumulative: num_tables x table_width (= num_symbols + 1) row-major, int64.
 * All inputs are assumed already validated Python-side (matching
 * range_coder.py's _validate_inputs) - this function does no input
 * validation of its own.
 *
 * Returns 0 on success: *out_data points at a malloc'd buffer of *out_len
 * bytes - free it with rc_free once the caller has copied the bytes out
 * (e.g. into a Python `bytes` object). Returns -1 on allocation failure
 * (OOM); out_data and out_len are left untouched (NULL and 0) and there is
 * nothing to free in that case.
 */
NVC_EXPORT
int32_t rc_encode(
    const int64_t *symbols, int64_t n,
    const int64_t *cumulative, int64_t table_width,
    const int64_t *table_index,
    uint8_t **out_data, int64_t *out_len
) {
    BitWriter w;
    int ok = bw_init(&w);
    if (!ok) {
        *out_data = NULL;
        *out_len = 0;
        return -1;
    }

    int64_t low = 0;
    int64_t high = WHOLE - 1;
    int64_t pending = 0;

    for (int64_t i = 0; i < n && ok; i++) {
        int64_t symbol = symbols[i];
        int64_t table = table_index[i];
        const int64_t *cum = cumulative + table * table_width;
        int64_t total = cum[table_width - 1];
        int64_t span = high - low + 1;

        high = low + (span * cum[symbol + 1]) / total - 1;
        low = low + (span * cum[symbol]) / total;

        for (;;) {
            if (high < HALF) {
                bw_write(&w, 0, &ok);
                for (int64_t k = 0; k < pending; k++) bw_write(&w, 1, &ok);
                pending = 0;
            } else if (low >= HALF) {
                bw_write(&w, 1, &ok);
                for (int64_t k = 0; k < pending; k++) bw_write(&w, 0, &ok);
                pending = 0;
                low -= HALF;
                high -= HALF;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                pending += 1;
                low -= QUARTER;
                high -= QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
        }
    }

    if (ok) {
        pending += 1;
        if (low < QUARTER) {
            bw_write(&w, 0, &ok);
            for (int64_t k = 0; k < pending; k++) bw_write(&w, 1, &ok);
        } else {
            bw_write(&w, 1, &ok);
            for (int64_t k = 0; k < pending; k++) bw_write(&w, 0, &ok);
        }
        bw_finish(&w, &ok);
    }

    if (!ok) {
        free(w.data);
        *out_data = NULL;
        *out_len = 0;
        return -1;
    }

    *out_data = w.data;
    *out_len = w.len;
    return 0;
}

NVC_EXPORT
void rc_free(uint8_t *ptr) {
    free(ptr);
}

/* Inverse of rc_encode. out_symbols must already be allocated by the
 * caller, sized for n int64_t values. Returns 0 on success, -1 if any
 * required pointer is NULL (defensive - should never happen from the
 * Python wrapper, which always supplies real buffers; guards against a
 * ctypes-level misuse bug rather than a normal runtime condition) or if
 * n/table_width are non-positive. Never allocates, so there is no OOM
 * path here the way there is in rc_encode. */
NVC_EXPORT
int32_t rc_decode(
    const uint8_t *payload, int64_t payload_len,
    int64_t n,
    const int64_t *cumulative, int64_t table_width,
    const int64_t *table_index,
    int64_t *out_symbols
) {
    if (cumulative == NULL || table_index == NULL || out_symbols == NULL
        || n <= 0 || table_width <= 0 || (payload == NULL && payload_len != 0)) {
        return -1;
    }

    BitReader r;
    br_init(&r, payload, payload_len);

    int64_t low = 0;
    int64_t high = WHOLE - 1;
    int64_t value = 0;
    for (int i = 0; i < PRECISION; i++) {
        value = (value << 1) | br_read(&r);
    }

    for (int64_t pos = 0; pos < n; pos++) {
        int64_t table = table_index[pos];
        const int64_t *cum = cumulative + table * table_width;
        int64_t total = cum[table_width - 1];
        int64_t span = high - low + 1;

        int64_t scaled = ((value - low + 1) * total - 1) / span;
        int64_t symbol = bisect_right_i64(cum, table_width, scaled) - 1;
        out_symbols[pos] = symbol;

        high = low + (span * cum[symbol + 1]) / total - 1;
        low = low + (span * cum[symbol]) / total;

        for (;;) {
            if (high < HALF) {
                /* leading bit settled at 0; nothing to undo here */
            } else if (low >= HALF) {
                value -= HALF;
                low -= HALF;
                high -= HALF;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                value -= QUARTER;
                low -= QUARTER;
                high -= QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
            value = (value << 1) | br_read(&r);
        }
    }
    return 0;
}
