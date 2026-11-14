# P-coder hyperprior, gate 1 (2026-11-14)

`scripts/p3_hyperprior.py`, gate 1 of `docs/PLAN_pframe_hyperprior.md`. Run by
`run_gate1.sh`.

**Setup:**
- Two copies of the stage 3 P-coder per lambda get the same fine-tune: 5,000 steps, batch 8,
  lr 5e-5, seed 7, the same DAVIS TRAIN block16 pairs in the same order.
- One copy has the hyperprior (`z`: 32 channels at stride 64, new layers at lr 1e-3, the
  last `h_s` conv zero-initialized).
- Both are scored with the motion oracle's closed loop: VAL-B, GOP 10, block16 motion paid,
  ideal bits including `z`.

| lambda | coder | bits/P | of which z | P PSNR |
|---|---|---|---|---|
| 1e-3 | plain | 17,807 | 0 | 32.075 |
| 1e-3 | hyperprior | 17,851 | 8 | 32.076 |
| 3e-4 | plain | 28,131 | 0 | 33.093 |
| 3e-4 | hyperprior | 28,094 | 8 | 33.105 |
| 1e-4 | plain | 40,186 | 0 | 33.621 |
| 1e-4 | hyperprior | 40,186 | 8 | 33.633 |

P-frame BD-rate, hyperprior vs plain: **-0.4%**.

**GATE 1 NOT PASSED.** The bar, fixed in the plan before the run, was -8%.
- **The side information collapsed.** `z` shrank to 8 bits per frame at every rate (about
  0.03% of the P-frame). The coder learned to send almost nothing in it, and the two
  copies ended up the same.
- **What this does and does not show.** It shows this design, warm-started and fine-tuned
  for 5,000 steps, finds nothing worth sending. It does not prove side information cannot
  help: a zero-initialized side path with a rate cost on `z` from step 0 is a known recipe
  for this kind of collapse. A fair retry would need a rate-free warm-up for `z`.
- **Why a retry is not next.** `docs/RESEARCH_2026-11-14.md`, Part 0: on the videos where we
  lose most to x264, quality is capped by the transform's capacity, not by what the bits
  cost. A better prior makes a given quality cheaper and does not lift that cap.

Per the plan, the line stops here. `gate1.json` holds every number. `gate1.log` and the
fine-tuned models are not committed.
