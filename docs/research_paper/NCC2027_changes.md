# What changed, and why — NCC 2027 submission

Target: **NCC 2027**, IISc Bangalore (with IIT Palakkad and IIT Goa), 17–20 Feb 2027.
**Deadline 31 October 2026, 23:59 IST.** 6 pages, IEEE format, **double-blind**.

Output: `Research_Paper_NCC2027_corrected.docx` — 3,796 body words, 4 figures,
2 tables, 1.43 MiB, IEEE template and section structure preserved.

Every number in the paper was checked against the repository. Numbers the repo
does not support were removed rather than reworded.

---

## 1. Factual errors fixed

**The water-filling claim was wrong for the design the paper advocates.** It read
as a remaining gain for the 64-channel transform ("0.06–0.16 dB of 6.65–8.33
percent savings"). Those figures belong to the *192-channel* model (−6.65%) and
the *593 k baseline* (−8.33%). On the trained 64-channel model it is **+0.00%** —
zero channels below 1% of peak energy, so water-filling allocates the full depth
to all 64 and changes nothing (`outputs/stage1_latent_shape/latent_shape_64ch_trained.json`).
Now stated as "reverse water-filling bit allocation recovers nothing on this
transform."

**Motion's byte share was from the wrong operating points.** The paper said
"2–5%". Measured: **1.98% at 8-bit, 5.47% at 4-bit**, and ~4–9% across the
deployed 3/4/5-bit points. Now given as "about 2% at 8-bit and 5.5% at 4-bit",
with the 2:1 payback retained (confirmed).

**"Seven candidate changes were rejected" → nine.** The 192-channel latent and
the 2-residual-block transform were both rejected on measured BD-rate.

**A sign-inverted sentence.** "the wider-transform reconstructs with a 30.56 dB
vs. 30.14 dB (+0.43 dB) **error**" — those are PSNRs, where higher is better.
This was the single most damaging sentence in the draft; a signal-processing
reviewer would have stopped there.

**`Fig. 2(c)` did not exist.** Fig. 2 has no subpanels. The +87.9% result now
cites Table II, where it actually lives.

**Reference [3] was credited with GDN.** GDN is [11]. Corrected.

**"Ballalle et al."** and several other broken fragments ("codes 3.16–3.26
codes", "Already-in-disabled", "flimed to another discontinuity-vfree validation
subset") — the draft had been through a paraphraser and large parts were not
parseable. All prose rewritten.

---

## 2. Missing results added

**The 2-residual-block rejection** (19 Oct), absent entirely. 9,864,643
parameters at an identical symbol count — free in rate — regressed to **+101.5%**
against 1 block's +87.9%, losing at every operating point *while fitting Vimeo
better* (val loss 1.19× lower, val PSNR 0.9 dB higher). Reported as a
**generalisation gap, not a training failure**, which is the more interesting
claim.

**The quantization-fragility decomposition**, the strongest current finding and
now the Conclusion's analytical core. The 64-channel transform leads the baseline
by a **flat +2.45 dB** with no quantizer in the path, but surrenders **0.18, 0.63
and 1.71 dB more** than the baseline at 5/4/3-bit. Subtracting the two reproduces
the measured gate advantage of +2.37/+1.88/+0.79 dB almost exactly. By 3 bits,
fragility consumes 70% of what the transform earned.

**Table II extended** from one all-intra row to four, so the whole transform
study is in a table rather than buried in prose:

| | PSNR | MS-SSIM |
|---|---|---|
| 593 k baseline, all-intra | +176.2% | +167.6% |
| GDN/residual 192-ch | +560.6% | +456.1% |
| **GDN/residual 64-ch** | **+87.9%** | **+110.1%** |
| GDN/residual 64-ch, 2 blocks | +101.5% | +118.8% |

**Future Work now follows the diagnosis** — quantization-aware fine-tuning first
(it targets the 1.71 dB), then the hyperprior as what makes a wider latent
affordable, then learned flow and joint training. The draft listed the hyperprior
and optical flow without connecting them to any measurement.

---

## 3. Title changed

From *"Real-Time Neural Video Compression: …"* to

> **A Motion-Compensated Learned Video Codec with Context-Adaptive Entropy
> Coding: Measured Bitstreams and a Diagnosis of the Gap to H.264/H.265**

The paper's own Limitations section states 96 ms/frame to encode and 118–170
ms/frame to decode. Claiming "Real-Time" in the title and disclaiming it on the
last page is the kind of contradiction a reviewer enjoys finding. The new title
also states the actual contribution — measurement and diagnosis — which is the
paper's real strength given that it does not beat H.264.

**Revert this if the project title must stay fixed for your NTCC records**, but
then drop "Real-Time" from the abstract too, or add "toward" somewhere.

---

## 4. Double-blind anonymisation

- Author names, affiliations and e-mail addresses removed from the body.
- **Three `mailto:` hyperlink relationships survived inside the package** after
  the visible text was deleted. Removed.
- **Document metadata carried `dc:creator = "Aditya Aggarwal"` and the old title.**
  Removed. This would have reached reviewers.
- Repository name, package name and the hosting service de-identified
  ("a Python package", "a hosted notebook service").
- Verified by scanning **every XML part** in the package, not just the visible
  text: no occurrence of any author name, affiliation, address or `mailto:`.

**Keep the original file** — you need the named version for the camera-ready.

---

## 5. Cut to 6 pages

Was ~20% over. Removed:

- **Section VI (Per-Frame Coding Behaviour) and Fig. 5.** The biggest single
  saving: the figure alone was 3.8 MB and the section was illustrative rather
  than evidential. The orphaned image was purged from the package too, taking the
  file from 5.35 MiB to 1.43 MiB.
- **Related Work: five paragraphs → three.**
- **Implementation: six subsections → three paragraphs.** All the measurements
  survive (0.026% coder overhead, 54/54 byte-identical streams, 463→96 ms/frame,
  437→16 ms/P-frame, the 38-of-196,608 determinism finding); only the headings
  and repetition went.
- The interactive demo is no longer claimed as a contribution, since its section
  is gone. Add it back only if you also cut something else.

Figures 1–4 and both tables are kept, with all equations.

---

## 6. Verified correct — no change needed

Worth knowing so you don't re-check: **Table I matches the benchmark JSON to
four decimal places** on all nine rows, PSNR and MS-SSIM. So do **all four
classical BD-rate rows in Table II**, the parameter counts (593,411 / 8,437,827 /
7,208,899 / 9,864,643), λ = 3×10⁻⁴, −18.12%, −32.22%/−41.86%, −4.81%, +0.02%,
70–88%, 99.97–100.02%, the entropy-model parameter ranges
(11,880–12,672 and 12,744–13,536), K = 512, 400 uniform calibration frames,
97.3%, 1.568→1.239 bits, 37 of 64 channels, M22's 1.06/1.31/1.76%, +306.2%,
the 2.8×-vs-1.8× all-intra penalty, 3.16–3.26×, 68% of bit depth, and +560.6% /
+87.9% / +101.5%.

---

## 7. Still to do before you submit

1. **Check the figures for identifying content** — I cannot read what is inside
   the images. Look for file paths, window titles, usernames or watermarks in
   Figs. 1–4.
2. **Confirm the 6-page fit in Word.** My 3,796-word estimate assumes the figures
   stay at their current sizes; the real test is the page count on screen.
3. **Decide on the title** (Section 3 above).
4. **Equation formatting** survived as text and may need the template's equation
   style re-applied — check (1)–(5) render correctly.
5. **Do not wait for the QAT run.** Nine days is enough to polish this; it is not
   enough to train, gate and write up QAT. The +87.9% result is a complete
   contribution on its own, and QAT belongs in the camera-ready or a follow-up.

---

## 8. QAT folded in (2026-10-24, `cd4c35e`)

The QAT run landed after the first corrected draft, so the paper was updated:

- **Abstract** now ends on +46.5% and frames it as a *confirmed prediction*
  rather than an extra result.
- **Contribution (iv)** upgraded: the diagnosis "makes a quantitative prediction
  about where quantization-aware training should help that a subsequent run
  confirms." That is a materially stronger claim than a diagnosis alone.
- **New paragraph** in Transform Capacity carrying the fragility decomposition:
  ceiling 32.589 → 33.092 dB (+0.503, training) plus +0.332/+0.391/+0.633 dB
  (fragility), summing to within 0.06 dB of the measured +0.86/+0.95/+1.16 dB,
  with the fragility share growing 42% → 57% as the quantizer coarsens.
- **Table II** gains `GDN/residual 64-ch + QAT, all-intra | +46.5% | +55.8%`.
- **Conclusion** reframed around the three-step sequence: +176.2% → +87.9% →
  +46.5%, each confirmed in real coded bytes.
- **Future Work** now leads with refitting the P-frame machinery, since QAT is
  done. Per-channel bit allocation is mentioned as newly worth −3.3%.
- **Limitations** gains the missing QAT control: the fine-tune also trained for
  further epochs, so it does not isolate the noise; only the fragility component
  is isolated, via the float ceiling.

### To fit six pages

Figure 3 (BD-Rate Contribution per Milestone) was **removed** — it restated
numbers the text already gives, making it the cheapest of the four to lose. The
decoded-frame figure survives as the new Fig. 3, renumbered, and its caption was
normalised to the template's `Fig.<NBSP>n.<EM-SPACE>` typography. Prose was
trimmed in Related Work, the container/bundle paragraph, Experimental Setup, the
protocol paragraph and the reference-conditioned-model paragraph. No measurement
was dropped.

**Now 4,172 body words, 3 figures, 2 tables, 1.36 MiB.** That is close to the
limit rather than comfortably inside it, so **check the page count in Word before
submitting**; if it runs a few lines over, shrinking Fig. 1 or Fig. 2 is the
least damaging fix.
