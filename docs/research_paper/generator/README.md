# Rebuilding the NCC 2027 paper

The paper is generated, not hand-edited. Change the text in `paper_content.py`,
then rebuild. All three scripts run from the repository root with the project venv.

```
./.venv/Scripts/python.exe docs/research_paper/generator/figs.py <figs_dir>
./.venv/Scripts/python.exe docs/research_paper/generator/build_paper.py docs/research_paper/generator/template.docx <figs_dir> docs/research_paper/Research_Paper_NCC2027_submission.docx
./.venv/Scripts/python.exe docs/research_paper/generator/build_paper.py docs/research_paper/generator/template.docx <figs_dir> docs/research_paper/Research_Paper_NCC2027_camera_ready.docx --authors
```

- `paper_content.py`: title, abstract, body, tables and references. Inline markup is
  `*italic*`, `_{sub}`, `^{sup}`; cite with `[@key]`, numbered by first appearance.
  A reference that is never cited stops the build.
- `figs.py`: Fig. 1 (architecture of the current codec) and Fig. 2 (RD curves on UVG,
  MCL-JCV and DAVIS, read from `outputs/p13_audit_fixes/`, `outputs/p11_mcljcv/` and the
  x264/x265 anchor files in `outputs/benchmarks/`). The 2026-11-26 revision dropped the
  per-frame views figure, so `per_frame_views_4rows.png` is no longer used.
- `build_paper.py`: writes the body into the IEEE template (`template.docx`, the
  pre-revision paper). It sets 0.764 in margins on every side (NCC 2027 requires at
  least 3/4 in), single line spacing, two columns, and strips author metadata.
  Without `--authors` the output has no author block, for double-blind review.

Check the page count in Word after any edit: NCC 2027 allows six pages including
references.
