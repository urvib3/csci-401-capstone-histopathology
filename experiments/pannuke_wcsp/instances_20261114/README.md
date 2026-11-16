# PanNuke WCSP over fixed nucleus instances

One variable represents each detected nucleus and takes one of five foreground
types. The objective adds a per-nucleus type cost to a weighted neighbor
cost. The solver can change the type assigned to a mask. Every detected mask
and all background pixels are retained.

- **Inputs:** same frozen HoVer-Net checkpoint. Fuse four reflect-padded tiles
  over the full 256×256 patch. Apply the installed HoVer-Net watershed
  to the fused nucleus probability and horizontal/vertical maps.
- **Baseline:** label each fixed mask by the most frequent non-background
  pixel type. If every pixel votes Background, use the highest pooled
  foreground probability so the mask still has a label. This baseline is
  computed on the full fused instance maps for a matched comparison.
- **Unary costs:** smoothed foreground votes, negative log of the pooled
  foreground probabilities, or mean foreground negative log probability.
  A development follow-up also tested a smoothed area likelihood learned
  from the ground-truth fitting masks. Edge-clipped nuclei skip that term.
- **Pairwise costs:** distance-weighted Potts or a scale-appropriate hybrid
  fitted on ground-truth nucleus neighbors, with an 80/20 learned/biological
  blend. Use up to four nearest candidates per nucleus within 32 pixels,
  symmetrize the edges, and normalize weights by weighted degrees.
- **Solver:** deterministic, bidirectional ICM from the baseline and unary
  solutions. Keep the lowest energy found and audit all single-node moves.
- **Selection:** 200 existing fitting patches for adjacency and area counts,
  12 existing development patches for model choice, and 100 newly sampled
  patch IDs for evaluation. The diagnostic with correct types on IoU > 0.5
  matches leaves unmatched detections as they were. Patch-level bootstrap
  uses 5,000 paired replicates. Patient and checkpoint independence are unverified.

Run from this folder using the repository environment:

```sh
../../../.venv/bin/python prepare.py prepare
../../../.venv/bin/python prepare.py dev
../../../.venv/bin/python run.py dev
../../../.venv/bin/python prepare.py test
../../../.venv/bin/python run.py final
../../../.venv/bin/python validate.py
```

The initial development grid without size costs is retained locally in
`initial_dev/`. The size-cost follow-up and final selection were also
completed on development patches before evaluating any models on the new
sample. The comparison includes the selected positive-beta WCSP, its
matched beta-zero control, the fixed-mask vote baseline, Raw TP,
and the diagnostic with ground-truth types.

See [settings](frozen.json), [development scores](development.json),
[split](split.json), [final comparison](evaluation.json), and
[per-patch scores](per_patch.csv).
