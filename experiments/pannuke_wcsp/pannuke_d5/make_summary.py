"""Readable factual results, including class coverage and solver audits."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'results.json').read_text());c=json.loads((HERE/'coverage.json').read_text());f=json.loads((HERE/'frozen_config.json').read_text())
short=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic Epithelial']
rows=['# Fixed beta 0.30: learned pixel compatibility costs','','Only raw TP argmax and frozen WCSP were evaluated. No sweep or post-test tuning.','',
'| Metric | Raw TP | WCSP |','|---|---:|---:|']
for label,key in [('Pooled biological mIoU','biological_miou'),('Pooled six-class mIoU','six_class_miou'),('Pixel accuracy','pixel_accuracy')]:rows.append(f"| {label} | {r['baseline'][key]:.6f} | {r['wcsp'][key]:.6f} |")
rows+=['','| Class | GT pixels | Test patch coverage | Raw TP IoU | WCSP IoU |','|---|---:|---:|---:|---:|']
for k,name in enumerate(short):rows.append(f"| {name} | {c['test']['class_pixels'][k]:,} | {c['test']['patch_coverage'][k]}/20 | {r['baseline']['iou'][k]:.6f} | {r['wcsp']['iou'][k]:.6f} |")
lo,hi=r['bootstrap']['ci95'];d=r['biological_miou_difference']
rows+=['',f'Biological mIoU difference: **{d:+.6f}**. Paired patch-bootstrap 95% CI: **[{lo:+.6f}, {hi:+.6f}]**, 5,000 replicates; seed 20261102.']
if lo>0:rows+=['Observed improvement, with the patch-bootstrap interval excluding zero. This is conditional on this coverage-enriched sample and the patch-independence assumption.']
elif hi<0:rows+=['No improvement: the observed biological mIoU decreases, and the patch-bootstrap interval excludes zero.']
else:rows+=['No reliable improvement is established: the interval includes zero. The observed biological mIoU decreases; uncertainty includes both decreases and increases.']
rows+=['','The 114 net corrections improve pixel accuracy, but inflammatory, connective and Dead IoUs fall. Biological mIoU gives each class equal weight, so pixel-count gains do not guarantee a biological mIoU gain.']
rows+=['',f"Changed: {r['changed_pixels']:,}; wrong → correct: {r['wrong_to_correct']:,}; correct → wrong: {r['correct_to_wrong']:,}; wrong → different wrong: {r['wrong_to_different_wrong']:,}.",
f"Background is {c['test']['class_pixels'][0]/537920:.2%} of GT. No GT class is absent across the pooled test set. Dead is rare (1,301 test pixels across three patches; 2,842 training pixels across two patches). Bootstrap replicates may omit rare-class patches, but fixed class denominators and the zero-union policy remain unchanged.",
'',f"Inference: {r['inference_seconds']:.3f}s; solver including audits: {r['solver_seconds']:.3f}s; model setup: {r['model_setup_seconds']:.3f}s. The explicit synthetic warmup and beta-zero validation are excluded; any additional real-layout JIT specialization inside a timed solve is included.",'',
'| Fold 2 patch | Tissue | Rounds | Stop | Verified local minimum | Residual gain | Fixes | Breaks | Inference s | Solver s |','|---|---|---:|---|---|---:|---:|---:|---:|---:|']
for x in r['patches']:
 s=x['solver'];rows.append(f"| {x['image_index']:04d} | {x['tissue']} | {s['rounds']} | {s['stop_reason']} | {s['verified_local_minimum']} | {s['max_remaining_gain']:.3g} | {x['wrong_to_correct']} | {x['correct_to_wrong']} | {x['inference_seconds']:.3f} | {x['solver_seconds']:.3f} |")
rows+=['','All beta-zero outputs exactly match raw TP argmax. Every round has an independently recomputed non-increasing energy; local-move checks use separate vectorized code and tolerance 1e-10. No global optimum is claimed.',
'','The 200 training and 20 test patches are disjoint, across 18 and 19 tissues respectively. Prior-use scan excludes 266 patch IDs, but unrecorded use cannot be verified. No slide/patient identifiers are available. Held out refers to WCSP fitting; pretrained HoVer-Net Fold 2 exposure is unknown.',
'','Pairwise preferences cannot enforce literal stromal barriers or detect dense immune clusters. The learned table sees only two Dead-containing training patches; rare-class cost estimates need caution. Test selection intentionally enriches coverage and is not a prevalence-representative sample.',
'',f"Checkpoint SHA256: `{f['checkpoint']['sha256']}`.",f"Frozen config SHA256: `{r['prediction_provenance']['frozen_config_sha256']}`.",f"Starting commit: `{f['provenance']['git_commit']}`. No commit or push made."]
if 'publication_metadata' in f:
 rows += ['', 'Original evaluation config SHA256: `' + f['publication_metadata']['original_frozen_config_sha256'] + '`. File-path metadata was sanitized after evaluation; scientific settings and saved predictions are unchanged.']
(HERE/'RESULTS.md').write_text('\n'.join(rows)+'\n')
