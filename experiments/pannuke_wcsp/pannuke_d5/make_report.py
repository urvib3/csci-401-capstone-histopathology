"""Three-page office-hours PDF. Run with a Python providing ReportLab."""
import json,textwrap
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
HERE=Path(__file__).resolve().parent
OUT=HERE/'report.pdf'
r=json.loads((HERE/'results.json').read_text());coverage=json.loads((HERE/'coverage.json').read_text());examples=json.loads((HERE/'selected_examples.json').read_text())
c=canvas.Canvas(str(OUT),pagesize=(792,612));c.setTitle('PanNuke pixel WCSP: frozen learned costs at beta 0.30');c.setAuthor('Histopathology Capstone')
INK=HexColor('#202733');MUTED=HexColor('#506174');TEAL=HexColor('#227e80')
def text(x,y,s,size=13,bold=False,color=None):
    c.setFillColor(color or INK);c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawString(x,y,str(s))
def para(x,y,s,width=94,size=13,leading=18):
    for line in textwrap.wrap(s,width):text(x,y,line,size);y-=leading
    return y

def page(n,title):
    c.setFillColor(HexColor('#fafbf9'));c.rect(0,0,792,612,stroke=0,fill=1)
    text(34,578,'PANNUKE / FIXED BETA 0.30',11,True,TEAL);text(34,544,title,26,True)
    text(34,20,'200 GT training patches | 20 WCSP-held-out patches | CPU inference | No parameter sweep',10,color=MUTED);text(735,20,f'{n}/3',10)

def image(path,x,y,w,h):c.drawImage(str(path),x,y,width=w,height=h,preserveAspectRatio=True,anchor='c',mask='auto')

page(1,'Can learned neighbor costs improve raw TP labels?')
para(34,511,'Compare raw six-channel HoVer-Net TP argmax with pixel WCSP. Keep model probabilities as unaries; learn only the neighbor-cost table from training GT.',101)
text(34,448,'The pixel experiment',17,True)
y=423
for line in ['26,896 pixels; all six labels at every pixel.','Aligned central 164 x 164 GT and predictions.','24 neighbors; nearer neighbors matter more.','Raster ICM; strict improvements; up to 100 rounds.']:
    text(34,y,line,12);y-=20
text(34,326,'Train / test GT coverage',17,True)
text(34,302,'Class',12,True);text(151,302,'Train pixels',11,True);text(244,302,'Test pixels',11,True);text(333,302,'Patches',11,True)
names=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic']
for k,name in enumerate(names):
    y=280-k*20;text(34,y,name,11);text(151,y,f"{coverage['train']['class_pixels'][k]:,}",11);text(244,y,f"{coverage['test']['class_pixels'][k]:,}",11);text(333,y,f"{coverage['test']['patch_coverage'][k]}/20",11)
text(34,145,f"Tissues: {len(coverage['train']['tissues'])} train; {len(coverage['test']['tissues'])} test. No patch overlap.",12)
image(HERE/'figures/final_table.png',414,185,345,280)
para(414,165,'Counts -> prevalence-adjusted incompatibility -> 80% data + 20% fixed rule prior -> normalize mean off-diagonal cost to 1.',43,12,16)
para(34,98,'Four soft preferences: same-type neighbors; isolate Dead from Non-Neoplastic; discourage Immune/Non-Neoplastic contact; favor connective tissue at tumor/healthy boundaries.',98,12,16)
para(34,62,'Held out from WCSP fitting only. HoVer-Net training-fold exposure and patient/slide independence could not be verified.',108,10,13)
c.showPage()

page(2,'Results on the frozen 20-patch test set')
text(34,503,'Pooled metric',15,True);text(247,503,'Raw TP',15,True);text(337,503,'WCSP',15,True)
rows=[('Biological mIoU','biological_miou'),('Six-class mIoU','six_class_miou'),('Pixel accuracy','pixel_accuracy')]
for n,(label,key) in enumerate(rows):
    y=474-n*25;text(34,y,label,13);text(247,y,f"{r['baseline'][key]:.4f}",13);text(337,y,f"{r['wcsp'][key]:.4f}",13)
text(34,385,'Per-class IoU',16,True)
for k,name in enumerate(names):
    y=358-k*22;text(34,y,name,12);text(247,y,f"{r['baseline']['iou'][k]:.4f}",12);text(337,y,f"{r['wcsp']['iou'][k]:.4f}",12)
image(HERE/'figures/fixes_breaks.png',431,288,330,209)
d=r['biological_miou_difference'];lo,hi=r['bootstrap']['ci95']
text(431,262,f"Changed pixels: {r['changed_pixels']:,} / 537,920",13,True)
text(431,239,f"Biological mIoU change: {d:+.4f}",14,True)
text(431,216,f"Paired 95% CI: [{lo:+.4f}, {hi:+.4f}]",13)
text(431,194,'5,000 paired patch-bootstrap replicates',11)
conclusion=('Observed improvement; interval excludes zero.' if lo>0 else 'Observed decrease; interval excludes zero.' if hi<0 else 'No reliable improvement: interval includes zero.')
para(34,200,conclusion,38,17,22)
bg=coverage['test']['class_pixels'][0]/537920
para(34,151,f"Background is {bg:.1%} of GT pixels, so accuracy can hide biological-class errors. Biological mIoU gives each biological class equal weight. Pool confusion counts first; zero union gives IoU 0; no classes are dropped.",99,12,16)
rare=min(range(1,6),key=lambda k:coverage['test']['class_pixels'][k]);absent=[names[k] for k in range(6) if not coverage['test']['class_pixels'][k]]
para(34,98,f"GT-absent classes: {', '.join(absent) if absent else 'none'}. Rarest biological class: {names[rare]}, {coverage['test']['class_pixels'][rare]:,} pixels in {coverage['test']['patch_coverage'][rare]} patches. Patch-bootstrap uncertainty does not account for unknown patient clustering.",105,11,15)
mins=sum(x['solver']['verified_local_minimum'] for x in r['patches']);limits=sum(x['solver']['stop_reason']=='round_limit' for x in r['patches'])
text(34,49,f"Inference {r['inference_seconds']:.1f}s; solver + audits {r['solver_seconds']:.1f}s. Local minima {mins}/20; round limits {limits}.",11)
c.showPage()

page(3,'Where the labels changed, and what this means')
text(34,510,'Examples are extremes by fixed five-class patch mIoU change; they do not represent every patch.',12)
for n,kind in enumerate(['improvement','degradation']):
    p=HERE/f'figures/{kind}.png'
    if p.exists():image(p,28,300-n*188,736,185)
    else:text(34,420-n*184,f'No {kind} patch exists in this test set.',15,True)
para(34,100,'Limits: local pixel costs cannot enforce a stromal barrier or detect a dense immune cluster. ICM finds a single-pixel local minimum, not a global optimum. Coverage-enriched patches and unknown patient overlap limit generalization.',107,11,15)
para(34,53,'Office-hours questions: Are pixel contacts the right biological scale? How should rare Dead cells be protected? Would region-level context justify a separately preregistered experiment?',108,11,14)
c.save();print(OUT)
