// Editable review deck. All historical numbers come from local records cited in notes.
import fs from 'node:fs/promises';
import path from 'node:path';
const HERE=path.dirname(new URL(import.meta.url).pathname);
const ROOT=path.resolve(HERE,'../..');
const INK='#202733',TEAL='#227e80',MUTED='#506174',BG='#FAFBF9';
const FONT='Arial';
export async function build(presentation) {
 const r=JSON.parse(await fs.readFile(path.join(HERE,'results.json'),'utf8'));
 const coverage=JSON.parse(await fs.readFile(path.join(HERE,'coverage.json'),'utf8'));
 const tables=JSON.parse(await fs.readFile(path.join(HERE,'tables.json'),'utf8'));
 const config=JSON.parse(await fs.readFile(path.join(HERE,'frozen_config.json'),'utf8'));
 const examples=JSON.parse(await fs.readFile(path.join(HERE,'selected_examples.json'),'utf8'));
 const W=1280,H=720;
 // The live presentation uses the standard 16:9 canvas.
 function text(s,t,x,y,w,h,size=30,bold=false,color=INK){
  const box=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text=t;box.text.style={fontFamily:FONT,fontSize:size,bold,color};return box;
 }
 function slide(title,source='',titleSize=43){
  const s=presentation.slides.add({width:W,height:H});s.background.fill=BG;
  text(s,'HISTOPATHOLOGY CAPSTONE / REVIEW GUIDE',60,30,1120,24,16,true,TEAL);
  text(s,title,60,85,1160,titleSize>43?150:100,titleSize,true);
  text(s,`${presentation.slides.items.length}  |  Historical records are context; the new experiment has one fixed beta`,60,677,1160,24,15,false,MUTED);
  s.speakerNotes.text=`Sources: ${source}\nHistorical runs were read only, not rerun. Different baselines and aggregation methods are not controlled comparisons.`;return s;
 }
 function lines(s,items,y=210,size=30,step=80){for(const [i,t] of items.entries())text(s,t,66,y+i*step,1145,step-8,size);}
 function table(s,values,x=65,y=207,w=1150,h=330,size=27,widths){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.borders.assign({fill:'#D8DFE3',width:1,style:'solid'});
  for(let a=0;a<values.length;a++)for(let b=0;b<values[a].length;b++){
   const cell=t.getCell(a,b);cell.text.style={fontFamily:FONT,fontSize:size,bold:a===0,color:a===0?'#FFFFFF':INK};cell.fill=a===0?TEAL:(a%2?'#FFFFFF':'#F0F3F2');
  }return t;
 }
 async function image(s,name,x,y,w,h){s.images.add({path:path.join(HERE,'figures',name),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
 let s=slide('Can biological preferences improve segmentation?',path.join(ROOT,'deliverables/deliverable_3.md')+'; '+path.join(HERE,'frozen_config.json'),56);
 text(s,'A plain-language guide to the experiments so far',65,237,1100,70,35);
 text(s,'PanNuke nuclei • BCSS tissue regions • WCSP refinement',65,340,1100,60,29,false,TEAL);
 text(s,'The new test: learn neighbor costs from 200 GT patches, then compare one fixed pixel WCSP against raw TP argmax on 20 different patches.',65,450,1100,135,30);
 s=slide('The basic vocabulary',path.join(HERE,'README.md'));
 lines(s,["Model probability: how plausible each class is at one pixel.","Unary cost: penalize labels the model considers unlikely.","Pairwise cost: penalize combinations of nearby pixel labels.","Beta: controls neighbor influence. New experiment: fixed 0.30.","ICM: update one pixel at a time while reducing total cost."],200,29,83);
 s=slide('Baselines and attempts to reduce variables',path.join(ROOT,'deliverables/deliverable_3.md'));
 table(s,[['Experiment','Recorded result / lesson'],['PanNuke HoVer-Net, all Fold 2','Detection F1 0.833; multi-class PQ 0.437'],['Nucleus clustering, 10 patches','55 nuclei → 47 variables with DBSCAN / Ward'],['K-means nucleus clustering','55 → 21 variables; 9 mixed predicted-class groups'],['BCSS Phikon-v2 / U-Net','Linear probe mIoU 0.652; pretrained U-Net 0.649'],['BCSS region methods','Phikon, SLIC, Quickshift, RAG, Felzenszwalb, N-cut'] ],65,198,1150,360,24,[460,690]);
 text(s,'Oracle region scores use GT to label regions: they measure potential, not a deployed classifier. Baselines use different grids and tasks.',65,587,1140,68,24);
 s=slide('Early WCSP checks: correct solving ≠ better biology',path.join(ROOT,'deliverables/deliverable_4.md')+'; '+path.join(ROOT,'experiments/pannuke_d4/pannuke_task2/results.json'));
 table(s,[['Check','What happened'],['PanNuke 3 × 3 crop','8 active pixels; all 390,625 assignments checked'],['Strong fidelity / weak fidelity','No changes / 3 changes; no GT accuracy claim'],['BCSS: 500 validation tiles','U-Net mIoU 0.6414 → WCSP 0.6428'],['BCSS uncertainty','ROI-bootstrap Δ CI +0.0003 to +0.0023'] ],65,205,1150,320,27,[450,700]);
 text(s,'BCSS labels tissue regions; PanNuke labels nuclei. The small BCSS gain does not establish that the same rules help PanNuke.',65,570,1130,85,28);
 s=slide('BCSS: later table learning and solver comparisons',path.join(ROOT,'deliverables/deliverable_5.md'));
 table(s,[['Recorded validation method: 2,124 tiles','mIoU'],['U-Net baseline','0.6489'],['WCSP, handwritten costs, lambda 8','0.6507'],['WCSP, data-derived costs, lambda 8','0.6506'],['Time-limited toulbar2, lambda 4','0.6492'] ],65,205,1150,330,28,[890,260]);
 text(s,'Handwritten-table gain CI: +0.0010 to +0.0027. Most gain was in the mixed BCSS “other” class. This is not the PanNuke Non-Neoplastic class.',65,575,1130,80,26);
 s=slide('BCSS: image terms and search limits',path.join(ROOT,'deliverables/deliverable_5.md'));
 lines(s,['Image-color term on 185 dev tiles: no image term mIoU 0.7097; beta 1 gives 0.7092. No improvement over the same dev reference.','Saved record marks image-term validation as still running. No completed validation result is invented here.','At lambda 4: ICM used about 12 s/tile; toulbar2 about 87 s/tile with 20-second block limits.','An exact-search algorithm with a time limit does not certify a global optimum. Lower energy is also not a guarantee of better GT accuracy.'],195,28,110);
 s=slide('PanNuke: active pixels and flat change penalties',path.join(ROOT,'deliverables/deliverable_5_pannuke_multipatch.md'));
 lines(s,['Initial 20-patch run: only 2 pixels changed; 0 fixes, 2 breaks.','No patch improved; one worsened; 19 were unchanged.','Earlier development lambda sweep: lowering fidelity still produced more breaks than fixes. At lambda 0: 2 fixes vs 61 breaks.','Lesson: hard-label costs and disconnected active pixels offered little useful correction.'],210,29,102);
 s=slide('Two-stage WCSP: mask first, then nucleus type',path.join(ROOT,'experiments/pannuke_d5/README_TWO_STAGE.md'));
 table(s,[['15-patch development method','Historical mean biological mIoU'],['Saved HoVer-Net','0.474933'],['Stage 1 only: refine mask','0.474896'],['Both stages: flat costs','0.426603'],['Both stages: class-dependent unaries','0.432511'] ],65,205,1150,325,28,[740,410]);
 text(s,'The class-dependent version made 85 fixes and 1,961 breaks. Nucleus reconstruction and voting belong to this historical method only.',65,572,1120,80,27);
 s=slide('One-stage probability costs and initialization',path.join(ROOT,'experiments/pannuke_d5/README_FULLGRID.md'));
 table(s,[['15-patch historical method','Mean biological mIoU'],['Saved postprocessed HoVer-Net','0.474933'],['NP/conditional-TP probability argmax','0.443796'],['Probability WCSP, saved-label initialization','0.475899'],['Same plus calibrated type prior','0.475922'],['Pixel-only initialization','0.447382'] ],65,195,1150,360,26,[820,330]);
 text(s,'These historical probabilities combine heads differently from the new raw-TP baseline. Starting labels materially affected the local solution.',65,585,1130,68,25);
 s=slide('Raw TP and the earlier frozen evaluation',path.join(ROOT,'experiments/pannuke_d5/README_FULLGRID.md'));
 lines(s,['Earlier raw six-channel TP development test: mean biological mIoU 0.438640 → 0.440987; saved HoVer-Net was 0.474933.','A previous strength calibration chose unchanged HoVer-Net overall. Its frozen WCSP candidate used strength 2.','That candidate on 20 new patches: class mIoU 0.555525 → 0.559888, but binary nucleus IoU fell 0.770946 → 0.764872.','These are historical records. This task does not repeat that sweep or reuse that chosen strength.'],200,28,110);
 s=slide('100 more patches: small gains and metric traps',path.join(ROOT,'experiments/pannuke_d5/README_FULLGRID.md')+'; '+path.join(ROOT,'experiments/pannuke_d5/pixel_wcsp_extended_evaluation/BACKGROUND_REASSESSMENT.md'));
 lines(s,['Earlier 100-patch test: mean biological mIoU 0.595110 → 0.595380. Paired 95% interval: -0.007761 to +0.008223.','44 patches improved; 55 worsened; one was unchanged.','A later metric reassessment pooled six-class counts: 0.608028 → 0.636522. Most net gain came from rare Dead cells.','Patch means, GT-present-class averages, and pooled fixed-class mIoU answer different questions. Do not compare them as one scoreboard.'],195,28,113);
 s=slide('The new experiment: raw TP → pixel WCSP',path.join(HERE,'frozen_config.json'));
 lines(s,['HoVer-Net fast PanNuke → raw TP logits → six-channel softmax.','Baseline: take the most probable type at every output pixel.','WCSP: all 164 × 164 pixels; all six classes; probability unaries.','24 spatial neighbors; cost = 0.30 × learned table / distance.','No NP confidence, HV, instance IDs, centroids, or nucleus voting.'],200,29,84);
 s=slide('Frozen split and biological class coverage',path.join(HERE,'coverage.json')+'; '+path.join(HERE,'split.json'));
 const short=['Background','Neoplastic','Inflammatory','Connective','Dead','Non-Neoplastic'];
 table(s,[['GT class','Train pixels','Test pixels','Test patches'],...short.map((n,k)=>[n,coverage.train.class_pixels[k].toLocaleString('en-US'),coverage.test.class_pixels[k].toLocaleString('en-US'),`${coverage.test.patch_coverage[k]} / 20`])],65,195,1150,350,26,[420,260,250,220]);
 text(s,'200 training patches / 18 tissues; 20 test patches / 19 tissues. 266 prior-use IDs excluded. Patient IDs unavailable; checkpoint exposure unknown.',65,575,1140,85,26);
 s=slide('How training GT becomes a compatibility table',path.join(HERE,'tables.json')+'; '+path.join(HERE,'README.md'));
 lines(s,['1. Count actual neighbor labels in training GT; add both orientations.','2. Add 1 to every count; compare joint adjacency to class prevalence.','3. Convert unexpectedly rare contacts to nonnegative costs; zero diagonal.','4. Blend 80% learned table with 20% fixed biological prior.','5. Normalize mean off-diagonal cost to 1; freeze before testing.'],195,28,77);
 text(s,'Rules: homophily; Dead isolation; immune infiltration preference; connective boundary preference. These are soft modeling choices.',65,598,1140,60,24,false,TEAL);
 s=slide('The final frozen table',path.join(HERE,'tables.json'));
 const tags=['BG','Neo','Immune','Stroma','Dead','Non-Neo'];
 const t=table(s,[['',...tags],...tables.final.map((row,i)=>[tags[i],...row.map(v=>v.toFixed(2))])],150,195,980,360,27,[170,135,135,135,135,135,135]);
 const mx=Math.max(...tables.final.flat());
 for(let i=0;i<6;i++)for(let j=0;j<6;j++){
  const v=tables.final[i][j]/mx;
  const rr=Math.round(255-60*v),gg=Math.round(250-200*v),bb=Math.round(230-170*v);
  t.getCell(i+1,j+1).fill=`#${[rr,gg,bb].map(x=>x.toString(16).padStart(2,'0')).join('')}`;
  t.getCell(i+1,j+1).text.style={fontFamily:FONT,fontSize:27,color:v>.65?'#FFFFFF':INK};
 }
 text(s,'Higher values discourage contact more strongly. Diagonal = 0. Background costs come from data. Beta 0.30 is applied after this table.',65,580,1140,75,27);
 s=slide('New results: pooled fixed-class IoU',path.join(HERE,'results.json'));
 table(s,[['Metric','Raw TP argmax','WCSP, beta 0.30'],['Biological mIoU',r.baseline.biological_miou.toFixed(4),r.wcsp.biological_miou.toFixed(4)],['Six-class mIoU',r.baseline.six_class_miou.toFixed(4),r.wcsp.six_class_miou.toFixed(4)],['Pixel accuracy',r.baseline.pixel_accuracy.toFixed(4),r.wcsp.pixel_accuracy.toFixed(4)],...short.map((n,k)=>[n+' IoU',r.baseline.iou[k].toFixed(4),r.wcsp.iou[k].toFixed(4)])],65,185,1150,400,24,[610,270,270]);
 text(s,`Δ biological mIoU ${r.biological_miou_difference>=0?'+':''}${r.biological_miou_difference.toFixed(4)}; paired 95% CI [${r.bootstrap.ci95.map(v=>v.toFixed(4)).join(', ')}].`,65,645,1150,30,24,true,TEAL);
 s=slide('Corrections, new errors, and changed wrong labels',path.join(HERE,'results.json'));
 const correctionChart=s.charts.add('bar',{position:{left:85,top:210,width:1110,height:340},categories:['Wrong → correct','Correct → wrong','Wrong → different wrong'],series:[{name:'Changed pixels',values:[r.wrong_to_correct,r.correct_to_wrong,r.wrong_to_different_wrong],fill:TEAL}],hasLegend:false,dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:25,fill:INK}},xAxis:{textStyle:{fontSize:24,fill:INK}},yAxis:{textStyle:{fontSize:20,fill:MUTED}}});
 for (const style of [correctionChart.xAxis.textStyle,correctionChart.yAxis.textStyle,correctionChart.dataLabels.textStyle]) style.typeface=FONT;
 text(s,`${r.changed_pixels.toLocaleString('en-US')} changed pixels. Background is ${(100*coverage.test.class_pixels[0]/537920).toFixed(1)}% of GT. Accuracy is dominated by background; biological mIoU weights each biological class equally.`,65,584,1130,80,27);
 for(const kind of ['improvement','degradation']){
  const e=examples.examples.find(x=>x.kind===kind);s=slide(kind==='improvement'?'Example improvement':'Example degradation',path.join(HERE,'selected_examples.json')+'; '+path.join(HERE,'predictions'));
  if(e){await image(s,kind+'.png',45,208,1190,410);text(s,'Largest '+(kind==='improvement'?'positive':'negative')+' fixed-five-class patch mIoU change; an illustrative extreme, not a representative sample.',65,625,1130,40,21);}
  else lines(s,[`No ${kind} patch exists in this frozen test set.`],300,34,90);
 }
 s=slide('Interpretation, limits, and office-hours questions',path.join(HERE,'results.json')+'; '+path.join(HERE,'README.md'));
 const [lo,hi]=r.bootstrap.ci95;
 const conclusion=lo>0?'Observed gain; the patch-bootstrap interval excludes zero.':hi<0?'Observed degradation; the patch-bootstrap interval excludes zero.':'No reliable improvement: the patch-bootstrap interval includes zero.';
 lines(s,[conclusion,'Pairwise contacts cannot literally enforce stromal barriers or detect dense immune clusters. Rare Dead cells remain fragile evidence.','All results are single-pixel local solutions, not global optima. Unknown patient clustering and pretrained-fold exposure limit claims.','Questions: Is pixel adjacency the right biological scale? Would region-level context help? What should a separately frozen follow-up test measure?'],195,28,105);
 s.speakerNotes.text+=`\nReproducibility: ${HERE}/README.md; frozen config SHA256 ${r.prediction_provenance.frozen_config_sha256}; checkpoint SHA256 ${config.checkpoint.sha256}. Numerical tests, manifests, tables, label predictions, confusion matrices and bootstrap replicates accompany the report. No commit or push. Deliverable 5 files untouched.`;
 return {slides:presentation.slides.items.length};
}
