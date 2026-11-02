// Export the editable review deck and slide previews.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {build} from './deck_builder.mjs';
const HERE=path.dirname(new URL(import.meta.url).pathname);
const require=createRequire(new URL('./.build/runtime.cjs',import.meta.url));
const {Presentation,PresentationFile}=await import(require.resolve('@oai/artifact-tool'));
const output=path.join(HERE,'review_deck.pptx');
try{await fs.access(output);throw new Error('Preserve existing review_deck.pptx; reproduce in a new copied source directory.');}catch(e){if(e.code!=='ENOENT')throw e;}
const p=Presentation.create({slideSize:{width:1280,height:720}});
await build(p);
await fs.mkdir(path.join(HERE,'.build/deck_previews'),{recursive:true});
for(const [i,s] of p.slides.items.entries()){
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(HERE,'.build/deck_previews',`slide_${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
await (await PresentationFile.exportPptx(p)).save(output);
console.log(output);
