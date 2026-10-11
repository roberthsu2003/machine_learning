// Verify scene renderers and the educational steps, without a browser or dependencies.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const catalog=JSON.parse(await readFile(new URL('./chapters.json',import.meta.url)));
const context=new Proxy({}, {get:(_,key)=>key==='measureText'?text=>({width:String(text).length*10}):()=>{},set:()=>true});
let states=0,scenes=0;
const render=(def,params,progress)=>{
 const s={params:{seed:42,...params},progress,time:progress*16,phase:progress*4,canvas:{getBoundingClientRect:()=>({width:760})}};
 const steps=typeof def.steps==='function'?def.steps(s):def.steps??12;
 s.step=Math.min(steps,Math.floor(progress*steps+1e-6));
 const result=def.render(context,s);states++;
 const finite=(value,path='metrics')=>{if(typeof value==='number')assert.ok(Number.isFinite(value),`${def.id}: ${path}=${value}`);else if(value&&typeof value==='object')for(const [key,v] of Object.entries(value))finite(v,`${path}.${key}`);};
 finite(result.metrics);assert.ok(result.message,`${def.id}: missing explanation`);
 return result.metrics;
};
for(const chapter of catalog.chapters){
 const {definitions}=await import(`./${chapter.slug}/scenes/index.js`);
 assert.deepEqual(definitions.map(d=>d.id),chapter.scenes.map(s=>s.id));
 for(const def of definitions){
  scenes++;
  const defaults=Object.fromEntries(def.fields.map(field=>[field.key,field.value]));
  const cases=[defaults];
  for(const field of def.fields){const values=field.type==='select'?field.options.map(([value])=>value):field.type==='checkbox'?[false,true]:[field.min,field.max];for(const value of values)cases.push({...defaults,[field.key]:value});}
  for(const params of cases)for(const progress of [0,.25,.5,.75,1])render(def,params,progress);
  if(def.id==='selection'){
   // Balanced orthogonal demo: β=(3,.25,1.5,0); holdout reverses the spurious X2 effect.
   const {fitSelection,selectionCandidates}=await import('./04-feature-engineering/scenes/selection.js');
   for(const alpha of [0,.1,.25,.6,1.5,3,3.5]){
    const model=fitSelection([0,1,2,3],alpha);
    const expected=[3,.25,1.5,0].map(w=>Math.max(0,w-alpha));
    model.weights.forEach((w,j)=>assert.ok(Math.abs(w-expected[j])<1e-9));
    const m=render(def,{...defaults,method:'embedded',alpha},1);
    assert.deepEqual(m.selected,expected.flatMap((w,j)=>w>1e-8?[j]:[]));
    assert.equal(m.alpha,alpha);
   }
   for(const keep of [1,2,3,4]){
    const list=selectionCandidates(keep),total=list.length*3+1;
    assert.equal(list.length,[0,4,6,4,1][keep]);
    for(let step=0;step<=total;step++)assert.equal(render(def,{...defaults,method:'wrapper',keep},step/total).evaluated,Math.min(list.length,Math.floor(step/3)));
    const result=render(def,{...defaults,method:'wrapper',keep},1);
    assert.equal(result.selected.length,keep);
    assert.ok(Math.abs(result.best.validationMSE-Math.min(...list.map(m=>m.validationMSE)))<1e-9);
    if(keep===2){assert.deepEqual(result.selected,[0,2]);assert.ok(Math.abs(result.best.validationMSE-.1025)<1e-9);}
   }
  }
  if(def.id==='scaling'){
   const {scalingExample}=await import('./04-feature-engineering/scenes/scaling.js');
   for(const outlier of [false,true])for(const method of ['raw','z','minmax']){
    const original=scalingExample({method,unit:'m2',outlier});
    const converted=scalingExample({method,unit:'kilo',outlier});
    assert.equal(original.trainingCount,outlier?7:6);
    assert.ok(Math.abs(original.stats[0].mean-(outlier?9010/7:4010/6))<1e-9);
    assert.equal(original.beforeNearest,'A');assert.equal(converted.beforeNearest,'B');
    if(method!=='raw'){
     assert.equal(original.afterNearest,'B');assert.equal(converted.afterNearest,'B');
     original.scaled.forEach((row,i)=>row.forEach((v,j)=>assert.ok(Math.abs(v-converted.scaled[i][j])<1e-9)));
     original.after.forEach((p,i)=>assert.ok(Math.abs(p.distance-converted.after[i].distance)<1e-9));
    }else assert.deepEqual(original.after,original.before);
    for(const unit of ['m2','kilo'])for(let step=0;step<=4;step++){
     const m=render(def,{...defaults,method,unit,outlier},step/4);assert.equal(m.step,step);
     for(const p of [...m.before,...m.after])assert.ok(Math.abs(p.distance**2-p.parts.reduce((a,b)=>a+b,0))<1e-8);
    }
   }
   assert.equal(scalingExample({method:'raw'}).before[0].squared,109);
   assert.equal(scalingExample({method:'raw'}).before[1].squared,40000);
  }
  if(def.id==='split-strategies')for(const mode of ['random','stratified','time'])for(const ratio of [40,70,90]){
   const m=render(def,{...defaults,mode,ratio},1);
   assert.equal(m.train.length+m.test.length,40);assert.equal(new Set([...m.train,...m.test]).size,40);assert.equal(m.arrived.flat().length,40);
   if(mode==='time')assert.ok(Math.max(...m.train)<Math.min(...m.test));
  }
  if(def.id==='cross-validation')for(const k of [3,4,5,6])for(let fold=0;fold<=k;fold++)assert.equal(render(def,{...defaults,k},fold/k).completed,fold);
  if(def.id==='stacking')for(const folds of [3,4,5])for(let fold=0;fold<=folds;fold++)assert.equal(render(def,{...defaults,folds},fold/folds).completed,fold);
  if(def.id==='bagging')for(const models of [3,5,7])for(let model=0;model<=models;model++)assert.equal(render(def,{...defaults,models},model/models).used,model);
  if(def.id==='inner-loop')for(const n of [16,24,64,128])for(const batch of [4,16,32])for(const drop of [false,true]){
   const params={...defaults,n,batch,drop},per=drop?Math.floor(n/batch):Math.ceil(n/batch),total=per*defaults.epochs;
   for(let i=0;i<=total;i++){const m=render(def,params,total?i/total:0);assert.equal(m.completed,i);assert.equal(m.iterationsPerEpoch,per);assert.equal(m.totalIterations,total);if(i===total)assert.equal(m.batchSize,0);}
  }
 }
}
console.log(`Passed: ${scenes} scenes / ${states} render states; all controls, finite metrics, sample conservation, folds and batch counts.`);
