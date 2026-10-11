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
