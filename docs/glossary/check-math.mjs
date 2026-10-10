import assert from 'node:assert/strict';
import * as M from './shared/math.js';

const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);
const data=Array.from({length:40},(_,id)=>({id,label:id%5===0?1:0}));
for(const mode of ['random','stratified','time']){
  const {train,test}=M.split(data,.7,mode);
  assert.equal(new Set([...train,...test].map(p=>p.id)).size,40);
  assert.ok(!train.some(p=>test.includes(p)));
  if(mode==='time')assert.ok(train.at(-1).id<test[0].id);
  if(mode==='stratified')assert.equal(train.filter(p=>p.label===1).length,6);
}
const line=[-1,0,1,2].map(x=>({x,y:2*x+3}));
close(M.linear(line).w,2);close(M.linear(line).b,3);
const quadratic=M.samples(42,20,0),weights=M.polynomial(quadratic,2,0);
close(M.mse(quadratic,x=>M.predict(weights,x)),0);
assert.ok(M.descent(.15,3,20).at(-1)**2<.001);
assert.ok(Math.abs(M.descent(1.1,3,10).at(-1))>3);
const scored=[{y:1,score:.9},{y:0,score:.8},{y:1,score:.4},{y:0,score:.1}];
const confusion=M.metrics(scored,.5);
assert.deepEqual([confusion.tp,confusion.fp,confusion.fn,confusion.tn],[1,1,1,1]);
close(confusion.precision,.5);close(confusion.recall,.5);close(confusion.f1,.5);
close(M.roc(scored).auc,.75);
assert.equal(M.metrics(scored,1).precision,null);
assert.ok(M.regressionMetrics(line,()=>100).r2<0);
assert.equal(M.regressionMetrics([{x:0,y:3},{x:1,y:3}],()=>3).r2,null);
const nearest=M.knn([{x:0,y:0,label:0},{x:1,y:0,label:1},{x:1,y:1,label:1}],{x:.9,y:.5},3);
assert.equal(nearest.label,1);assert.deepEqual(nearest.votes,[1,2,0]);
const separable=[{x:-1,y:0,label:0},{x:-2,y:1,label:0},{x:1,y:0,label:1},{x:2,y:1,label:1}];
const tree=M.tree(separable,2);
assert.ok(separable.every(p=>M.treePredict(tree,p)===p.label));
const clusters=M.kmeans(separable,2,10);
assert.ok(clusters.centers.every((center,i)=>{
  const assigned=clusters.assigned.filter(p=>p.group===i);
  return Math.abs(center.x-M.mean(assigned.map(p=>p.x)))<1e-6;
}));
console.log('Passed: splits, regression, gradient updates, classification metrics, ROC, R², KNN, tree and clustering.');
