import {demo,range,select} from '../../shared/scene.js';
import * as M from '../../shared/math.js';
import * as G from '../../shared/geometry.js';
const C=G.colors;
const names=indices=>indices.map(j=>`X${j+1}`).join(' + ')||'無特徵（僅截距）';
// Two balanced, orthogonal blocks. Only the training block has a spurious X2 effect.
// All columns already have training mean 0 and standard deviation 1.
export const selectionData=Array.from({length:32},(_,i)=>{
 const x=[0,1,2,3].map(bit=>(i&(1<<bit))?1:-1);
 return {x,y:3*x[0]+1.5*x[2]+(i<16?.25*x[1]+.2*x[0]*x[3]:-.25*x[1]+.2*x[1]*x[3])};
});
const train=selectionData.slice(0,16),validation=selectionData.slice(16);
export function fitSelection(indices,alpha=0){
 const weights=[0,0,0,0],intercept=M.mean(train.map(p=>p.y));
 for(let iteration=0;iteration<100;iteration++){
  let change=0;
  for(const j of indices){
   const rho=M.mean(train.map(p=>p.x[j]*(p.y-intercept-M.sum(p.x.map((v,k)=>k===j?0:v*weights[k])))));
   const norm=M.mean(train.map(p=>p.x[j]**2));
   const next=Math.sign(rho)*Math.max(0,Math.abs(rho)-alpha)/norm;
   change=Math.max(change,Math.abs(next-weights[j]));weights[j]=next;
  }
  if(change<1e-10)break;
 }
 const error=data=>M.mean(data.map(p=>(p.y-intercept-M.sum(p.x.map((v,j)=>v*weights[j])))**2));
 const trainMSE=error(train),validationMSE=error(validation);
 return {weights,intercept,trainMSE,validationMSE,objective:trainMSE/2+alpha*M.sum(weights.map(Math.abs))};
}
export function selectionCandidates(keep){
 return Array.from({length:15},(_,i)=>[0,1,2,3].filter(j=>(i+1)&(1<<j))).filter(indices=>indices.length===keep).map(indices=>({indices,...fitSelection(indices)}));
}
const scores=[0,1,2,3].map(j=>Math.abs(M.mean(train.map(p=>p.x[j]*p.y)))/Math.sqrt(M.mean(train.map(p=>p.y**2))));
const cache=new Map();
const candidates=keep=>{if(!cache.has(keep))cache.set(keep,selectionCandidates(keep));return cache.get(keep);};
function chips(c,selected,y){
 [0,1,2,3].forEach(j=>{const active=selected?.includes(j);G.rect(c,100+j*150,y,110,34,active?C.mint:C.white,active?C.green:C.line);G.text(c,`X${j+1} ${selected===null?'待選定':active?'保留':'移除'}`,155+j*150,y+17,15,active?C.green:C.muted);});
}
function wrapper(c,s){
 const list=candidates(s.params.keep),total=list.length*3+1,pos=Math.min(total,s.progress*total),step=Math.floor(pos+1e-6);
 const done=Math.min(list.length,Math.floor(step/3)),finished=step===total,current=list[Math.min(done,list.length-1)],phase=finished?3:step%3;
 const best=list.slice(0,done).reduce((a,b)=>!a||b.validationMSE<a.validationMSE-1e-10?b:a,null);
 G.heading(c,'包裝法：每個候選組合都訓練，再比較驗證誤差');
 G.text(c,`固定資料：訓練 16 筆／驗證 16 筆 · 枚舉所有 ${s.params.keep} 欄組合`,380,64,14,C.muted);
 const labels=['① 選候選組合','② 訓練線性模型','③ 驗證集評分','④ 選出最佳'];
 labels.forEach((label,i)=>{G.rect(c,25+i*185,87,155,43,i===phase?C.mint:C.white,i===phase?C.green:C.line);G.text(c,label,102+i*185,109,14);if(i<3)G.arrow(c,182+i*185,109,204+i*185,109,C.line);});
 if(!finished){const segment=pos-step;if(phase<2)G.travelling(c,[182+phase*185,109],[204+phase*185,109],segment,C.orange,5);}
 G.text(c,'候選組合',55,156,14,C.muted,'left');G.text(c,'驗證 MSE ↓',339,156,14,C.muted);
 list.forEach((model,i)=>{
  const y=183+i*30,evaluated=i<done,active=!finished&&i===done;
  G.rect(c,30,y-13,365,26,active?C.mint:C.white,best===model?C.green:C.line,4);
  G.text(c,names(model.indices),45,y,14,C.ink,'left');
  G.text(c,evaluated?M.fmt(model.validationMSE,3):active?['待訓練','訓練中','驗證中'][phase]:'待評估',365,y,14,evaluated?C.green:C.muted,'right');
 });
 G.rect(c,430,148,300,211,C.white,C.line);
 G.text(c,finished?'最後選出的模型':'目前候選模型',580,171,17,C.green);
 const shown=finished?best:current;
 G.text(c,names(shown.indices),580,202,17);
 [0,1,2,3].forEach(j=>G.text(c,`X${j+1}：${shown.indices.includes(j)?(finished||phase>=1?M.fmt(shown.weights[j],2):'等待訓練'):'未放入模型'}`,450,233+j*25,14,C.muted,'left'));
 G.text(c,`已評估 ${done} / ${list.length} 組`,220,385,16,C.green);
 G.text(c,best?`目前最佳 MSE：${M.fmt(best.validationMSE,3)}`:'評分只使用驗證集',580,385,15,C.green);
 chips(c,finished?best.indices:null,402);
 const selected=finished?best.indices:[];
 return {message:`包裝法：${finished?`保留 ${names(selected)}，驗證 MSE = ${M.fmt(best.validationMSE,3)}。`:`正在${labels[phase].slice(2)}：${names(current.indices)}；已評估 ${done}/${list.length} 組。`}每組只用訓練集擬合，再用同一驗證集選擇；驗證分數不是獨立測試成績。`,metrics:{method:'wrapper',evaluated:done,candidateCount:list.length,phase,selected,best:best?{indices:best.indices,validationMSE:best.validationMSE}:null}};
}
const paths=new Map();
function embedded(c,s){
 const alpha=s.params.alpha*s.progress,model=fitSelection([0,1,2,3],alpha),selected=model.weights.flatMap((w,j)=>Math.abs(w)>1e-8?[j]:[]);
 G.heading(c,'嵌入法：Lasso 在訓練目標中加入 L1 懲罰');
 G.text(c,'目標 = 訓練 MSE / 2 + α × Σ|係數| · 特徵已標準化',380,64,15,C.muted);
 G.text(c,`α：0 → ${M.fmt(s.params.alpha,1)}　目前 ${M.fmt(alpha,2)}`,380,96,18,C.green);
 const a=G.axes(c,{xmin:0,xmax:Math.max(.1,s.params.alpha),ymin:0,ymax:3.2,x:65,y:153,w:345,h:173,xlabel:'L1 強度 α（不是訓練時間）',ylabel:'係數值 β'});
 if(!paths.has(s.params.alpha))paths.set(s.params.alpha,Array.from({length:141},(_,i)=>{const alpha=s.params.alpha*i/140;return {alpha,weights:fitSelection([0,1,2,3],alpha).weights};}));
 const path=paths.get(s.params.alpha);
 [0,1,2,3].forEach(j=>{const color=G.palette[j];let previous=path[0];for(const point of path.slice(1)){if(point.alpha>alpha)break;G.line(c,a.X(previous.alpha),a.Y(previous.weights[j]),a.X(point.alpha),a.Y(point.weights[j]),color,3);previous=point;}G.line(c,a.X(previous.alpha),a.Y(previous.weights[j]),a.X(alpha),a.Y(model.weights[j]),color,3);G.marker(c,a.X(alpha),a.Y(model.weights[j]),j,5);});
 G.text(c,'目前係數（0 表示不使用此欄）',586,139,15,C.green);
 [0,1,2,3].forEach(j=>{const y=179+j*45,w=model.weights[j];G.text(c,`X${j+1}`,466,y,16,G.palette[j]);G.rect(c,491,y-10,161,20,C.mint,null,3);if(w>0)G.rect(c,491,y-10,w/3*161,20,G.palette[j],null,3);G.text(c,M.fmt(w,2),704,y,16,w>1e-8?C.ink:C.muted);});
 G.legend(c,[0,1,2,3].map(j=>[`X${j+1}`,G.palette[j]]),100,384,93);
 G.text(c,`訓練目標值：${M.fmt(model.objective,3)}`,581,384,15,C.green);
 chips(c,selected,402);
 return {message:`Lasso 正則化路徑：每個 α 都重新擬合訓練集。目前 α = ${M.fmt(alpha,2)}，保留 ${names(selected)}（${selected.length} 欄）；係數為 0 的欄位被移除。調整「L1 強度」決定路徑終點；「保留特徵數」不控制嵌入法。`,metrics:{method:'embedded',alpha,selected,...model}};
}
function filter(c,s){
 const selected=scores.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,s.params.keep).map(o=>o.i);
 G.heading(c,'過濾法：先算各欄與目標 y 的相關係數');
 G.text(c,'固定 32 筆示範資料 · 只用訓練 16 筆算分 · 表內預覽前 6 筆',380,64,14,C.muted);
 [0,1,2,3].forEach(j=>{
  c.globalAlpha=selected.includes(j)?1:M.lerp(1,.18,s.progress);
  G.text(c,`X${j+1}`,140+j*160,100,20);
  train.slice(0,6).forEach((p,i)=>G.cell(c,140+j*160,139+i*31,p.x[j],C.mint,95,25));
  G.text(c,`|r| = ${M.fmt(scores[j],2)}`,140+j*160,341,17,C.green);
 });c.globalAlpha=1;
 G.text(c,'獨立評分 → 排序 → 取前 k 欄；不需要訓練模型',380,381,16,C.green);
 chips(c,selected,402);
 return {message:`過濾法：由訓練資料實算 |Pearson r|，保留 ${names(selected)}。它獨立評分各欄；包裝法比較組合的驗證誤差，嵌入法則在模型訓練中產生稀疏係數。`,metrics:{method:'filter',scores,selected}};
}
export const selection=demo('selection','特徵選擇',[
 select('method','選取方法',[['filter','過濾：相關係數'],['wrapper','包裝：組合訓練與驗證'],['embedded','嵌入：Lasso 正則化路徑']]),
 range('keep','過濾／包裝：保留特徵數',1,4,2),
 range('alpha','嵌入：L1 強度 α',0,3.5,.6,.1)
],(c,s)=>s.params.method==='wrapper'?wrapper(c,s):s.params.method==='embedded'?embedded(c,s):filter(c,s),
'固定且已標準化的示範資料；三種選取方法的分數與模型係數各自實算。',{
 steps:s=>s.params.method==='wrapper'?candidates(s.params.keep).length*3+1:12,
 guide:['過濾法：看各欄 |r| 分數；「保留特徵數」決定取前幾欄。','包裝法：每三步完成一組的選取、訓練與驗證；所有候選評估後選最佳。','嵌入法：播放 α 從 0 增加的路徑，看係數線與長條縮到 0。','嵌入法保留數由非零係數決定；調整 L1 強度，對照移除的欄位。']
});
