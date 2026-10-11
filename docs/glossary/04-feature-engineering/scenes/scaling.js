import {demo,select,check} from '../../shared/scene.js';
import * as M from '../../shared/math.js';
import * as G from '../../shared/geometry.js';
const C=G.colors;
const training=[[150,1],[350,2],[610,6],[800,3],[900,4],[1200,5]];
const houses=[[600,3],[610,6],[800,3]];
const labels=['Q：要找鄰居','A：面積接近','B：房間數相同'];
const methods={raw:'不縮放',z:'Z-score',minmax:'Min-Max'};
export function scalingExample({method='z',unit='m2',outlier=false}={}){
 const factor=unit==='kilo'?.001:1;
 const data=(outlier?[...training,[5000,3]]:training).map(row=>[row[0]*factor,row[1]]);
 const raw=houses.map(row=>[row[0]*factor,row[1]]);
 const stats=[0,1].map(j=>{const values=data.map(row=>row[j]),mean=M.mean(values);return {mean,std:Math.sqrt(M.mean(values.map(x=>(x-mean)**2))),min:Math.min(...values),max:Math.max(...values)};});
 const transform=row=>row.map((v,j)=>method==='z'?(v-stats[j].mean)/stats[j].std:method==='minmax'?(v-stats[j].min)/(stats[j].max-stats[j].min):v);
 const scaled=raw.map(transform);
 const distances=rows=>rows.slice(1).map((row,i)=>{const parts=row.map((v,j)=>(v-rows[0][j])**2),squared=M.sum(parts);return {id:['A','B'][i],parts,squared,distance:Math.sqrt(squared)};});
 const before=distances(raw),after=distances(scaled);
 const nearest=values=>values.reduce((a,b)=>a.distance<=b.distance?a:b).id;
 return {raw,scaled,stats,before,after,beforeNearest:nearest(before),afterNearest:nearest(after),trainingCount:data.length,method,unit};
}
function comparison(c,x,title,values,{revealed=true,amount=1,winner=true,coordinates=null}={}){
 G.rect(c,x,173,340,203,C.white,C.line);
 G.text(c,title,x+170,193,18,C.green);
 const max=Math.max(...values.map(p=>p.squared));
 values.forEach((p,i)=>{
  const y=225+i*70;
  G.text(c,`${p.id} → Q`,x+15,y,15,C.ink,'left');
  G.text(c,revealed?`d = ${M.fmt(p.distance,3)}`:'等待計算距離',x+325,y,15,C.muted,'right');
  G.rect(c,x+15,y+14,310,15,C.paper,C.line,2);
  if(revealed){
   // Stacked segment lengths are squared differences, never Euclidean lengths.
   let cursor=x+15;
   p.parts.forEach((value,j)=>{const width=310*value/max*amount;if(width>0)G.rect(c,cursor,y+14,width,15,j===0?C.orange:C.purple,null,0);cursor+=width;});
  }
  G.text(c,revealed?`d² = ${M.fmt(p.parts[0],3)} + ${M.fmt(p.parts[1],3)}`:coordinates?`${p.id} = (${coordinates[i+1].map(v=>M.fmt(v,3)).join(', ')})`:'用相同訓練統計轉換 Q、A、B',x+15,y+43,13,C.muted,'left');
 });
 const nearest=values.reduce((a,b)=>a.distance<=b.distance?a:b).id;
 G.text(c,winner?`兩候選中較近：${nearest}`:'播放後比較兩個候選',x+170,359,16,winner?C.green:C.muted);
}
export const scaling=demo('scaling','特徵縮放',[
 select('method','縮放方式',[['z','Z-score 標準化'],['minmax','Min-Max'],['raw','不縮放（對照）']]),
 select('unit','面積單位（房屋不變）',[['m2','平方公尺 m²'],['kilo','千平方公尺（數值 ÷ 1000）']]),
 check('outlier','訓練資料加入面積 5000 m² 的離群值',false)
],(c,s)=>{
 const m=scalingExample(s.params),step=Math.min(4,Math.floor(s.progress*4+1e-6)),unit=s.params.unit==='kilo'?'千 m²':'m²';
 G.heading(c,'為什麼要縮放？數值大，不代表特徵更重要');
 G.text(c,'找 Q 的近鄰：只比較 A、B；距離同時使用面積與房間數',380,62,15,C.muted);
 m.raw.forEach((row,i)=>{
  const x=25+i*250;
  G.rect(c,x,82,230,58,i===0?C.mint:C.white,i===0?C.green:C.line);
  G.text(c,labels[i],x+115,98,16,i===0?C.green:C.ink);
  G.text(c,`${M.fmt(row[0],s.params.unit==='kilo'?3:0)} ${unit}／${row[1]} 房`,x+115,122,15,C.muted);
 });
 G.legend(c,[['面積差²',C.orange],['房間數差²',C.purple]],245,155,170);
 comparison(c,25,'未縮放：直接混用原始數值',m.before);
 comparison(c,395,`${methods[s.params.method]}：逐欄轉換後`,m.after,{coordinates:step>=2?m.scaled:null,revealed:s.progress>.5,amount:M.clamp(s.progress*4-2),winner:step>=4});
 const stages=['① fit 訓練統計','② 轉換兩個特徵','③ 計算距離','④ 比較近鄰'];
 stages.forEach((label,i)=>G.text(c,label,113+i*180,396,14,step===i+1?C.green:C.muted));
 const st=m.stats;
 const footer=step===0?'按播放：觀察縮放後，較近的候選是否改變。':step===1?(s.params.method==='z'?`訓練 σ：面積 ${M.fmt(st[0].std,3)}、房間 ${M.fmt(st[1].std,3)}；z = (x − μ) / σ`:s.params.method==='minmax'?`訓練範圍：面積 ${M.fmt(st[0].max-st[0].min,3)}、房間 ${M.fmt(st[1].max-st[1].min,3)}；(x − min) / 範圍`:'不轉換；原始單位繼續影響距離。'):step===2?`轉換後 Q = (${m.scaled[0].map(v=>M.fmt(v,3)).join(', ')})；每欄使用自己的統計量。`:step===3?'長條表示 d² = 面積差² + 房間數差²；同圖共用尺度，左右各自定尺度。':s.params.method==='raw'?'試著切換面積單位：房屋不變，原始距離排名卻會變。':'縮放讓兩個特徵共同影響距離；換面積單位後，縮放結果仍相同。';
 G.text(c,footer,380,426,14,C.green);
 return {message:`${stages[Math.max(0,step-1)]}。${m.trainingCount} 筆訓練資料 fit，Q 不參與 fit。未縮放較近：${m.beforeNearest}；${step>=4?`${methods[s.params.method]} 較近：${m.afterNearest}。`:'播放完成後比較結果。'}橘／紫長條表示兩欄的平方差，顯示數值範圍如何主導距離。${s.params.outlier?'離群值也會改變兩種縮放的統計與距離。':''}縮放不是保證預測更準；KNN、K-means 等依賴距離的模型常需要它，決策樹通常不需要。`,metrics:{...m,step}};
},'Canvas 2D 距離示範；數值由固定房屋資料實算。長條各圖內可比，左右不同尺度不直接比較長度。',{
 steps:4,
 guide:['Q 是查詢房屋，A 的面積接近、B 的房間數相同；比較誰離 Q 較近。','看橘色面積差²與紫色房間差²：未縮放時，面積數值容易主導距離。','播放四步：fit 訓練統計 → 轉換兩欄 → 計算距離 → 比較近鄰。','切換面積單位：房屋沒變；原始排名會變，Z-score／Min-Max 的結果不變。','加入離群值，看訓練統計與距離改變；縮放不等於消除離群值。']
});
