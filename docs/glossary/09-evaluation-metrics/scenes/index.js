import {demo,range,select,check} from '../../shared/scene.js';import * as M from '../../shared/math.js';import * as G from '../../shared/geometry.js';const C=G.colors;
function confusion(c,s,data){
 const counts=M.metrics(data,s.params.threshold),names=['TP','FP','FN','TN'],groups=[data.filter(p=>p.y&&p.score>=s.params.threshold),data.filter(p=>!p.y&&p.score>=s.params.threshold),data.filter(p=>p.y&&p.score<s.params.threshold),data.filter(p=>!p.y&&p.score<s.params.threshold)],positions=new Map();
 G.text(c,'原始樣本 · 30 筆',103,88,15,C.green);
 G.text(c,'真實正類',385,82,15,C.orange);G.text(c,'真實負類',575,82,15,C.green);
 G.text(c,'預測正類',259,160,14,C.ink);G.text(c,'預測負類',259,310,14,C.ink);
 groups.forEach((group,i)=>{
  const x=385+(i%2)*190,y=145+Math.floor(i/2)*150;
  G.rect(c,x-75,y-35,150,120,i===0||i===3?C.mint:'#f6e6d3',C.line);
  const arrived=group.filter(p=>s.progress*data.length-p.id>=1-1e-6).length;G.text(c,`${names[i]}：${arrived}/${group.length}`,x,y-18,17,i===0||i===3?C.green:C.orange);
  group.forEach((p,j)=>positions.set(p.id,[x-54+(j%7)*18,y+5+Math.floor(j/7)*17]));
 });
 data.forEach((p,i)=>{
  const from=[43+(i%5)*26,130+Math.floor(i/5)*26],to=positions.get(p.id),t=M.clamp(s.progress*data.length-i);
  G.rect(c,from[0]-9,from[1]-9,18,18,C.white,C.line,2);G.text(c,p.id+1,...from,9,C.muted);
  const x=M.lerp(from[0],to[0],t),y=M.lerp(from[1],to[1],t);
  G.rect(c,x-7,y-7,14,14,p.y?C.orange:C.green,null,2);G.text(c,p.id+1,x,y,9,C.white);
 });
 G.legend(c,[['真實負類',C.green],['真實正類',C.orange]],35,327,115);
 G.text(c,'格內數字：已移入／總筆數；方塊數字：樣本 ID',380,420,15,C.green);
 return counts;
}
const errorData=[{x:0,y:1},{x:1,y:2},{x:2,y:2.5},{x:3,y:4},{x:4,y:4.2}];
export const definitions=[
demo('confusion-matrix','混淆矩陣',[range('threshold','判為正類的門檻',0,1,.5,.05)],(c,s)=>{const data=M.scored(30,.3),out=confusion(c,s,data);G.heading(c,'P/N 是預測類別，T/F 是猜對與否');return {message:`最終分格：TP=${out.tp}, FP=${out.fp}, FN=${out.fn}, TN=${out.tn}，總數 30。預測 score ≥ ${s.params.threshold} 為正類；每個方塊保留原本的真實標籤。`,metrics:out};},undefined,{guide:['左邊 30 格是樣本（顏色代表真實類別），右邊是 TP／FP／FN／TN 四格。','拖動門檻滑桿：看方塊重新落格，數量同步更新。','按「下一步」或拖時間軸：看方塊從左邊移入四格。','下方訊息顯示四格數量，總數恆為 30。']}),
demo('classification-metrics','分類指標與 ROC',[range('threshold','分類門檻（分數 ≥ 門檻判正類）',0,1,.5,.05),range('positive','正類比例（%）：重抽 40 筆樣本',10,90,30,10)],(c,s)=>{const data=M.scored(40,s.params.positive/100),out=M.metrics(data,s.params.threshold),roc=M.roc(data),a=G.axes(c,{xmin:0,xmax:1,ymin:0,ymax:1,x:65,y:90,w:325,h:280,xlabel:'FPR',ylabel:'TPR / Recall'});G.line(c,a.X(0),a.Y(0),a.X(1),a.Y(1),C.line,1,[5,4]);const n=Math.max(1,Math.floor(s.progress*roc.curve.length));roc.curve.slice(0,n).forEach((p,i,arr)=>{if(i)G.line(c,a.X(arr[i-1].fpr),a.Y(arr[i-1].recall),a.X(p.fpr),a.Y(p.recall),C.green,3);});G.diamond(c,a.X(out.fpr),a.Y(out.recall),8,C.orange);[['Accuracy',out.accuracy],['Precision',out.precision],['Recall',out.recall],['F1',out.f1]].forEach(([label,value],i)=>{G.rect(c,455,100+i*68,250,54,C.white,C.line);G.text(c,label,475,127+i*68,16,C.muted,'left');G.text(c,M.fmt(value),675,127+i*68,21,C.green,'right');});G.heading(c,'門檻改變，四象限與指標一起變');return {message:`TP ${out.tp} / FP ${out.fp} / FN ${out.fn} / TN ${out.tn}；AUC=${M.fmt(roc.auc)}。AUC 描述此分數的排序能力；它可小於 0.5，且類別不平衡時仍需看 PR 指標與情境。分母為零顯示未定義。`,metrics:{...out,auc:roc.auc}};},undefined,{guide:['分類門檻：模型輸出「像正類的分數」，≥ 門檻才判正類；是「判多嚴」的開關。','正類比例：40 筆樣本中真實正類的占比；改變會重抽整批數據，ROC 形狀與 AUC 跟著變。','左圖是 ROC 曲線（綠）與目前門檻點（橘菱形），右邊是指標卡。','拖動門檻：橘點沿曲線走，指標此消彼長，但 AUC 不動（門檻無關）。','分母為零時顯示「未定義」；AUC 可小於 0.5。']}),
demo('regression-errors','MAE、MSE、RMSE',[range('slope','預測斜率',0,2,.8,.1),check('outlier','加入離群值',false)],(c,s)=>{
 const data=s.params.outlier?[...errorData,{x:4.5,y:8}]:errorData,a=G.axes(c,{xmin:0,xmax:5,ymin:0,ymax:9,x:55,w:390,h:280,xlabel:'特徵 X₁',ylabel:'標籤 y'}),fn=x=>s.params.slope*x+.8,out=M.regressionMetrics(data,fn);
 const errors=data.map(p=>Math.abs(p.y-fn(p.x))),scale=65/Math.max(...errors,1);
 G.dots(c,a,data);G.curve(c,a,fn);
 G.text(c,'平方誤差：面積 ∝ 誤差²',606,84,16,C.purple);G.text(c,`共用尺：1 單位誤差 = ${M.fmt(scale,1)} px`,606,108,13,C.muted);
 data.forEach((p,i)=>{
  const e=errors[i],x=505+(i%3)*78,y=150+Math.floor(i/3)*125,side=e*scale*s.progress;
  G.line(c,a.X(p.x),a.Y(p.y),a.X(p.x),a.Y(fn(p.x)),C.purple,2);
  G.text(c,`樣本 ${i+1}`,x+30,y-17,13,C.muted);G.rect(c,x,y,side,side,'#ebdff3',C.purple,0);
  G.text(c,`|e|=${M.fmt(e,2)}`,x+30,y+78,12,C.ink);G.text(c,`e²=${M.fmt(e*e,2)}`,x+30,y+96,12,C.purple);
 });
 G.heading(c,'絕對誤差是一段長度，平方誤差是一塊面積');
 G.legend(c,[['真實樣本',C.green],['預測線',C.orange],['殘差／平方誤差',C.purple]],80,425,190);
 return {message:`MAE=${M.fmt(out.mae)}（原單位），MSE=${M.fmt(out.mse)}（平方單位），RMSE=${M.fmt(out.rmse)}（原單位）。本畫面每個平方塊共用相同倍率 ${M.fmt(scale,1)} px／誤差單位；更換設定時倍率會重新計算，請一起比較實際誤差數字。`,metrics:{...out,squareScale:scale,absoluteErrors:errors}};
},undefined,{guide:['綠圓是真實資料，橘線是預測線，紫線是殘差。','右側每筆都有 |e| 與 e²；平方塊面積按相同倍率呈現。','拖動斜率或勾選離群值，對照 MAE／MSE／RMSE 與各筆誤差。','倍率標在圖上；更換設定時會調整畫布倍率，避免色塊重疊或小到看不見。']}),
demo('r-squared','R²',[select('model','比較預測模型',[['good','接近資料的模型'],['mean','只猜平均值'],['bad','很差的預測'],['constant','常數標籤（邊界情況）']])],(c,s)=>{const data=s.params.model==='constant'?errorData.map(p=>({...p,y:3})):errorData,mu=M.mean(data.map(p=>p.y)),fn=s.params.model==='mean'?()=>mu:s.params.model==='bad'?()=>8:s.params.model==='constant'?()=>2:x=>.8*x+1,stats=M.regressionMetrics(data,fn),a=G.axes(c,{xmin:0,xmax:5,ymin:0,ymax:9,x:55,w:390,h:280,xlabel:'特徵 X₁',ylabel:'標籤 y'});G.dots(c,a,data);G.curve(c,a,()=>mu,C.green,2);G.curve(c,a,fn,C.orange,3);data.forEach(p=>{G.line(c,a.X(p.x),a.Y(p.y),a.X(p.x),a.Y(fn(p.x)),C.purple,2);});const max=Math.max(stats.sse,stats.sst,1);G.rect(c,510,150,170*s.progress*stats.sst/max,45,C.mint,null,0);G.rect(c,510,250,170*s.progress*stats.sse/max,45,'#ebdff3',null,0);G.text(c,`SST = ${M.fmt(stats.sst,2)}`,590,125,16,C.green);G.text(c,`SSE = ${M.fmt(stats.sse,2)}`,590,225,16,C.purple);G.heading(c,'R² = 1 − SSE / SST','色條逐步展開到總平方和；同一畫面共用比例尺');G.legend(c,[['真實樣本／平均基準',C.green],['比較模型',C.orange],['模型殘差',C.purple]],75,398,245);return {message:`R²=${M.fmt(stats.r2)}。${stats.sst===0?'y 是常數，SST=0，此處顯示未定義。':stats.r2<0?'比直接猜本批標籤平均更差；負分不只可能來自模型架構，也可能是分布或資料問題。':stats.r2===0?'SSE 等於 SST，與平均值基準相同。':'對照相同資料上的平均值基準。'}`,metrics:stats};},undefined,{guide:['綠線是平均值基準，橘線是比較模型，紫線是殘差。','切換四種模型：比較綠條 SST 與紫條 SSE 的長度。','R²=1−SSE/SST，試「很差的預測」看負值。','常數標籤時 SST=0，R² 顯示未定義。']})];
