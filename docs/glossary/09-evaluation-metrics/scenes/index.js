import {demo,range,select,check} from '../../shared/scene.js';import * as M from '../../shared/math.js';import * as G from '../../shared/geometry.js';const C=G.colors;
function confusion(c,s,data){
 const counts=M.metrics(data,s.params.threshold);
 const names=['TP','FN','FP','TN'];
 const groups=[
  data.filter(p=>p.y&&p.score>=s.params.threshold),  // TP: 真實正、預測正 (左上)
  data.filter(p=>p.y&&p.score<s.params.threshold),   // FN: 真實正、預測負 (右上)
  data.filter(p=>!p.y&&p.score>=s.params.threshold), // FP: 真實負、預測正 (左下)
  data.filter(p=>!p.y&&p.score<s.params.threshold)   // TN: 真實負、預測負 (右下)
 ];
 const positions=new Map();
 G.text(c,'原始樣本 · 30 筆',103,88,15,C.green);
 // 橫軸（上方）：模型預測
 G.text(c,'模型預測 (Predicted)',480,62,12,C.muted);
 G.text(c,'預測正類（病例）',385,84,14,C.ink);
 G.text(c,'預測負類（健康）',575,84,14,C.ink);
 // 縱軸（左側）：真實情況
 G.text(c,'真實情況 (Actual)',235,62,12,C.muted);
 G.text(c,'真實正類（病例）',235,145,13,C.orange);
 G.text(c,'真實負類（健康）',235,295,13,C.green);
 const fills=[C.mint,'#fde8e8','#fef3c7',C.mint];
 const strokes=[C.line,'#fca5a5','#fde68a',C.line];
 const textColors=[C.green,'#c53030','#c47d35',C.green];
 groups.forEach((group,i)=>{
  const x=385+(i%2)*190,y=145+Math.floor(i/2)*150;
  G.rect(c,x-75,y-35,150,120,fills[i],strokes[i]);
  const arrived=group.filter(p=>s.progress*data.length-p.id>=1-1e-6).length;
  G.text(c,`${names[i]}：${arrived}/${group.length}`,x,y-18,17,textColors[i]);
  group.forEach((p,j)=>positions.set(p.id,[x-54+(j%7)*18,y+5+Math.floor(j/7)*17]));
 });
 data.forEach((p,i)=>{
  const from=[43+(i%5)*26,130+Math.floor(i/5)*26],to=positions.get(p.id),t=M.clamp(s.progress*data.length-i);
  G.rect(c,from[0]-9,from[1]-9,18,18,C.white,C.line,2);G.text(c,p.id+1,...from,9,C.muted);
  const x=M.lerp(from[0],to[0],t),y=M.lerp(from[1],to[1],t);
  G.rect(c,x-7,y-7,14,14,p.y?C.orange:C.green,null,2);G.text(c,p.id+1,x,y,9,C.white);
 });
 G.legend(c,[['真實負類（健康）',C.green],['真實正類（病例）',C.orange]],10,327,135);
 G.text(c,'格內數字：已移入／總筆數；方塊數字：樣本 ID',380,420,15,C.green);
 return counts;
}
const errorData=[{x:0,y:1},{x:1,y:2},{x:2,y:2.5},{x:3,y:4},{x:4,y:4.2}];
export const definitions=[
demo('confusion-matrix','混淆矩陣',[range('threshold','判為正類的門檻',0,1,.5,.05)],(c,s)=>{const data=M.scored(30,.3),out=confusion(c,s,data);G.heading(c,'正例＝病例，負例＝健康 ｜ P/N 為預測、T/F 為對錯');return {message:`最終分格：TP=${out.tp}, FP=${out.fp}, FN=${out.fn}, TN=${out.tn}，總數 30。設定「正例=病例、負例=健康」；預測 score ≥ ${s.params.threshold} 判為正類。每個方塊保留原本真實標籤。`,metrics:out};},undefined,{guide:['左邊 30 格是樣本（橘色=真實病例，綠色=真實健康），右邊是混淆矩陣四象限。','四象限結構（對齊教材圖解）：橫軸為預測（左正／右負），縱軸為真實（上正／下負）；左上 TP、右上 FN、左下 FP、右下 TN。','情境設定：正例 (Positive) = 病例、負例 (Negative) = 健康。','拖動門檻滑桿：看方塊重新落格，數量同步更新。','按「下一步」或拖時間軸：看方塊從左邊移入四格。','下方訊息顯示四格數量，總數恆為 30。']}),
demo('classification-metrics','分類指標與 ROC',[select('view','指標觀察視角與切片',[
  ['overview','總覽：混淆矩陣切片與指標'],
  ['precision','縱看 Precision（預測正類欄 TP/(TP+FP)）'],
  ['recall','橫看 Recall（真實正類列 TP/(TP+FN)）'],
  ['accuracy','斜看 Accuracy（對角猜對 (TP+TN)/全部）'],
  ['f1','天平 F1-Score（調和平均平衡）']
]),range('threshold','分類門檻（分數 ≥ 門檻判正類）',0,1,.5,.05),range('positive','正類比例（%）：重抽 40 筆樣本',10,90,30,10)],(c,s)=>{
  const data=M.scored(40,s.params.positive/100),out=M.metrics(data,s.params.threshold),roc=M.roc(data);
  const a=G.axes(c,{xmin:0,xmax:1,ymin:0,ymax:1,x:50,y:80,w:285,h:265,xlabel:'FPR（代價：誤報率）',ylabel:'TPR / Recall（收益：抓全率）'});
  G.line(c,a.X(0),a.Y(0),a.X(1),a.Y(1),C.line,1,[5,4]);
  const n=Math.max(1,Math.floor(s.progress*roc.curve.length));
  roc.curve.slice(0,n).forEach((p,i,arr)=>{if(i)G.line(c,a.X(arr[i-1].fpr),a.Y(arr[i-1].recall),a.X(p.fpr),a.Y(p.recall),C.green,3);});
  G.diamond(c,a.X(out.fpr),a.Y(out.recall),8,C.orange);
  G.text(c,`AUC = ${M.fmt(roc.auc,2)}`,a.X(.72),a.Y(.15),13,C.green);

  const view=s.params.view||'overview';
  const mx=375,my=44;
  G.text(c,'混淆矩陣 2×2 切片視角',mx+110,my,13,C.muted);
  G.text(c,'預測正(+)',mx+55,my+18,11,C.ink);
  G.text(c,'預測負(−)',mx+160,my+18,11,C.ink);
  G.text(c,'真實正(+)',mx-32,my+42,10,C.orange);
  G.text(c,'真實負(−)',mx-32,my+76,10,C.green);

  const isP=view==='precision',isR=view==='recall',isA=view==='accuracy',isF=view==='f1';
  const tpFill=(isP||isR||isA||isF)?'#d1fae5':C.mint;
  const fnFill=isR?'#fee2e2':(isP||isA)?'#f8fafc':'#fde8e8';
  const fpFill=isP?'#fef3c7':(isR||isA)?'#f8fafc':'#fef3c7';
  const tnFill=isA?'#e2e8f0':(isP||isR)?'#f8fafc':C.mint;

  // 矩陣 4 格：高度 28，完全收斂在 y=72 ~ 134
  G.rect(c,mx+5,my+28,100,28,tpFill,isA||isP||isR?C.green:C.line,5);
  G.text(c,`TP: ${out.tp} (確診)`,mx+55,my+42,12,C.green);

  G.rect(c,mx+110,my+28,100,28,fnFill,isR?'#dc2626':C.line,5);
  G.text(c,`FN: ${out.fn} (漏報)`,mx+160,my+42,12,isR?'#dc2626':'#c53030');

  G.rect(c,mx+5,my+62,100,28,fpFill,isP?'#d97706':C.line,5);
  G.text(c,`FP: ${out.fp} (誤報)`,mx+55,my+76,12,isP?'#d97706':'#c47d35');

  G.rect(c,mx+110,my+62,100,28,tnFill,isA?C.green:C.line,5);
  G.text(c,`TN: ${out.tn} (平安)`,mx+160,my+76,12,C.green);

  // 探照燈光罩與提示（y = 150 前全部結束）
  if(isP){
    G.rect(c,mx+2,my+25,106,68,'rgba(37,99,235,0.08)','#2563eb',6);
    G.text(c,'↓ 縱看預測正欄',mx+55,my+104,11,'#2563eb');
  }else if(isR){
    G.rect(c,mx+2,my+25,211,34,'rgba(217,119,6,0.08)','#d97706',6);
    G.text(c,'→ 橫看真實正列',mx+265,my+42,11,'#d97706');
  }else if(isA){
    G.line(c,mx+15,my+32,mx+195,my+84,'#16a34a',2,[4,3]);
    G.text(c,'⤨ 斜看對角總和 (猜對比例)',mx+110,my+104,11,'#16a34a');
  }else if(isF){
    G.text(c,'⚖️ 調和天平：平衡 Precision 與 Recall',mx+110,my+104,11,C.purple);
  }else{
    G.text(c,'四象限人數隨門檻即時更新',mx+110,my+104,11,C.muted);
  }

  // 四大指標卡片從 y = 168 開始，徹底杜絕任何重疊！
  const cards=[
    {key:'precision',name:'Precision 精確率',formula:`TP/(TP+FP) = ${out.tp}/${out.tp+out.fp}`,val:out.precision,color:'#2563eb'},
    {key:'recall',name:'Recall 召回率',formula:`TP/(TP+FN) = ${out.tp}/${out.tp+out.fn}`,val:out.recall,color:'#d97706'},
    {key:'accuracy',name:'Accuracy 準確率',formula:`(TP+TN)/40 = ${out.tp+out.tn}/40`,val:out.accuracy,color:'#16a34a'},
    {key:'f1',name:'F1-Score',formula:'2PR / (P+R)',val:out.f1,color:C.purple}
  ];

  cards.forEach((card,i)=>{
    const cy=168+i*52;
    const active=view===card.key;
    G.rect(c,mx-8,cy,342,46,active?'#f0fdf4':C.white,active?card.color:C.line,6);
    G.text(c,card.name,mx+8,cy+15,13.5,active?card.color:C.ink,'left');
    G.text(c,card.formula,mx+8,cy+32,10.5,C.muted,'left');
    G.text(c,M.fmt(card.val),mx+320,cy+16,16,active?card.color:C.green,'right');
    if(typeof card.val==='number'){
      G.rect(c,mx+225,cy+30,95,5,'#e2e8f0',null,2);
      G.rect(c,mx+225,cy+30,Math.max(0,Math.min(95,card.val*95)),5,card.color,null,2);
    }
  });

  G.heading(c,'分類指標體系：觀察視角與動態切片');

  let msg=`TP ${out.tp} / FP ${out.fp} / FN ${out.fn} / TN ${out.tn}；AUC=${M.fmt(roc.auc)}。`;
  if(view==='precision'){
    msg+=`【Precision 縱向視角】= ${M.fmt(out.precision)}。在所有模型判定患病的 ${out.tp+out.fp} 人中，抓對 ${out.tp} 人（冤枉誤報 ${out.fp} 人）。適用於垃圾郵件、定罪等「絕不能冤枉好人」的情境。`;
  }else if(view==='recall'){
    msg+=`【Recall 橫向視角】= ${M.fmt(out.recall)}。在真實世界全部患病的 ${out.tp+out.fn} 人中，成功抓出 ${out.tp} 人（遺漏 ${out.fn} 人！）。適用於癌症篩檢、地震警報等「寧可誤報、絕不能漏放」的情境。`;
  }else if(view==='accuracy'){
    msg+=`【Accuracy 對角視角】= ${M.fmt(out.accuracy)}。全部 40 筆樣本猜對了 ${out.tp+out.tn} 筆。請留意：當正類極少時，模型即使全部瞎猜健康，Accuracy 也能飆高，容易掩蓋抓不出病人的致命缺陷。`;
  }else if(view==='f1'){
    msg+=`【F1-Score 調和天平】= ${M.fmt(out.f1)}。Precision 與 Recall 彼此牽制拉扯；F1 採用調和平均，任何一項短板都會大幅拖垮總分。`;
  }else{
    msg+=`拖動門檻滑桿：門檻調高通常 Precision 提升但 Recall 下降。切換「觀察視角」可啟動混淆矩陣切片探照燈！`;
  }

  return {message:msg,metrics:{...out,auc:roc.auc}};
},undefined,{guide:['指標觀察視角：切換「縱看 Precision / 橫看 Recall / 斜看 Accuracy / 天平 F1」，右側矩陣會打出切片探照燈！','混淆矩陣切片：右上為微縮 2×2 矩陣，左欄為「預測正類」、上列為「真實正類」。','分類門檻：移動滑桿觀察 ROC 曲線橘點滑動，四象限人數即時變動，指標彼此拉扯。','正類比例：可調整真實正類占比（如罕見疾病不平衡數據），體會 Accuracy 的失真陷阱。','分母為零時指標顯示「未定義」。']}),
demo('regression-errors','MAE、MSE、RMSE',[
  check('outlier','🚨 加入極端離群值（觀察 MAE、MSE 與 RMSE 的跳變反應）',false)
],(c,s)=>{
  const data=s.params.outlier?[...errorData,{x:4.5,y:8}]:errorData;
  const a=G.axes(c,{xmin:0,xmax:5,ymin:0,ymax:9,x:50,y:75,w:280,h:265,xlabel:'特徵 X₁',ylabel:'標籤 y'});
  const fn=x=>.8*x+.8,out=M.regressionMetrics(data,fn);
  const errors=data.map(p=>Math.abs(p.y-fn(p.x))),scale=65/Math.max(...errors,1);

  G.dots(c,a,data);
  G.curve(c,a,fn,C.orange,3);

  // 繪製殘差垂線
  data.forEach((p,i)=>{
    const isOutlier=s.params.outlier&&i===data.length-1;
    const yHat=fn(p.x);
    G.line(c,a.X(p.x),a.Y(p.y),a.X(p.x),a.Y(yHat),isOutlier?'#dc2626':C.purple,isOutlier?3:2);
    if(isOutlier){
      G.circle(c,a.X(p.x),a.Y(p.y),6,'#dc2626');
      G.text(c,'🚨 離群值 (4.5, 8.0)',a.X(p.x)-50,a.Y(p.y)-10,11.5,'#dc2626','left');
      G.text(c,`殘差 e = 3.60`,a.X(p.x)-50,a.Y(p.y)+14,10.5,'#dc2626','left');
    }
  });

  G.heading(c,'回歸評估指標：離群值對 MAE、MSE、RMSE 的衝擊');
  G.legend(c,[['真實樣本',C.green],['預測線 (y=0.8x+0.8)',C.orange],[s.params.outlier?'離群殘差 (大錯)':'正常殘差 e',s.params.outlier?'#dc2626':C.purple]],50,420,120);

  const rx=355;
  const ratio=out.mae?out.rmse/out.mae:1;
  const sumAbs=errors.reduce((sum,v)=>sum+v,0);

  // 頂部核心診斷狀態列 (y: 48 ~ 92)
  if(s.params.outlier){
    G.rect(c,rx,48,375,44,'#fef2f2','#fca5a5',8);
    G.text(c,'🚨 警報！偵測到極端離群值大失誤（翻車樣本）',rx+16,66,13,'#b91c1c','left');
    G.text(c,`僅 1 筆離群大錯，MSE 暴衝 13 倍！RMSE 飆開 MAE 達 ${ratio.toFixed(2)} 倍！`,rx+16,82,11,'#b91c1c','left');
  }else{
    G.rect(c,rx,48,375,44,'#f0fdf4','#86efac',8);
    G.text(c,'✅ 狀態良好：模型預測均勻穩定，無離群爆雷',rx+16,66,13,'#15803d','left');
    G.text(c,`RMSE (${M.fmt(out.rmse,2)}) ≈ MAE (${M.fmt(out.mae,2)})，比值 ${ratio.toFixed(2)}（在健康門檻 ≤ 1.25 內）`,rx+16,82,11,'#15803d','left');
  }

  // 三大指標卡片 (y: 102, 178, 254，高度各 68px)
  // 1. MAE 卡片
  G.rect(c,rx,102,375,68,C.white,'#16a34a',7);
  G.rect(c,rx,102,6,68,'#16a34a',null,3);
  G.text(c,'MAE 平均絕對誤差',rx+18,118,13.5,'#15803d','left');
  G.text(c,`${M.fmt(out.mae,3)} (原單位)`,rx+360,118,16,'#15803d','right');
  G.text(c,`公式：Σ|e| / ${data.length} = ${M.fmt(sumAbs,2)} / ${data.length}`,rx+18,136,11,C.muted,'left');
  G.text(c,'白話：平均每筆猜錯多少 ｜ 特性：對離群值一視同仁，抗噪穩健',rx+18,153,11,'#15803d','left');

  // 2. MSE 卡片
  G.rect(c,rx,178,375,68,C.white,C.purple,7);
  G.rect(c,rx,178,6,68,C.purple,null,3);
  G.text(c,'MSE 均方誤差',rx+18,194,13.5,C.purple,'left');
  G.text(c,`${M.fmt(out.mse,3)} (平方單位²)`,rx+360,194,16,C.purple,'right');
  G.text(c,`公式：Σ(e²) / ${data.length} = ${M.fmt(out.sse,2)} / ${data.length}`,rx+18,212,11,C.muted,'left');
  G.text(c,'白話：誤差取平方當面積平均 ｜ 缺陷：單位不是人話，嚴禁與 MAE 比！',rx+18,229,11,C.purple,'left');

  // 3. RMSE 卡片
  G.rect(c,rx,254,375,68,C.white,'#d97706',7);
  G.rect(c,rx,254,6,68,'#d97706',null,3);
  G.text(c,'RMSE 均方根誤差',rx+18,270,13.5,'#b45309','left');
  G.text(c,`${M.fmt(out.rmse,3)} (原單位)`,rx+360,270,16,'#b45309','right');
  G.text(c,`公式：√MSE = √${M.fmt(out.mse,3)}`,rx+18,288,11,C.muted,'left');
  G.text(c,'白話：開根號還原原單位 ｜ 核心：兼顧人話單位與嚴懲大錯，專與 MAE 比',rx+18,305,10.5,'#b45309','left');

  // 底部離群值診斷條 (y: 332 ~ 398)
  G.rect(c,rx,332,375,66,s.params.outlier?'#fffbeb':'#f8fafc',C.line,8);
  G.text(c,'🎯 診斷尺：比值 RMSE / MAE',rx+18,348,12.5,C.ink,'left');
  G.text(c,`${ratio.toFixed(2)} 倍 ${s.params.outlier?'(異常！)':'(健康)'}`,rx+360,348,14.5,s.params.outlier?'#b91c1c':'#15803d','right');
  
  // 診斷條進度背景
  G.rect(c,rx+18,363,338,8,'#e2e8f0',null,4);
  const barW=Math.min(338,Math.max(20,(ratio-1)/1.2*338));
  G.rect(c,rx+18,363,barW,8,s.params.outlier?'#dc2626':'#16a34a',null,4);
  
  if(s.params.outlier){
    G.text(c,'⚠️ 比值達 1.72（遠大於 1.25）：明確存在少數極端離群大失誤！',rx+18,386,11,'#b91c1c','left');
  }else{
    G.text(c,'✅ 比值為 1.24（≤ 1.25）：代表各筆誤差小且均勻分布，無離群爆雷。',rx+18,386,11,'#15803d','left');
  }

  let msg=`MAE=${M.fmt(out.mae)}（原單位），MSE=${M.fmt(out.mse)}（平方單位），RMSE=${M.fmt(out.rmse)}（原單位）。`;
  if(s.params.outlier){
    msg+=`【離群值診斷】勾選加入 1 筆離群值後，RMSE / MAE 比值飆升至 ${ratio.toFixed(2)} 倍！這說明：不需要很多離群值，只要有一兩筆大失誤，RMSE 就會大幅甩開 MAE。`;
  }else{
    msg+=`【無離群值】RMSE (${M.fmt(out.rmse,2)}) 與 MAE (${M.fmt(out.mae,2)}) 非常接近（比值 ${ratio.toFixed(2)}）。代表模型預測極其穩定，各筆誤差均勻分布。`;
  }

  return {message:msg,metrics:{...out,squareScale:scale,absoluteErrors:errors}};
},undefined,{interactiveOnly:true,guide:['核心操作：勾選下方「加入極端離群值」，直接觀察右上角指標跳變！','MAE (平均絕對誤差)：對大失誤一視同仁，抗離群值干擾。','MSE (均方誤差)：平方放大誤差，演算法好算，但平方單位不是人話，嚴禁與 MAE 比！','RMSE (均方根誤差)：開根號變回原單位，專門用來跟 MAE 相比。','離群值診斷尺：當 RMSE 遠大於 MAE（比值 > 1.25）時，代表模型出現了致命的大翻車樣本！']}),
demo('r-squared','R² 決定係數',[
  select('model','比較預測模型',[
    ['good','正常好模型（捕捉規律）'],
    ['mean','瞎猜笨模型（只猜平均值）'],
    ['bad','幫倒忙模型（預測嚴重離譜）'],
    ['constant','常數標籤（邊界情況：SST=0）']
  ])
],(c,s)=>{
  const data = s.params.model==='constant' ? errorData.map(p=>({...p,y:3})) : errorData;
  const mu = M.mean(data.map(p=>p.y));
  const fn = s.params.model==='mean' ? ()=>mu : s.params.model==='bad' ? ()=>8 : s.params.model==='constant' ? ()=>2 : x=>0.8*x+1;
  const stats = M.regressionMetrics(data,fn);

  // 1. 左側坐標軸與幾何散佈圖 (x: 55, y: 56, w: 270, h: 280)
  const a = G.axes(c,{xmin:0,xmax:5,ymin:0,ymax:9,x:55,y:56,w:270,h:280,xlabel:'特徵 X₁',ylabel:'標籤 y'});
  
  // 基準水平虛線：笨模型猜平均線 y = mu
  G.line(c, a.X(0), a.Y(mu), a.X(5), a.Y(mu), '#dc2626', 2, [6, 4]);
  G.text(c, `基準線 ȳ = ${mu.toFixed(2)}`, a.X(2.5), a.Y(mu) - 10, 11, '#dc2626', 'center');

  // 模型預測線 (藍色實線)
  G.curve(c, a, fn, '#2563eb', 3);

  // 殘差線 (紫色垂直線) 與 真實樣本點
  data.forEach(p=>{
    const yHat = fn(p.x);
    G.line(c, a.X(p.x), a.Y(p.y), a.X(p.x), a.Y(yHat), C.purple, 2);
    G.circle(c, a.X(p.x), a.Y(p.y), 5, '#1e293b');
  });

  // 頂部大標題與左下圖例
  G.heading(c, 'R² 決定係數：消滅了多少笨模型的失誤？');
  G.legend(c, [
    ['真實樣本 y', '#1e293b'],
    ['基準線 ȳ', '#dc2626'],
    ['模型 ŷ', '#2563eb'],
    ['殘差 e', C.purple]
  ], 40, 418, 76);

  // 2. 右側看板 (rx = 355, 寬度 375)
  const rx = 355;
  const isConst = s.params.model==='constant';
  const isBad = stats.r2 !== null && stats.r2 < 0;
  const isMean = stats.r2 !== null && Math.abs(stats.r2) < 1e-4;

  // A. 頂部核心分數卡片 (y: 48 ~ 116，高度 68)
  if(isConst){
    G.rect(c, rx, 48, 375, 68, '#fffbeb', '#fde68a', 8);
    G.rect(c, rx, 48, 6, 68, '#d97706', null, 3);
    G.text(c, 'R² 決定係數 (衡量規律解釋力)', rx+18, 65, 13, '#b45309', 'left');
    G.text(c, '未定義 (SST = 0)', rx+360, 65, 15, '#b45309', 'right');
    G.text(c, '⚠️ 數據標籤全為常數，完全無波動！分母 SST=0 無法計算。', rx+18, 94, 11, '#b45309', 'left');
  } else if(isBad){
    G.rect(c, rx, 48, 375, 68, '#fef2f2', '#fca5a5', 8);
    G.rect(c, rx, 48, 6, 68, '#dc2626', null, 3);
    G.text(c, 'R² 決定係數 (衡量規律解釋力)', rx+18, 65, 13, '#b91c1c', 'left');
    G.text(c, `${stats.r2.toFixed(3)} (負數！)`, rx+360, 65, 16, '#b91c1c', 'right');
    G.text(c, '🚨 嚴重災難！比直接猜平均還爛！負數代表方向搞反或嚴重過擬合。', rx+18, 94, 11, '#b91c1c', 'left');
  } else if(isMean){
    G.rect(c, rx, 48, 375, 68, '#f8fafc', '#cbd5e1', 8);
    G.rect(c, rx, 48, 6, 68, '#64748b', null, 3);
    G.text(c, 'R² 決定係數 (衡量規律解釋力)', rx+18, 65, 13, '#475569', 'left');
    G.text(c, '0.000 (0.0%)', rx+360, 65, 16, '#475569', 'right');
    G.text(c, '😐 毫無價值：預測與盲猜平均值完全相同，消滅失誤為 0%。', rx+18, 94, 11, '#475569', 'left');
  } else {
    G.rect(c, rx, 48, 375, 68, '#f0fdf4', '#86efac', 8);
    G.rect(c, rx, 48, 6, 68, '#16a34a', null, 3);
    G.text(c, 'R² 決定係數 (衡量規律解釋力)', rx+18, 65, 13, '#15803d', 'left');
    G.text(c, `${stats.r2.toFixed(3)} (${(stats.r2*100).toFixed(1)}%)`, rx+360, 65, 16, '#15803d', 'right');
    G.text(c, `🌟 表現優異：成功消滅了原本笨模型 ${(stats.r2*100).toFixed(1)}% 的失誤！`, rx+18, 94, 11.5, '#15803d', 'left');
  }

  // B. 中部失誤對照長條圖 (y: 124 ~ 268，高度 144)
  G.rect(c, rx, 124, 375, 144, C.white, C.line, 8);
  G.text(c, '📊 失誤量三線對照尺（基準 vs 殘差 vs 消滅）', rx+18, 139, 12.5, C.ink, 'left');

  const barMaxW = 210;
  const barX = rx + 18;

  // 1. SST 基準長條 (y: 154)
  G.text(c, '基準笨模型失誤 (SST 總平方和)', barX, 156, 11, '#dc2626', 'left');
  G.rect(c, barX, 164, barMaxW, 11, '#fee2e2', '#fca5a5', 3);
  G.text(c, `${stats.sst.toFixed(2)} (100%)`, barX + barMaxW + 8, 170, 11, '#b91c1c', 'left');

  // 2. SSE 殘差長條 (y: 188)
  const sseRatio = stats.sst > 0 ? stats.sse / stats.sst : 0;
  const sseW = Math.min(barMaxW, Math.max(3, sseRatio * barMaxW));
  G.text(c, '模型殘留失誤 (SSE 殘差平方和)', barX, 187, 11, C.purple, 'left');
  G.rect(c, barX, 195, barMaxW, 11, '#f1f5f9', '#e2e8f0', 3);
  G.rect(c, barX, 195, sseW, 11, '#c084fc', C.purple, 3);
  const sseText = stats.sst > 0 ? `${stats.sse.toFixed(2)} (${(sseRatio*100).toFixed(1)}%)` : `${stats.sse.toFixed(2)}`;
  G.text(c, sseText, barX + barMaxW + 8, 201, 11, C.purple, 'left');

  // 3. 消滅失誤量長條 (y: 219)
  if (isConst) {
    G.text(c, '• 數據完全無波動，無法衡量消滅失誤', barX, 230, 11, C.muted, 'left');
  } else if (stats.r2 > 0) {
    const elimW = Math.min(barMaxW, Math.max(3, stats.r2 * barMaxW));
    G.text(c, '模型消滅失誤 (SST − SSE ＝ R² 貢獻)', barX, 218, 11, '#15803d', 'left');
    G.rect(c, barX, 226, barMaxW, 11, '#f0fdf4', '#dcfce7', 3);
    G.rect(c, barX, 226, elimW, 11, '#4ade80', '#16a34a', 3);
    G.text(c, `${(stats.sst - stats.sse).toFixed(2)} (${(stats.r2*100).toFixed(1)}%)`, barX + barMaxW + 8, 232, 11, '#15803d', 'left');
    G.text(c, `🎯 R² 就等於綠條所佔的比例（消滅了 ${(stats.r2*100).toFixed(1)}%）！`, barX, 252, 11, '#15803d', 'left');
  } else if (stats.r2 === 0) {
    G.text(c, '模型消滅失誤 (SST − SSE ＝ R² 貢獻)', barX, 218, 11, '#64748b', 'left');
    G.rect(c, barX, 226, barMaxW, 11, '#f8fafc', '#e2e8f0', 3);
    G.text(c, '0.00 (0.0%)', barX + barMaxW + 8, 232, 11, '#64748b', 'left');
    G.text(c, '⚠️ 完全沒有消滅任何失誤，與瞎猜平均值完全相同！', barX, 252, 11, '#64748b', 'left');
  } else {
    G.text(c, '模型消滅失誤 (SST − SSE ＝ R² 貢獻)', barX, 218, 11, '#b91c1c', 'left');
    G.rect(c, barX, 226, barMaxW, 11, '#fef2f2', '#fca5a5', 3);
    G.text(c, '負值 (爆表)', barX + barMaxW + 8, 232, 11, '#b91c1c', 'left');
    G.text(c, `🚨 幫倒忙！失誤不減反增多出 ${(stats.sse - stats.sst).toFixed(2)}，比瞎猜還糟！`, barX, 252, 11, '#b91c1c', 'left');
  }

  // C. 底部白話公式與診斷卡片 (y: 276 ~ 406，高度 130)
  G.rect(c, rx, 276, 375, 130, '#f8fafc', C.line, 8);
  G.text(c, '💡 決定係數計算公式與白話看圖心法', rx+18, 292, 12.5, C.ink, 'left');

  const formulaStr = isConst ? 'R² = 1 − SSE / 0 （分母為 0，未定義）' :
    `R² = 1 − (${stats.sse.toFixed(2)} / ${stats.sst.toFixed(2)}) = 1 − ${sseRatio.toFixed(3)} = ${stats.r2.toFixed(3)}`;
  G.text(c, formulaStr, rx+18, 312, 12, '#0f766e', 'left');

  G.text(c, '① 紅虛線是基準笨模型：不看特徵，盲猜標籤平均值 ȳ。', rx+18, 336, 10.5, C.muted, 'left');
  G.text(c, '② 模型越準，紫色殘差條越短，綠色消滅失誤越多，R² 逼近 1。', rx+18, 356, 10.5, C.muted, 'left');
  G.text(c, '③ 若 R² < 0，代表預測不如直接盲猜平均值，徹底「幫倒忙」！', rx+18, 376, 10.5, '#b91c1c', 'left');

  const msg = stats.sst === 0 ? 'y 為常數，SST=0（無波動），R² 未定義。' :
    stats.r2 < 0 ? `R²=${M.fmt(stats.r2)}（負數！）。幫倒忙模型：預測誤差比直接猜平均還離譜，代表方向搞反或嚴重過擬合。` :
    stats.r2 === 0 ? 'R²=0.00 (0%)。瞎猜笨模型：誤差與直接猜平均值一模一樣，完全沒學到特徵。' :
    `R²=${M.fmt(stats.r2)}（約 ${(stats.r2*100).toFixed(0)}%）。正常好模型：成功消滅了原本笨模型約 ${(stats.r2*100).toFixed(0)}% 的失誤！`;

  return { message: msg, metrics: stats };
}, undefined, {
  interactiveOnly: true,
  guide: [
    '1. 左側紅虛線是「笨模型基準線」：不論特徵，一律猜標籤平均值 ȳ。',
    '2. 左側藍實線是「你的預測模型」：垂直紫色線段是預測殘差 e。',
    '3. 右側「失誤對照尺」直接對比：紅色 SST (笨模型總失誤) vs 紫色 SSE (目前殘差)。',
    '4. 綠色區塊就是 R²：代表你的模型幫忙消滅了百分之幾的笨模型失誤！',
    '5. 切換下拉選單體驗：好模型 (R²≈0.94)、瞎猜笨模型 (R²=0.00)、幫倒忙 (R²<0)、常數 (未定義)。'
  ]
})];

