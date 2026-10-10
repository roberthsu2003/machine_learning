import {demo,range,select,check} from '../../shared/scene.js';
import * as M from '../../shared/math.js';
import * as G from '../../shared/geometry.js';
const C=G.colors;
export const definitions=[
demo('supervised','監督式：分類與回歸',[
 select('task','① 選擇要預測的標籤 y',[['class','分類：房屋類型'],['reg','回歸：成交價']]),
 range('query','② 新房屋的面積 X₁（坪）',20,80,50,5)
],(c,s)=>{
 const areas=[20,30,40,60,70,80],query=s.params.query,p=s.progress;
 const learned=p>=.5,revealed=p>=1,move=M.clamp((p-.5)*2);
 const stage=p===0?0:p<.5?1:revealed?3:2;
 const titles=['先看已知的 X 與 y','由已知資料建立模型','模型已建立，準備預測','用新 X 得到預測 ŷ'];
 G.heading(c,titles[stage]);
 ['1 看資料','2 學模型','3 預測新房屋'].forEach((v,i)=>{
  G.rect(c,80+i*225,62,205,36,(i===0&&p===0)||(i===1&&p>0&&p<=.5)||(i===2&&p>.5)?C.mint:C.white,C.line);
  G.text(c,v,182+i*225,80,15);
 });
 const x=v=>155+(v-20)/60*500;
 let prediction,model;
 if(s.params.task==='class'){
  const labels=areas.map(v=>v<=40?'小坪數':'大坪數');
  const boundary=(Math.max(...areas.filter((v,i)=>labels[i]==='小坪數'))+Math.min(...areas.filter((v,i)=>labels[i]==='大坪數')))/2;
  model={boundary};prediction=query<=boundary?'小坪數':'大坪數';
  G.text(c,'標籤 y：房屋類型',90,122,15,C.ink,'left');
  [[180,'大坪數',C.orange],[290,'小坪數',C.green]].forEach(([y,label,color])=>{
   G.line(c,145,y,675,y);G.text(c,label,130,y,16,color,'right');
  });
  areas.forEach((v,i)=>{const y=labels[i]==='小坪數'?290:180;G.circle(c,x(v),y,8,labels[i]==='小坪數'?C.green:C.orange);G.text(c,`${v} 坪`,x(v),y-22,13);});
  if(p>0){c.globalAlpha=Math.min(1,p*2);G.line(c,x(boundary),145,x(boundary),322,C.purple,2,[5,4]);G.text(c,'模型門檻：50 坪',x(boundary),135,14,C.purple);c.globalAlpha=1;}
  if(learned){const y=M.lerp(235,prediction==='小坪數'?290:180,move);G.diamond(c,x(query),y,11,C.purple);G.text(c,revealed?`ŷ = ${prediction}`:'新房屋：y 未知',x(query),y+27,14,C.purple);}
  G.text(c,learned?'模型規則：X₁ ≤ 50 → 小坪數；X₁ > 50 → 大坪數':'每個圓點的位置是 X₁，所屬的房屋類型是 y',380,410,15,learned?C.ink:C.muted);
 }else{
  const prices=[1000,1550,1950,3050,3450,4000],data=areas.map((v,i)=>({x:v,y:prices[i]})),fit=M.linear(data);
  model=fit;prediction=fit.w*query+fit.b;
  const plot=G.axes(c,{xmin:20,xmax:80,ymin:0,ymax:4500,x:155,y:155,w:500,h:175,xlabel:'特徵 X₁：面積（坪）',ylabel:'標籤 y：成交價（萬元）'});
  data.forEach(q=>G.circle(c,plot.X(q.x),plot.Y(q.y),7,C.green));
  if(p>0){const end=20+60*Math.min(1,p*2);c.save();c.beginPath();c.rect(plot.x,plot.y,plot.w,plot.h);c.clip();G.line(c,plot.X(20),plot.Y(fit.w*20+fit.b),plot.X(end),plot.Y(fit.w*end+fit.b),C.orange,3);c.restore();}
  if(learned){G.diamond(c,plot.X(query),M.lerp(plot.Y(0),plot.Y(prediction),move),11,C.purple);G.line(c,plot.X(query),plot.Y(0),plot.X(query),plot.Y(prediction),C.purple,1,[4,4]);}
  G.text(c,learned?`模型：ŷ = ${M.fmt(fit.w,2)} × X₁ + ${M.fmt(fit.b,2)}`:'綠圓是已知面積與成交價的房屋',380,410,15);
 }
 if(s.params.task==='class'){
  G.line(c,155,350,655,350);
  areas.forEach(v=>G.text(c,v,x(v),368,12,C.muted));
  G.text(c,'特徵 X₁：面積（坪）',405,390,14,C.muted);
 }
 const answer=s.params.task==='class'?prediction:`${M.fmt(prediction,0)} 萬元`;
 const message=p===0?'先看圓點：每筆訓練資料都有特徵 X₁（面積）與已知標籤 y。按一次「下一步」建立模型。':!learned?'正在展示模型；這段動畫呈現模型建立結果，不是訓練迭代紀錄。':!revealed?'已建立模型。紫色菱形是新房屋，它的 y 未知。再按一次「下一步」查看預測 ŷ。':`新房屋的 X₁ = ${query} 坪 → 預測 ŷ = ${answer}。這是模型的預測，並非已知的真實標籤 y。可調整面積，再按兩次「下一步」比較。`;
 return {message,metrics:{stage,feature:query,prediction:revealed?prediction:null,model:learned?model:null}};
},'X 是整份特徵矩陣；此例只有一欄面積，以 X₁ 標示。y 是已知標籤，ŷ 是模型預測。資料為教學示例；分類門檻取兩類最近樣本的中點，回歸線用最小平方法計算。',{
 steps:2,
 guide:['圓點是已知資料：橫向看面積 X₁，縱向看標籤 y。','按一次「下一步」：顯示由 X 與 y 建立的模型。','再按一次「下一步」：紫色菱形代表新房屋，顯示預測 ŷ。','改面積或切換任務會回到起點；再按兩次「下一步」。也可播放完整過程。']
}),
demo('unsupervised','非監督式：聚類與降維',[select('mode','學習任務',[['cluster','K-Means 分群'],['projection','2D → 1D 投影示意']]),range('k','群數 k',2,4,3),range('angle','投影角度（度）',0,180,30)],(c,s)=>{
 const data=M.points(s.params.seed,30),a=G.axes(c,{xmin:-2,xmax:2,ymin:-1.5,ymax:2});
 if(s.params.mode==='projection'){const angle=s.params.angle*Math.PI/180,ux=Math.cos(angle),uy=Math.sin(angle);G.line(c,a.X(-2*ux),a.Y(-2*uy),a.X(2*ux),a.Y(2*uy),C.orange,3);data.forEach(p=>{const v=p.x*ux+p.y*uy,tx=v*ux,ty=v*uy;G.line(c,a.X(p.x),a.Y(p.y),a.X(tx),a.Y(ty),C.line,1);G.circle(c,a.X(M.lerp(p.x,tx,s.progress)),a.Y(M.lerp(p.y,ty,s.progress)),5,C.green);});G.heading(c,'把二維座標投影到一條軸','投影方向可手動旋轉；不是自動 PCA 求解');return {message:`目前投影角 ${s.params.angle}°。每個點變成一個投影座標，垂直方向的資訊會被捨棄。`,metrics:{angle:s.params.angle}};}
 const result=M.kmeans(data,s.params.k,s.step,s.params.seed);G.dots(c,a,result.assigned,'group');result.centers.forEach((p,i)=>{G.diamond(c,a.X(p.x),a.Y(p.y),13,G.palette[i]);G.text(c,`C${i+1}`,a.X(p.x),a.Y(p.y)-22,13,G.palette[i]);});G.heading(c,'K-Means：指派群組，再移動中心',`第 ${s.step} 次中心更新`);return {message:`圓點未使用真實標籤。每一步先找最近中心，再以群內平均更新中心；目前 ${s.params.k} 群。`,metrics:{centers:result.centers,iteration:s.step}};
},undefined,{steps:10,resample:true}),
demo('paradigm-comparison','範式比較',[select('mode','比較學習範式',[['supervised','監督式'],['unsupervised','非監督式'],['semi','半監督式（示意）'],['rl','強化學習（示意）']])],(c,s)=>{
 const m=s.params.mode;const labels=m==='rl'?['智能體','行動','環境','獎勵']:['樣本 X',m==='unsupervised'?'無標籤':m==='semi'?'少量 y':'完整 y','學習模型','輸出'];G.flow(c,labels,s.progress*3.99);
 for(let i=0;i<12;i++){G.cell(c,80+(i%6)*28,320+Math.floor(i/6)*28,(m==='unsupervised'||(m==='semi'&&i>2))?'?':i%2,i%2?'#f1e0ca':C.mint,22,22);}
 if(m==='rl'){G.arrow(c,660,260,100,260,C.orange);G.text(c,'由行動結果回饋獎勵',380,315,19,C.orange);}
 G.heading(c,'比較「模型得到什麼資訊」');const messages={supervised:'監督式提供 X 與 y，學習預測答案。',unsupervised:'非監督式只有 X，尋找資料結構。',semi:'半監督式結合少量標籤與大量無標籤資料；此處僅示意資訊來源。',rl:'強化學習觀察行動與環境回饋；這是回饋流程示意，並非已訓練策略。'};return {message:messages[m],metrics:{mode:m}};
},undefined,{steps:4})];
