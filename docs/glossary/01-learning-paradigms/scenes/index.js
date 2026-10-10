import {demo,range,select,check} from '../../shared/scene.js';
import * as M from '../../shared/math.js';
import * as G from '../../shared/geometry.js';
const C=G.colors;
export const definitions=[
demo('supervised','監督式：分類與回歸',[select('task','預測任務',[['class','分類'],['reg','回歸']]),check('labels','顯示真實標籤',true),range('query','新樣本的 x',-1,1,.5,.05)],(c,s)=>{
 const data=M.samples(42,18),a=G.axes(c,{ymax:2.7}),p=s.progress;
 if(s.params.task==='reg'){const fit=M.linear(data),w=M.lerp(0,fit.w,p),b=M.lerp(0,fit.b,p);G.curve(c,a,x=>w*x+b);G.dots(c,a,data);G.diamond(c,a.X(s.params.query),a.Y(w*s.params.query+b));G.heading(c,'回歸：預測連續數值',`目前 ŷ = ${M.fmt(w,2)}x + ${M.fmt(b,2)}`);return {message:`x = ${s.params.query}，目前預測 ŷ = ${M.fmt(w*s.params.query+b)}。橘線以參數插值展示擬合；終點是最小平方法解。`,metrics:{prediction:w*s.params.query+b}};}
 const points=M.points(42,24).map(p=>({...p,label:p.x>0?1:0})),plot=G.axes(c,{xmin:-2,xmax:2,ymin:-1.5,ymax:2});
 const boundary=M.lerp(-1,0,p);G.line(c,plot.X(boundary),plot.Y(-1.5),plot.X(boundary),plot.Y(2),C.purple,3);
 points.forEach((q,i)=>{const color=s.params.labels?G.palette[q.label]:C.muted;G.circle(c,plot.X(q.x),plot.Y(q.y),6,color);if(s.params.labels&&i%4===0)G.text(c,q.label?'B':'A',plot.X(q.x),plot.Y(q.y)-15,11,color);});G.diamond(c,plot.X(s.params.query),plot.Y(.2),9,s.params.query>boundary?C.orange:C.green);
 G.heading(c,'分類：預測離散類別','示意分界線逐步移到 x = 0');return {message:`新樣本 x = ${s.params.query} → 預測 ${s.params.query>boundary?'B':'A'}。圓點是有標籤的範例，菱形是新樣本；此分界線為機制示意。`,metrics:{boundary,prediction:s.params.query>boundary?'B':'A'}};
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
