import {demo,range,select} from '../../shared/scene.js';import * as M from '../../shared/math.js';import * as G from '../../shared/geometry.js';const C=G.colors;
const data=Array.from({length:40},(_,i)=>({id:i,label:i%5===0?1:0}));
function bins(c,s,groups,titles){
 const xs=groups.length===2?[440,630]:[345,495,645],targets=new Map();
 groups.forEach((group,i)=>{
  const active=groups.length===3&&{train:0,validation:1,test:2}[s.params.role]===i;
  G.rect(c,xs[i]-62,95,125,230,C.white,active?G.palette[i]:C.line);
  G.text(c,titles[i],xs[i],72,16);
  group.forEach((p,j)=>targets.set(p.id,[xs[i]-42+(j%4)*25,130+Math.floor(j/4)*22]));
  G.text(c,`${group.length} 筆`,xs[i],350,16,G.palette[i]);
 });
 G.text(c,'原始資料 · 40 筆',125,72,17);
 data.forEach((p,i)=>{
  const from=[35+(i%8)*26,120+Math.floor(i/8)*27],to=targets.get(p.id),t=M.clamp(s.progress*data.length-i);
  G.rect(c,from[0]-10,from[1]-10,20,20,C.white,C.line,3);G.text(c,String(p.id+1).padStart(2,'0'),...from,10,C.muted);
  const x=M.lerp(from[0],to[0],t),y=M.lerp(from[1],to[1],t);
  G.rect(c,x-10,y-10,20,20,p.label===1?C.orange:C.green,null,3);G.text(c,String(p.id+1).padStart(2,'0'),x,y,10,C.white);
 });
 G.legend(c,[['多數類',C.green],['少數類',C.orange]],45,300,100);
 G.text(c,'ID 保留作為來源對照',125,327,12,C.muted);
}
// Keep the source visible while a second set of numbered tiles demonstrates the split.
function sampleTile(c,p,x,y,{ghost=false,border=null}={}){
  G.rect(c,x-13,y-12,26,24,ghost?C.white:(p.label===1?C.orange:C.green),border,4);
  G.text(c,String(p.id+1).padStart(2,'0'),x,y,12,ghost?C.muted:C.white);
}
function splitStrategies(c,s){
  const mobile=s.canvas?.getBoundingClientRect().width<550,width=mobile?380:760;
  const {mode,ratio,seed}=s.params,out=M.split(data,ratio/100,mode,seed);
  const groups=[out.train,out.test],trainIds=new Set(out.train.map(p=>p.id));
  // Stratification samples each class separately; the numbered preview groups the classes.
  const ordered=mode==='stratified'?[...data].sort((a,b)=>b.label-a.label||a.id-b.id):[...out.train,...out.test];
  const counts=groups.map(group=>[group.length,group.filter(p=>p.label===1).length]);
  const stage=s.progress<.35?0:s.progress<1?1:2;
  const titles={random:'隨機：先打散，再按比例切分',stratified:'分層：各類別分別抽樣，保留類別比例',time:'時間：較早資料訓練，較晚資料測試'};
  G.text(c,mobile?{random:'隨機：打散後按比例切分',stratified:'分層：各類別按比例抽樣',time:'時間：過去訓練，未來測試'}[mode]:titles[mode],width/2,26,mobile?18:22,C.green);
  G.rect(c,mobile?25:190,49,12,12,C.green,null,2);G.text(c,'多數類 32 筆',mobile?45:210,55,13,C.ink,'left');
  G.rect(c,mobile?200:355,49,12,12,C.orange,null,2);G.text(c,'少數類 8 筆',mobile?220:375,55,13,C.ink,'left');
  G.text(c,'方塊數字＝樣本 ID',mobile?25:555,mobile?80:55,13,C.muted,'left');
  G.text(c,'① 原始資料 · 40 筆（保留對照）',25,mobile?106:84,15,C.ink,'left');
  const slot=i=>[(mobile?37:57)+(i%(mobile?10:20))*34,(mobile?133:112)+Math.floor(i/(mobile?10:20))*30];
  data.forEach((p,i)=>sampleTile(c,p,...slot(i)));
  const orderSlot=i=>[(mobile?37:57)+(i%(mobile?10:20))*34,(mobile?284:205)+Math.floor(i/(mobile?10:20))*30];
  const orderTitle={random:'② 打散順序 · 外框：綠色 Train／紫色 Test',stratified:'② 按類別抽樣 · 外框：綠色 Train／紫色 Test',time:'② 保留時間順序 · 外框：綠色過去／紫色未來'};
  G.text(c,mobile?{random:'② 隨機打散 · 綠框 Train／紫框 Test',stratified:'② 按類別抽樣 · 綠框 Train／紫框 Test',time:'② 時間順序 · 綠框過去／紫框未來'}[mode]:orderTitle[mode],25,mobile?257:177,mobile?14:15,C.ink,'left');
  const targets=new Map(),arrived=[[],[]],testColor=C.purple;
  groups.forEach((group,i)=>{
    const x=mobile?15:i===0?20:478,w=mobile?350:i===0?438:262,y=mobile?(i===0?455:650):280,color=i===0?C.green:testColor;
    G.rect(c,x,y,w,mobile?(i===0?185:155):136,C.white,color);
    const timeRole=mode==='time'?(i===0?' · 過去':' · 未來'):'';
    G.text(c,`${i===0?'Train 訓練集':'Test 測試集'}${timeRole} · ${group.length} 筆`,x+14,y+18,15,color,'left');
    const columns=mobile?10:i===0?12:8;
    group.forEach((p,j)=>targets.set(p.id,[mobile?37+(j%columns)*34:i===0?42+(j%columns)*34:496+(j%columns)*30,y+46+Math.floor(j/columns)*27]));
  });
  ordered.forEach((p,i)=>{
    const train=trainIds.has(p.id),color=train?C.green:testColor,from=slot(p.id),middle=orderSlot(i),to=targets.get(p.id);
    sampleTile(c,p,...middle,{ghost:true,border:color});
    const reorder=M.clamp(s.progress/.35*ordered.length-i),move=M.clamp((s.progress-.35)/.65*ordered.length-i);
    const eased=t=>t*t*(3-2*t);
    let position=middle;
    if(s.progress<.35){const t=eased(reorder);position=[M.lerp(from[0],middle[0],t),M.lerp(from[1],middle[1],t)];}
    else{const t=eased(move);position=[M.lerp(middle[0],to[0],t),M.lerp(middle[1],to[1],t)];}
    if(move===1)arrived[train?0:1].push(p);
    sampleTile(c,p,...position,{border:color});
  });
  if(mode==='time'){
    const last=out.train.at(-1).id+1,first=out.test[0].id+1;
    G.text(c,mobile?`時間分界：ID 1–${last} → Train（過去）`:`時間分界：ID ${last} 以前 → Train ｜ ID ${first} 起 → Test（不打散）`,width/2,mobile?405:263,14,C.ink);
    if(mobile)G.text(c,`ID ${first}–40 → Test（未來），不打散`,width/2,428,14,C.ink);
    const boundary=orderSlot(out.train.length);
    G.line(c,boundary[0]-17,boundary[1]-15,boundary[0]-17,boundary[1]+15,testColor,3);
  }else if(mode==='stratified'){
    G.text(c,`少數類：Train ${counts[0][1]} 筆／Test ${counts[1][1]} 筆${mobile?'':'；多數類各自按比例抽樣'}`,width/2,mobile?405:263,14,C.ink);
    if(mobile)G.text(c,'多數類也各自按比例抽樣',width/2,428,14,C.ink);
  }else {
    G.text(c,mobile?'每個 ID 只分配一次':'每個 ID 只分配一次；隨機切分不保證兩份資料的類別比例相同',width/2,mobile?405:263,14,C.ink);
    if(mobile)G.text(c,'隨機切分不保證兩份資料的類別比例相同',width/2,428,14,C.ink);
  }
  groups.forEach((group,i)=>{
    const minority=counts[i][1],percent=(minority/group.length*100).toFixed(1);
    G.text(c,`移入 ${arrived[i].length}/${group.length} · 目標少數類 ${minority}/${group.length} (${percent}%)`,mobile?29:i===0?34:492,mobile?(i===0?625:790):405,13,C.muted,'left');
  });
  const phase=['③ 等待分配：按播放或下一步，觀察編號的去向','③ 逐筆分配：上方保留原始資料與抽樣位置','③ 切分完成：40 筆全部分配，Train／Test 交集為 0'][stage];
  G.text(c,mobile?['③ 按播放或下一步，觀察 ID 去向','③ 逐筆分配，原始資料保留對照','③ 完成：40 筆已分配，交集 0 筆'][stage]:phase,width/2,mobile?830:437,14,C.green);
  const extra=mode==='time'?`ID 1–${out.train.at(-1).id+1} 用來訓練，ID ${out.test[0].id+1}–40 用來測試；時間順序不打散。`:mode==='stratified'?'各類別分別抽樣；整數筆數使比例只能近似原始的 20%。':'隨機打散後切分；可按「重抽資料」比較類別比例。';
  return {message:`原始少數類 8/40（20%）。Train 少數類 ${counts[0][1]}/${counts[0][0]}（${(counts[0][1]/counts[0][0]*100).toFixed(1)}%），Test ${counts[1][1]}/${counts[1][0]}（${(counts[1][1]/counts[1][0]*100).toFixed(1)}%）。${extra}`,metrics:{mode,counts,train:out.train.map(p=>p.id),test:out.test.map(p=>p.id),arrived:arrived.map(group=>group.map(p=>p.id)),stage}};
}
import {leakage} from './leakage.js';
export const definitions=[
demo('train-test','訓練／測試分割',[range('ratio','訓練比例（%）',40,90,70,5)],(c,s)=>{const out=M.split(data,s.params.ratio/100,'random',s.params.seed);G.heading(c,'同一個樣本只能進入一份資料');bins(c,s,[out.train,out.test],['Train','Test']);return {message:`Train ${out.train.length} 筆、Test ${out.test.length} 筆，交集 0 筆。重抽資料會更換固定種子。`,metrics:{train:out.train.map(p=>p.id),test:out.test.map(p=>p.id)}};},undefined,{resample:true,guide:['左邊 40 格是原始資料（顏色代表類別），右邊是 Train／Test 容器。','按「下一步」或拖時間軸：看方塊打散後移入兩個容器。','調整訓練比例：比較兩容器筆數，下方訊息的交集恆為 0。','按「重抽資料」換一組固定種子的分割結果。']}),
demo('train-validation-test','三份資料的角色',[range('ratio','訓練比例（%）',40,80,60,5),select('role','目前流程',[['train','學模型參數'],['validation','調整超參數'],['test','最終評估']])],(c,s)=>{const all=M.shuffle(data,s.params.seed),n=Math.floor(40*s.params.ratio/100),v=Math.floor((40-n)/2),groups=[all.slice(0,n),all.slice(n,n+v),all.slice(n+v)];G.heading(c,'Train 學參數，Validation 調設定，Test 留最後');bins(c,s,groups,['Train','Validation','Test']);const idx={train:0,validation:1,test:2}[s.params.role];const targetX=[345,495,645][idx];G.arrow(c,targetX,370,380,392,G.palette[idx]);G.text(c,['Train → 學習模型參數','Validation → 選擇超參數','Test → 定型後的最終評估'][idx],380,418,18,G.palette[idx]);return {message:`${s.params.role}：${[groups[0].length,groups[1].length,groups[2].length][idx]} 筆資料；Test 不參與訓練或調參。`,metrics:{counts:groups.map(g=>g.length),role:s.params.role}};},undefined,{resample:true,guide:['左邊是原始資料，右邊是 Train／Validation／Test 三個容器。','按「下一步」：看資料分別移入三個容器。','切換「目前流程」：看下方文字說明該份資料流向哪裡。','Test 保持封存，不參與訓練或調參。']}),
demo('split-strategies','隨機、分層與時間切分',[select('mode','切分方式',[['random','隨機'],['stratified','分層'],['time','時間順序']]),range('ratio','訓練比例（%）',40,90,70,5)],splitStrategies,undefined,{viewSize:width=>width<550?[380,850]:[760,450],resample:true,guide:['上方保留 40 筆原始資料；數字是樣本 ID，綠色為多數類、橘色為少數類。','按「播放」或「下一步」：先看抽樣順序，再追蹤同一 ID 移入 Train／Test。','中間外框表示去向：綠框進 Train，紫框進 Test；下方顯示移入進度與少數類比例。','切換隨機／分層／時間並重播；時間模式標示過去與未來的 ID 分界。','調整訓練比例或重抽資料後，動畫會回到起點；拖時間軸可以逐段回看。']}),
leakage];
