import {demo,range,select,check} from '../../shared/scene.js';
import * as M from '../../shared/math.js';import * as G from '../../shared/geometry.js';const C=G.colors;
const houses=[[32.5,3,5,180,1850],[21,2,18,650,920],[45.2,4,2,80,2680]],names=['面積','房數','屋齡','捷運距離','成交價 y'];
export const definitions=[
demo('features','特徵與資料型態',[
 select('type','選擇房屋特徵與資料型態',[
  ['continuous','面積：連續數值'],['discrete','房數：離散計數'],
  ['ordinal','屋況：有序類別'],['nominal','所在城市：無序類別'],
  ['unstructured','房屋描述：非結構化文字']
 ])
],(c,s)=>{
 const examples={
  continuous:{name:'面積',value:'32.5 坪',type:'連續數值',explain:'可以測量，也可以有小數。32.5 位於 30 與 35 之間。'},
  discrete:{name:'房數',value:'3 間',type:'離散計數',explain:'一間、一間地計數；這個例子的房間數是整數 3。'},
  ordinal:{name:'屋況',value:'良好',type:'有序類別',explain:'此例約定：需整修 < 普通 < 良好；順序不代表等距。'},
  nominal:{name:'所在城市',value:'台中',type:'無序類別',explain:'台北、台中、高雄是不同類別，沒有自然的大小順序。'},
  unstructured:{name:'房屋描述',value:'近捷運，採光佳。',type:'非結構化文字',explain:'原始文字不是固定數值欄位；需經編碼轉為模型可用的表示。'}
 };
 const item=examples[s.params.type],p=s.progress,extract=M.clamp(p*2),show=M.clamp((p-.5)*2);
 const stage=p===0?0:p<=.5?1:2;
 G.heading(c,['① 看房屋的原始欄位','② 把選定欄位抽出來','③ 看這個特徵的資料型態'][stage]);
 G.rect(c,35,92,260,265,C.white,C.line);
 G.text(c,'房屋樣本 A',165,123,20,C.green);
 G.text(c,`選定欄位：${item.name}`,165,166,17);
 G.text(c,'欄位的原始值',165,207,15,C.muted);
 G.rect(c,60,228,210,62,C.mint,C.green);
 G.text(c,item.value,165,259,20,C.green);
 G.text(c,'這是輸入特徵，不是售價 y',165,330,14,C.muted);
 G.arrow(c,303,180,353,180,C.green);
 G.rect(c,365,92,360,265,C.white,C.line);
 G.text(c,`${item.name} → ${item.type}`,545,123,19,C.green);
 if(p===0){
  G.text(c,'按一次「下一步」',545,224,20,C.muted);
  G.text(c,'把這個欄位抽到右邊',545,256,17,C.muted);
 }else{
  const x=M.lerp(165,545,extract),y=M.lerp(259,173,extract);
  G.rect(c,x-112,y-25,224,50,C.mint,C.green);
  G.text(c,item.value,x,y,19,C.green);
  if(p===.5)G.text(c,'再按一次「下一步」看型態',545,288,17,C.muted);
 }
 if(p>.5){
  if(s.params.type==='continuous'){
   const X=v=>395+(v-20)/20*300;
   G.line(c,395,282,695,282,C.green,2);
   [20,25,30,35,40].forEach(v=>{G.line(c,X(v),277,X(v),287);G.text(c,v,X(v),307,14);});
   const v=M.lerp(20,32.5,show);G.circle(c,X(v),282,8,C.purple);
   G.text(c,'32.5',X(32.5),253,16,C.purple);
   G.text(c,'面積（坪）：連續的測量位置',545,336,14,C.muted);
  }else if(s.params.type==='discrete'){
   const count=Math.min(3,Math.floor(show*3+1e-6));
   for(let i=0;i<3;i++){G.rect(c,412+i*91,245,76,58,i<count?C.mint:C.white,i<count?C.green:C.line);G.text(c,`${i+1} 間`,450+i*91,274,17,i<count?C.green:C.muted);}
   G.text(c,'一個方塊代表一間房，合計 3 間',545,335,14,C.muted);
  }else if(s.params.type==='ordinal'){
   ['需整修','普通','良好'].forEach((value,i)=>{G.rect(c,385+i*116,243,100,57,i===2&&show>.5?C.mint:C.white,i===2&&show>.5?C.green:C.line);G.text(c,value,435+i*116,272,17);if(i<2)G.text(c,'<',493+i*116,272,20,C.purple);});
   G.text(c,'由左到右有次序，但不代表等距',545,335,14,C.muted);
  }else if(s.params.type==='nominal'){
   ['台北','台中','高雄'].forEach((value,i)=>{G.circle(c,430+i*115,270,35,i===1&&show>.5?C.mint:C.white,i===1&&show>.5?C.green:C.line);G.text(c,value,430+i*115,270,17);});
   G.text(c,'三個並列類別：沒有 < 或 > 的關係',545,335,14,C.muted);
  }else{
   G.rect(c,390,234,120,72,C.white,C.line);
   [249,260,271].forEach(y=>G.line(c,405,y,492,y,C.muted,1));G.text(c,'原始文字',450,291,14);
   G.arrow(c,516,270,558,270);
   G.rect(c,565,234,130,72,show>.5?C.mint:C.white,C.line);G.text(c,'文字編碼',630,261,17);G.text(c,'→ 特徵向量',630,289,14);
   G.text(c,'流程示意：尚未執行文字編碼',545,335,14,C.muted);
  }
 }
 G.text(c,p>.5?item.explain:'先看欄位與原始值，再看如何表示資料。',380,405,16,C.green);
 const message=p===0?`選定「${item.name}」，原始值為「${item.value}」。按一次「下一步」抽出欄位。`:p<=.5?`已抽出「${item.name}」：${item.value}。再按一次「下一步」查看${item.type}的表示。`:`${item.name} = ${item.value}，屬於${item.type}。${item.explain}`;
 return {message,metrics:{type:s.params.type,feature:item.name,value:item.value,stage,extracted:extract===1}};
},'五個選項使用同一個房屋情境；切換選項會同步更換欄位、原始值與型態圖。屋況順序為此教學例子的約定，文字編碼只示意流程。',{
 steps:2,
 guide:['先選房屋特徵；左邊同時顯示對應的欄位與原始值。','按第一次「下一步」：把欄位的值抽到右側。','按第二次「下一步」：觀察數線、計數、等級、類別或文字編碼流程。','切換選項會回到起點，請再按兩次「下一步」。也可播放或拖動時間軸。']
}),
demo('labels','標籤與任務',[select('task','標籤型態',[['regression','回歸：成交價'],['classification','分類：價格區間']]),check('answer','顯示標籤答案',true)],(c,s)=>{
 G.heading(c,'X 是輸入，y 是要預測的答案');houses.forEach((h,i)=>{G.rect(c,55,95+i*88,410,62,C.white,C.line);G.text(c,`${h[0]} 坪 · ${h[1]} 房 · ${h[2]} 年 · ${h[3]} 公尺`,260,126+i*88,16);const value=s.params.answer?(s.params.task==='regression'?h[4]+' 萬元':h[4]>1500?'高價':'一般'):'?';G.cell(c,M.lerp(420,625,s.progress),126+i*88,value,i%2?'#f1e0ca':C.mint,140,50);});G.text(c,'特徵 X',260,390,20,C.green);G.text(c,'標籤 y',625,390,20,C.purple);return {message:s.params.task==='regression'?'成交價是連續數值標籤，單位為萬元。':'為演示分類，把成交價 > 1500 萬元定義為高價類，其餘為一般類。',metrics:{task:s.params.task,labels:houses.map(h=>s.params.task==='regression'?h[4]:h[4]>1500?1:0)}}},undefined,{guide:['左邊三張是房屋卡，右邊是抽出的標籤 y。','按「下一步」或拖時間軸：看標籤從卡片移到 y 欄。','切換分類／回歸：比較色塊標籤與萬元數線標籤的差異。','取消「顯示標籤答案」可先遮住答案，自己先猜再對照。']}),
demo('matrix','特徵矩陣 X 與標籤 y',[range('row','選取樣本列',1,3,1),range('col','選取特徵欄',1,4,1)],(c,s)=>{
 G.heading(c,'3 筆樣本 × 4 個特徵 → X 的形狀 (3, 4)');houses.forEach((h,i)=>h.forEach((v,j)=>{const sx=90+j*115,sy=125+i*85,tx=j===4?655:125+j*105,ty=130+i*85;G.cell(c,M.lerp(sx,tx,s.progress),M.lerp(sy,ty,s.progress),v,j===4?'#eadff3':(i===s.params.row-1||j===s.params.col-1)?C.mint:C.white,88,52);}));names.forEach((name,i)=>G.text(c,name,i===4?655:125+i*105,83,13,i===4?C.purple:C.green));G.text(c,'X：每列是一個樣本，每欄是一個特徵',330,384,17,C.green);G.text(c,'y：(3,)',655,384,17,C.purple);return {message:`X[${s.params.row-1}, ${s.params.col-1}] = ${houses[s.params.row-1][s.params.col-1]}；同一列的 y = ${houses[s.params.row-1][4]}。`,metrics:{shape:[3,4],selected:houses[s.params.row-1][s.params.col-1],label:houses[s.params.row-1][4]}}},undefined,{guide:['左邊是三筆原始房屋資料，右邊是排好的 X 矩陣與 y 向量。','按「下一步」或拖時間軸：看每個數字移入矩陣中的位置。','用「選取樣本列／特徵欄」：綠框標出 X[i,j] 與同列的 y。','下方訊息直接顯示選中格的取值，可逐格對照。']})];
