import {demo,range,select} from '../../shared/scene.js';
import * as M from '../../shared/math.js';
import * as G from '../../shared/geometry.js';
const C=G.colors,education=['高中','大學','碩士'],colors=['紅','綠','藍'];
export function encodingValue(mode,category){return mode==='onehot'?[0,1,2].map(i=>i===category?1:0):category;}
// A cursor follows the drawn ordinal path; stair height is a rank illustration, not a quantity.
export function ordinalCursor(category,progress){
 const path=[[155,288]];
 for(let rank=1;rank<=category;rank++)path.push([155+(rank-1)*220+100,288-(rank-1)*60],[155+(rank-1)*220+100,288-rank*60],[155+rank*220,288-rank*60]);
 const lengths=path.slice(1).map((point,i)=>Math.hypot(point[0]-path[i][0],point[1]-path[i][1]));
 let remaining=M.sum(lengths)*M.clamp(progress);
 for(let i=0;i<lengths.length;i++){if(remaining<=lengths[i])return [M.lerp(path[i][0],path[i+1][0],remaining/lengths[i]),M.lerp(path[i][1],path[i+1][1],remaining/lengths[i])];remaining-=lengths[i];}
 return path.at(-1);
}
function ordinal(c,s){
 const k=s.params.category,step=Math.min(4,Math.floor(s.progress*4+1e-6)),build=M.clamp(s.progress*4),mapped=s.progress>=.5;
 G.heading(c,'有序編碼：保留先後，不代表等距');
 G.text(c,'先指定語義順序：高中 < 大學 < 碩士；再依序指定 0、1、2',380,62,16,C.muted);
 education.forEach((name,i)=>{
  const x=65+i*220;
  G.rect(c,x,82,180,51,i===k?'#f1e0ca':C.white,i===k?C.orange:C.line);
  G.text(c,`${name}${i===k?'（選取）':''}`,x+90,107,19);
  if(i<2&&step>=1)G.arrow(c,x+183,107,x+215,107,C.green);
 });
 G.text(c,step>=2?'順序游標：從最低等級走到選取等級':'階梯高度只示意等級，沒有數量單位',380,152,14,C.muted);
 education.forEach((name,i)=>{
  const top=355-(55+i*60)*build,x=75+i*220,active=step===4&&i===k;
  G.rect(c,x,top,160,355-top,C.mint,active?C.orange:C.line,3);
  c.globalAlpha=build;G.text(c,name,x+80,318-i*60,18,C.green);c.globalAlpha=1;if(mapped)G.text(c,`代碼 ${i}`,x+80,top+41,22,active?C.orange:C.ink);
  G.text(c,`第 ${i+1} 級`,x+80,372,14,C.muted);
  if(mapped&&i<2){G.line(c,155+i*220,288-i*60,255+i*220,288-i*60,C.line,2,[5,5]);G.line(c,255+i*220,288-i*60,255+i*220,228-i*60,C.line,2,[5,5]);G.arrow(c,255+i*220,228-i*60,375+i*220,228-i*60,C.line);}
 });
 const cursor=ordinalCursor(k,M.clamp(s.progress*4-2));
 if(mapped)G.circle(c,...cursor,8,C.orange,C.white);
 const stages=['① 指定順序','② 對應整數','③ 走到選取等級','④ 讀取編碼'];
 stages.forEach((label,i)=>G.text(c,label,110+i*180,399,14,step===i+1?C.green:C.muted));
 G.text(c,step===4?`${education[k]} → ${k}；代碼表示順序，不能推論各級學歷差距相等。`:'高中 → 0　大學 → 1　碩士 → 2；數值大小只表示指定順序。',380,427,16,C.green);
 return {message:`${education[k]} → ${k}。高中 < 大學 < 碩士，對應 0 < 1 < 2；順序必須由資料語義明確指定，不能只按字母排序。階梯高度與整數間距只示意排名，不代表學歷、能力或年數的等量差距。若模型把代碼當連續數值，仍可能引入等距假設。`,metrics:{mode:'ordinal',encoding:k,order:education,step,cursor}};
}
function nominal(c,s){
 const k=s.params.category,onehot=s.params.mode==='onehot',step=Math.min(4,Math.floor(s.progress*4+1e-6));
 G.heading(c,onehot?'無序類別：One-Hot 不引入大小順序':'對照陷阱：顏色代碼會引入假的距離');
 G.text(c,'紅、綠、藍只有不同類別；沒有自然的低、中、高',380,62,16,C.muted);
 colors.forEach((name,i)=>{G.rect(c,50,100+i*77,155,51,i===k?'#f1e0ca':C.white,i===k?C.orange:C.line);G.text(c,name,127,125+i*77,22);});
 G.arrow(c,215,125+k*77,365,230,C.line);
 if(onehot){
  colors.forEach((name,j)=>{G.text(c,`${name}欄`,435+j*95,160,16,C.muted);G.cell(c,M.lerp(245+j*55,435+j*95,s.progress),M.lerp(125+k*77,222,s.progress),j===k?1:0,j===k?C.mint:C.white,48,48);});
  G.text(c,`${colors[k]} → [${encodingValue('onehot',k).join(', ')}]`,535,288,20,C.green);
  G.text(c,'不同類別之間的完整 One-Hot 距離都為 √2',380,363,17,C.green);
  G.text(c,'沒有「紅 < 綠 < 藍」的大小關係',380,405,17,C.muted);
 }else{
  G.line(c,400,238,665,238,C.line,3);
  colors.forEach((name,j)=>{const x=420+j*110;G.circle(c,x,238,7,j===k?C.orange:C.green);G.text(c,`${name} = ${j}`,x,198,17);});
  const x=M.lerp(245,420+k*110,s.progress);G.cell(c,x,M.lerp(125+k*77,294,s.progress),k,C.mint,65,45);
  G.text(c,'代碼距離：紅↔綠 = 1；紅↔藍 = 2',380,363,18,C.orange);
  G.text(c,'這些大小與距離是編碼造成的，並非顏色的自然關係',380,405,16,C.muted);
 }
 return {message:`${colors[k]} → ${onehot?`[${encodingValue('onehot',k).join(', ')}]`:k}。${onehot?'One-Hot 各欄表示是否屬於該類別；此完整三欄向量不引入類別大小順序。':'若直接用整數計算距離或線性關係，可能錯把無序顏色當成有大小、等距的量。'}切換有序學歷，對照有語義依據的排序。`,metrics:{mode:s.params.mode,encoding:encodingValue(s.params.mode,k),step}};
}
export const encoding=demo('encoding','類別編碼',[
 select('mode','編碼方式',[['ordinal','有序學歷 → 順序階梯'],['onehot','無序顏色 → One-Hot'],['bad','無序顏色 → 整數（對照陷阱）']]),
 range('category','選取類別（0／1／2 對應畫面順序）',0,2,1)
],(c,s)=>s.params.mode==='ordinal'?ordinal(c,s):nominal(c,s),
'Canvas 2D 編碼示範；階梯高度只表示先後等級，不表示數量或等距差異。',{
 steps:4,
 guide:['有序學歷同時展示高中、大學、碩士，讓三個等級與代碼可以直接比較。','播放四步：指定順序 → 對應整數 → 游標沿階梯走到選取等級 → 讀取編碼。','用類別滑桿切換等級，觀察游標停在代碼 0、1 或 2 的位置。','0 < 1 < 2 保留順序，但不能推論各級差距相等；階梯不是能力或年數刻度。','切換無序顏色：One-Hot 不引入大小；整數模式則示範人為大小與距離的陷阱。']
});
