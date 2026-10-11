import {demo,range,select,check} from '../../shared/scene.js';import * as M from '../../shared/math.js';import * as G from '../../shared/geometry.js';import {gapBars} from '../../shared/fitting.js';const C=G.colors;
const stages=['定義問題','收集探索','切分與特徵','模型訓練','評估調優','部署監控'],outputs=['成功指標、Baseline','乾淨資料、EDA','Train／Val／Test、X／y','候選模型參數','驗證分數、最終測試','推論服務、漂移監測'];
export const definitions=[
demo('lifecycle','端到端生命週期',[range('focus','選擇階段',1,6,1),check('feedback','顯示監控重訓回饋',true)],(c,s)=>{
 const stage=Math.min(5,Math.floor(s.progress*6)),selected=s.time===0?s.params.focus-1:stage;
 G.heading(c,'六階段連成可回饋的生命週期');
 const positions=[[115,135],[380,135],[645,135],[645,295],[380,295],[115,295]],short=['指標與 Baseline','乾淨資料與 EDA','Train／Val／Test','候選模型參數','驗證與最終測試','服務與漂移監控'];
 const routes=positions.slice(0,-1).map(([x,y],i)=>{const [nx,ny]=positions[i+1];return nx===x?[[x,y+43],[nx,ny-43]]:nx>x?[[x+95,y],[nx-95,ny]]:[[x-95,y],[nx+95,ny]];});
 routes.forEach(([from,to],i)=>{G.arrow(c,...from,...to,i<stage?C.green:C.line);if(i===stage&&s.progress<1)G.travelling(c,from,to,(s.progress*6)%1,C.green);});
 stages.forEach((name,i)=>{const [x,y]=positions[i];G.rect(c,x-95,y-43,190,86,i===selected?C.mint:C.white,i===selected?C.green:C.line);G.text(c,`${i+1}. ${name}`,x,y-14,17);G.text(c,short[i],x,y+18,13,C.muted);});
 if(s.params.feedback){G.line(c,115,338,115,380,C.orange);G.line(c,115,380,10,380,C.orange);G.line(c,10,380,10,135,C.orange);G.arrow(c,10,135,20,135,C.orange);G.text(c,'監控發現問題 → 回到問題與資料，啟動新一輪訓練',380,410,16,C.orange);}
 return {message:`${selected+1}. ${stages[selected]} → 產出：${outputs[selected]}。依箭頭走 1→2→3→4→5→6；先切分，再在 Train 上 fit 預處理。橘色閉環表示監控可能觸發重新定義問題與重訓。`,metrics:{stage:selected+1,feedback:s.params.feedback}};
},undefined,{steps:6,guide:['六格依蛇形箭頭連接 1→2→3→4→5→6，每格列出主要產出。','播放或按「下一步」：資料包沿連線前進，階段依序亮起。','用「選擇階段」滑桿直接看該階段產出。','橘色路徑從部署監控回到第一階段，表示新一輪問題與訓練。']}),
demo('inner-loop','訓練迭代與 Epoch／Batch／Iteration',[range('n','訓練樣本 N',16,128,64,8),range('batch','Batch Size',4,32,16,4),range('epochs','Epoch 數',1,5,2),check('drop','丟棄不足一批（drop_last）',false)],(c,s)=>{
 const n=s.params.n,b=s.params.batch,per=s.params.drop?Math.floor(n/b):Math.ceil(n/b),total=per*s.params.epochs,iteration=Math.min(total,Math.floor(s.progress*total+1e-6)),done=iteration===total,epoch=per?Math.min(s.params.epochs,Math.floor(iteration/per)+1):0,within=per?iteration%per:0,last=n%b;
 G.heading(c,'一批走完四個步驟，才算一次 Iteration');
 G.text(c,`每輪 ${per} 次 · 共 ${s.params.epochs} 輪 · 已完成 ${iteration}/${total} 次更新`,380,82,18,C.green);
 if(per){G.flow(c,['前向','損失','反傳','更新'],done?3.999:((s.progress*total)%1)*4);G.line(c,675,233,675,260,C.line);G.line(c,675,260,85,260,C.line);G.arrow(c,85,260,85,233,C.line);}
 else{G.rect(c,95,145,570,85,C.white,C.orange);G.text(c,'N 小於 Batch Size 且 drop_last=true',380,174,18,C.orange);G.text(c,'沒有完整批次，不會執行任何更新',380,204,16,C.muted);}
 const first=within*b,size=Math.min(b,n-first),count=Math.min(n,48);
 for(let i=0;i<count;i++){const active=!done&&per>0&&i>=first&&i<first+size;G.cell(c,50+(i%16)*43,310+Math.floor(i/16)*27,i+1,active?C.mint:'#eceee8',28,22);}
 const current=done?'本次訓練完成；沒有下一批':`下一批：ID ${first+1}–${first+size}（${size} 筆）`;
 G.text(c,per?current:'所有樣本均被丟棄',380,408,16,per?C.green:C.orange);
 G.text(c,'方塊只預覽前 48 筆；上方更新計數使用完整 N',380,435,13,C.muted);
 return {message:`每 Epoch ${per} 次更新，總計 ${total} 次；目前完成 ${iteration} 次、Epoch ${epoch}。${s.params.drop?`drop_last=true：每輪丟棄 ${last} 筆。`:`保留最後一批${last?`（${last} 筆）`:''}，使用 ceil(N / Batch Size)。`}每按一次下一步，完成一個批次的四步流程與一次參數更新。`,metrics:{iterationsPerEpoch:per,totalIterations:total,completed:iteration,epoch,discarded:s.params.drop?last:0,batchStart:done?null:first+1,batchSize:done?0:size}};
},undefined,{steps:s=>Math.max(1,(s.params.drop?Math.floor(s.params.n/s.params.batch):Math.ceil(s.params.n/s.params.batch))*s.params.epochs),guide:['四格依次是前向→損失→反傳→更新；走完四格，完成一次 Iteration。','每按一次「下一步」，完成一個批次；已完成更新數會加 1。','設定 N／Batch／Epoch，比較每輪更新次數、最後一批大小與總次數。','勾選 drop_last 會丟棄不足一批的資料；沒有完整批次時不執行更新。']}),
demo('training-health','健康度與檢查清單',[select('state','訓練狀態',[['under','欠擬合'],['good','良好擬合'],['over','過度擬合']]),check('split','已先切分資料',true),check('fitTrain','預處理僅 fit Train',true),check('test','測試集不參與調參',true)],(c,s)=>{const [train,val]={under:[.8,.85],good:[.05,.07],over:[.01,.5]}[s.params.state];G.heading(c,'數值診斷與流程關卡，都要一起檢查');gapBars(c,train*s.progress,val*s.progress);const flags=['split','fitTrain','test'],passed=flags.filter(k=>s.params[k]).length;flags.forEach((k,i)=>{G.circle(c,195+i*190,360,18,s.params[k]?C.green:C.orange);G.text(c,s.params[k]?'✓':'!',195+i*190,360,21,C.white);G.text(c,['先切分資料','預處理只 fit Train','Test 不調參'][i],195+i*190,404,14,C.muted);});return {message:`健康度為預設教學情境：Train=${train}、Val=${val}。流程 ${passed}/3 關通過；${passed===3?'仍需按任務與資料驗證模型。':'缺少關卡可能使評估失真，請檢查資料切分與洩漏。'}前章連結可回顧實作細節。`,metrics:{train,validation:val,checksPassed:passed}};},undefined,{guide:['兩條長條是訓練／驗證誤差，下方三燈是流程關卡。','切換三種狀態：比較長條高度代表的健康度。','逐一取消勾選：看未通過關卡時訊息的警告。','數值為預設情境；診斷需數值與流程一起看。']})];
