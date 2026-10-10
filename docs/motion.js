/* Canvas 2D teaching scenes: geometry moves continuously along a controllable timeline. */
(() => {
  'use strict';
  const C={ink:'#18332f',green:'#246d58',mint:'#deeee0',line:'#bed4c6',orange:'#bd7634',purple:'#8062a8',blue:'#467fad',white:'#ffffff',muted:'#5c706a',paper:'#f7faf6'};
  const clamp=x=>Math.max(0,Math.min(1,x));
  const ease=x=>{x=clamp(x);return x*x*(3-2*x);};
  const lerp=(a,b,t)=>a+(b-a)*t;
  const local=(t,start,end)=>ease((t-start)/(end-start));
  const titles={flow:'資料如何流入模型？',network:'神經網路：看見訊號的傳遞',features:'從線條組成零件，再辨識整輛車',training:'前向預測 → 誤差回傳 → 更新權重',regression:'讓回歸直線逐步靠近資料',inference:'四個特徵，如何合成一個預測？'};
  const captions={
    flow:['資料以方塊表示，每個方塊都有輸入 X 與已知答案 y。','歷史資料移入模型；模型利用資料調整內部的規律。','模型形成可套用的映射 f(X)，準備處理新的輸入。','新的資料方塊通過模型，變成菱形的預測結果 ŷ。'],
    network:['圓形代表神經元，連線代表可學習的權重；輸入訊號從左側開始。','綠色光點沿連線移動，第一個隱藏層接收並轉換訊號。','訊號繼續傳遞到第二個隱藏層，形成新的表示。','輸出層接收訊號並形成結果；這是前向傳播的幾何示意。'],
    features:['原始影像以像素方塊表示；模型的任務是辨識圖像中的車輛。','從影像中凸顯邊緣與線條：簡單特徵先出現。','線條逐漸組合成車輪、車窗與車身等幾何零件。','零件組合成整車，最後形成「車輛」的分類結果；層級只作概念示意。'],
    training:['帶有「狗」標籤的樣本流入模型，先做前向預測。','示範模型預測「貓 80%」，與真實標籤「狗」比較，得到損失。','橘色訊號沿反方向回傳，表示損失梯度的反向傳播。','連線粗細改變，表示權重更新；新的前向傳播將使用更新後的參數。'],
    regression:['綠色圓點是簡報中的六筆成交資料；橘色線是目前模型。','虛線表示每一筆資料的預測誤差；直線開始平移、旋轉。','直線持續靠近整體資料，誤差線隨著參數變動縮短。','停在六筆資料的最小平方法解；這段動畫是參數插值展示，不是梯度下降求解。'],
    inference:['坪數、地點、房間數與屋齡，各用一個帶標籤的方塊表示。','特徵方塊沿不同路徑匯入已訓練模型，推論不會重新訓練。','模型按照頁面上的教學公式加總各項特徵貢獻。','結果由模型移出，顯示預測價格；更改下方特徵再播放，可觀察結果變化。']
  };
  const scenes=[];
  const svgFallback='幾何動畫畫布；下方文字說明目前步驟，可用播放、暫停、重播或時間軸控制。';
  function mount(chapter,type){
    const container=document.querySelector(`#chapter-${chapter}`);
    const panel=document.createElement('div');
    panel.className='motion-scene';panel.dataset.scene=type;
    panel.innerHTML=`<div class="motion-heading"><h3>${titles[type]}</h3><span>2D GEOMETRY · ANIMATION</span></div><canvas width="800" height="420" role="img" aria-label="${titles[type]}" aria-describedby="motion-caption-${type}">${svgFallback}</canvas><p class="motion-caption" id="motion-caption-${type}" aria-live="polite"></p><div class="motion-controls"><button data-action="play" aria-pressed="false">▶ 播放動畫</button><button data-action="reset">↺ 重播</button><button data-action="step">下一步 →</button><label>速度<select aria-label="${titles[type]}播放速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input class="motion-timeline" type="range" min="0" max="16" step="0.01" value="0" aria-label="${titles[type]}動畫進度"><span class="motion-stage">1 / 4</span></div>`;
    const paragraph=container.querySelector('p');paragraph.after(panel);
    const canvas=panel.querySelector('canvas'),ctx=canvas.getContext('2d');
    if(!ctx){panel.querySelector('.motion-caption').textContent='此瀏覽器不支援 Canvas 2D；請閱讀本章的文字與原始投影片。';panel.querySelector('.motion-controls').hidden=true;return;}
    const scene={type,panel,canvas,ctx,time:0,playing:false,speed:1,visible:false,stage:-1,width:0,height:0};scenes.push(scene);
    const play=panel.querySelector('[data-action=play]');
    function state(){play.textContent=scene.playing?'Ⅱ 暫停':'▶ 播放動畫';play.setAttribute('aria-pressed',String(scene.playing));}
    scene.state=state;
    play.addEventListener('click',()=>{if(scene.time>=16)scene.time=0;scene.playing=!scene.playing;state();draw(scene);});
    panel.querySelector('[data-action=reset]').addEventListener('click',()=>{scene.time=0;scene.playing=true;state();draw(scene);});
    panel.querySelector('[data-action=step]').addEventListener('click',()=>{scene.playing=false;scene.time=scene.time>=12?16:(Math.floor(scene.time/4)+1)*4;state();draw(scene);});
    panel.querySelector('select').addEventListener('change',event=>scene.speed=Number(event.target.value));
    panel.querySelector('input').addEventListener('input',event=>{scene.time=Number(event.target.value);scene.playing=false;state();draw(scene);});
    new ResizeObserver(()=>resize(scene)).observe(canvas);
    new IntersectionObserver(entries=>{scene.visible=entries[0].isIntersecting;if(!scene.visible&&scene.playing){scene.playing=false;state();}if(scene.visible)draw(scene);},{threshold:.05}).observe(panel);
    resize(scene);
  }
  function resize(s){
    const rect=s.canvas.getBoundingClientRect();if(!rect.width)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    s.canvas.width=Math.round(rect.width*dpr);s.canvas.height=Math.round(rect.height*dpr);
    s.width=rect.width;s.height=rect.height;draw(s);
  }
  function text(ctx,value,x,y,size=16,color=C.ink,align='center'){
    ctx.font=`${size>=20?'650':'500'} ${size}px system-ui, sans-serif`;ctx.fillStyle=color;ctx.textAlign=align;ctx.textBaseline='middle';ctx.fillText(value,x,y);
  }
  function line(ctx,x1,y1,x2,y2,color=C.line,width=2,dash=[]){ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
  function box(ctx,x,y,w,h,fill=C.white,stroke=C.line,r=10){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}
  function circle(ctx,x,y,r,fill=C.green,stroke=null,width=2){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke();}}
  function arrow(ctx,x1,y1,x2,y2,color=C.line,width=2){line(ctx,x1,y1,x2,y2,color,width);const a=Math.atan2(y2-y1,x2-x1);line(ctx,x2,y2,x2-9*Math.cos(a-.5),y2-9*Math.sin(a-.5),color,width);line(ctx,x2,y2,x2-9*Math.cos(a+.5),y2-9*Math.sin(a+.5),color,width);}
  function diamond(ctx,x,y,r,fill=C.green){ctx.beginPath();ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath();ctx.fillStyle=fill;ctx.fill();}
  function packet(ctx,x,y,value,fill=C.mint){box(ctx,x-27,y-23,54,46,fill,C.green,6);text(ctx,value,x,y,14);}
  function motor(ctx,x,y,t,label='f(X)'){
    box(ctx,x-70,y-65,140,130,C.mint,C.green,18);
    ctx.save();ctx.translate(x,y-9);ctx.rotate(t*.7);
    for(let i=0;i<8;i++){ctx.rotate(Math.PI/4);box(ctx,-5,-30,10,14,C.green,C.green,2);}
    circle(ctx,0,0,22,C.white,C.green,3);circle(ctx,0,0,7,C.green);ctx.restore();text(ctx,label,x,y+43,18);
  }
  function flow(s){
    const {ctx:c,time:t}=s;
    text(c,'歷史資料',105,62,18);text(c,'學習中的模型',400,62,18);text(c,'新資料與預測',680,62,18);
    arrow(c,165,195,315,195);arrow(c,480,195,626,195);
    motor(c,400,195,t,t<8?'學習中':'f(X)');
    for(let i=0;i<5;i++){
      const k=local(t,.3+i*.45,4.5+i*.4);c.globalAlpha=t<7?1-k*.9:.15;
      packet(c,lerp(75+(i%2)*70,400,k),lerp(135+Math.floor(i/2)*63,195,k),i%2?'y':'X');
    }c.globalAlpha=1;
    const p=local(t,10,13.5);if(t>=9&&t<14){packet(c,lerp(100,400,p),195,'新 X','#e8e0f1');}
    if(t>=13){const p=local(t,13,15.5);diamond(c,lerp(400,685,p),195,32,C.purple);text(c,'ŷ',lerp(400,685,p),195,20,C.white);}
    text(c,t<8?'用已知資料建立模型':'用模型處理未見過的資料',400,325,21,C.green);
    text(c,'方塊 = 輸入資料　　齒輪 = 模型　　菱形 = 預測',400,368,14,C.muted);
  }
  function nodes(){return [[90,[160,230,300]],[280,[110,180,250,320]],[500,[110,180,250,320]],[710,[180,250]]];}
  function network(s,training=false){
    const {ctx:c,time:t}=s,groups=nodes();
    const forward=training?local(t,0,4)*3:local(t,0,13)*3;
    const backward=training&&t>=8&&t<12;
    const signal=backward?3-local(t,8,12)*3:forward;
    const update=training?local(t,12,15):0;
    groups.slice(0,-1).forEach(([x,ys],i)=>ys.forEach((y,j)=>groups[i+1][1].forEach((ny,k)=>{
      const width=training?lerp(1.2,1+((j+k)%3)*1.2,update):1.3;
      line(c,x,y,groups[i+1][0],ny,update>0?C.green:C.line,width);
      if(signal>=i&&signal<i+1){const p=signal-i;circle(c,lerp(x,groups[i+1][0],p),lerp(y,ny,p),4.5,backward?C.orange:C.green);}
    })));
    groups.forEach(([x,ys],i)=>ys.forEach(y=>{const hot=Math.abs(signal-i)<.28;circle(c,x,y,hot?20:17,hot?(backward?C.orange:C.green):C.white,hot?null:C.green,2);if(hot)circle(c,x,y,7,C.white);}));
    ['輸入層','隱藏層 1','隱藏層 2','輸出層'].forEach((v,i)=>text(c,v,groups[i][0],365,15,C.muted));
    if(training){
      text(c,'真實標籤：狗',130,45,18,C.green);
      const stage=Math.min(3,Math.floor(t/4));
      text(c,['前向預測','比較答案，計算損失','梯度沿反方向回傳','更新權重（連線粗細）'][stage],470,45,22,backward?C.orange:C.green);
      if(t>=4&&t<8){box(c,627,78,145,50,'#fff0dd',C.orange,8);text(c,'預測：貓 80%',700,103,16,C.orange);arrow(c,710,170,710,133,C.orange);}
      if(t>=12)text(c,'下一輪將套用新權重',400,397,14,C.green);
    }else{text(c,'訊號沿連線逐層傳遞',400,45,23,C.green);text(c,'光點 = 訊號　　圓形 = 神經元　　線條 = 權重',400,398,14,C.muted);}
  }
  function car(c,x,y,scale=1,phase=1){
    c.save();c.translate(x,y);c.scale(scale,scale);
    const body=phase<.6?C.white:C.mint;
    box(c,-66,-9,132,40,body,C.green,9);
    c.beginPath();c.moveTo(-42,-9);c.lineTo(-23,-37);c.lineTo(24,-37);c.lineTo(47,-9);c.closePath();c.fillStyle=body;c.fill();c.strokeStyle=C.green;c.lineWidth=3;c.stroke();
    line(c,-2,-34,-2,-11,C.green,2);circle(c,-40,31,15,C.green);circle(c,40,31,15,C.green);circle(c,-40,31,6,C.white);circle(c,40,31,6,C.white);c.restore();
  }
  function features(s){
    const {ctx:c,time:t}=s;
    const xs=[115,310,505,690];
    ['原始像素','邊緣線條','幾何零件','整體辨識'].forEach((label,i)=>{text(c,label,xs[i],72,17);if(i<3)arrow(c,xs[i]+60,222,xs[i+1]-65,222);});
    for(let row=0;row<8;row++)for(let col=0;col<8;col++){const on=(row>=3&&row<=5&&col>0&&col<7)||(row===2&&col>2&&col<6)||(row===6&&(col===2||col===5));box(c,55+col*15,151+row*15,12,12,on?C.green:'#e4ebe1',on?C.green:'#e4ebe1',1);}
    const edge=local(t,3,6);c.globalAlpha=.2+.8*edge;
    car(c,310,222,.82,0);c.globalAlpha=1;
    const assembly=local(t,6,11);
    c.save();c.translate(505,222);
    box(c,lerp(-95,-55,assembly),lerp(-66,-10,assembly),110,33,C.mint,C.green,6);
    circle(c,lerp(-75,-33,assembly),lerp(83,24,assembly),13,C.green);
    circle(c,lerp(80,33,assembly),lerp(73,24,assembly),13,C.green);
    const roofY=lerp(-92,-33,assembly);line(c,-38,-10,-21,roofY,C.green,3);line(c,-21,roofY,21,roofY,C.green,3);line(c,21,roofY,39,-10,C.green,3);c.restore();
    const final=local(t,11,15);c.globalAlpha=.15+.85*final;car(c,690,222,.78,1);c.globalAlpha=1;
    if(t>12){box(c,635,310,110,38,C.green,C.green,19);text(c,'✓ 車輛',690,329,17,C.white);}
    const scanX=lerp(46,177,local(t,0,3));if(t<3)line(c,scanX,140,scanX,280,C.orange,3);
    text(c,'由簡單特徵，逐步形成較完整的表示',400,389,21,C.green);
  }
  function regression(s){
    const {ctx:c,time:t}=s,data=[[18,180],[23,200],[42,400],[50,500],[66,650],[95,950]];
    const mx=data.reduce((v,[x])=>v+x,0)/6,my=data.reduce((v,[,y])=>v+y,0)/6;
    const target=data.reduce((v,[x,y])=>v+(x-mx)*(y-my),0)/data.reduce((v,[x])=>v+(x-mx)**2,0);
    const p=local(t,3,14),a=lerp(5,target,p),b=lerp(180,my-target*mx,p);
    const x=v=>80+v*6.1,y=v=>330-v*.23;
    text(c,'直線平移、旋轉，讓整體誤差變小',400,40,22,C.green);
    for(let v=0;v<=1000;v+=250){line(c,80,y(v),690,y(v),'#e1e9dd',1);text(c,String(v),65,y(v),12,C.muted,'right');}
    line(c,80,75,80,330);line(c,80,330,700,330);
    for(let v=0;v<=100;v+=20)text(c,String(v),x(v),350,12,C.muted);
    data.forEach(([dx,dy])=>{if(t>2)line(c,x(dx),y(dy),x(dx),y(a*dx+b),C.purple,2,[5,4]);circle(c,x(dx),y(dy),6,C.green);});
    line(c,x(0),y(b),x(100),y(100*a+b),C.orange,3);
    const mse=data.reduce((v,[dx,dy])=>v+(a*dx+b-dy)**2,0)/6;
    text(c,`ŷ = ${a.toFixed(2)}x ${b<0?'−':'+'} ${Math.abs(b).toFixed(1)}`,245,391,18,C.orange);
    text(c,`MSE = ${mse.toFixed(1)} 萬元²`,575,391,16,C.purple);
    text(c,'價格（萬元）',80,64,12,C.muted,'left');text(c,'坪數',718,349,12,C.muted);
  }
  function inference(s){
    const {ctx:c,time:t}=s;
    const area=Number(document.getElementById('infer-area').value)||0,rooms=Number(document.getElementById('infer-rooms').value)||0,age=Number(document.getElementById('infer-age').value)||0,location=Number(document.getElementById('infer-location').value);
    const labels=[`坪數 ${area}`,`地點 ${location}`,`房數 ${rooms}`,`屋齡 ${age}`];
    const ys=[105,170,235,300];
    text(c,'四個輸入特徵',130,45,18);text(c,'已訓練模型',400,45,18);text(c,'預測結果',682,45,18);
    ys.forEach((y,i)=>{arrow(c,200,y,324,205);const p=local(t,3+i*.3,8+i*.3);const px=lerp(125,400,p),py=lerp(y,205,p);c.globalAlpha=1-p*.8;box(c,px-66,py-22,132,44,i%2? '#e8e0f1':C.mint,i%2?C.purple:C.green,7);text(c,labels[i],px,py,16);});c.globalAlpha=1;
    motor(c,400,205,t,'固定參數');arrow(c,477,205,635,205);
    if(t>=9&&t<12)text(c,'加總特徵貢獻',400,310,19,C.green);
    if(t>=12){const p=local(t,12,15);const x=lerp(480,683,p);diamond(c,x,205,56,C.purple);text(c,String(Math.round(10*area+location+5*rooms-5*age)),x,205,21,C.white);if(t>14)text(c,'萬元',683,291,16,C.purple);}
    text(c,'此處採用下方的教學假設公式，沒有重新訓練模型',400,389,16,C.muted);
  }
  function draw(s){
    if(!s.width)return;const c=s.ctx;
    c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,s.canvas.width,s.canvas.height);
    const scale=s.canvas.width/800,offset=(s.canvas.height-420*scale)/2;
    c.setTransform(scale,0,0,scale,0,offset);
    if(s.type==='flow')flow(s);if(s.type==='network')network(s);if(s.type==='features')features(s);if(s.type==='training')network(s,true);if(s.type==='regression')regression(s);if(s.type==='inference')inference(s);
    const stage=Math.min(3,Math.floor(s.time/4));
    if(stage!==s.stage){s.stage=stage;s.panel.querySelector('.motion-caption').textContent=captions[s.type][stage];s.panel.querySelector('.motion-stage').textContent=`${stage+1} / 4`;}
    s.panel.querySelector('input').value=s.time;s.panel.dataset.time=s.time.toFixed(2);
  }
  [[5,'flow'],[6,'network'],[7,'features'],[11,'training'],[12,'regression'],[14,'inference']].forEach(([chapter,type])=>mount(chapter,type));
  document.getElementById('inference-form').addEventListener('input',()=>{const s=scenes.find(s=>s.type==='inference');s.time=0;s.playing=false;s.state();draw(s);});
  document.getElementById('reset-inference').addEventListener('click',()=>{const s=scenes.find(s=>s.type==='inference');s.time=0;s.playing=false;s.state();draw(s);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)scenes.forEach(s=>{s.playing=false;s.state();});});
  let previous=performance.now();
  function frame(now){const dt=Math.min(.1,(now-previous)/1000);previous=now;scenes.forEach(s=>{if(s.playing&&s.visible&&!document.hidden){s.time=Math.min(16,s.time+dt*s.speed);if(s.time>=16){s.playing=false;s.state();}draw(s);}});requestAnimationFrame(frame);}
  requestAnimationFrame(frame);
})();
