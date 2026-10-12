import {clamp} from './math.js';
const scenes=[];
function fieldControl(field,state,element){
  const label=document.createElement('label');label.className='parameter';
  const caption=document.createElement('span');caption.textContent=field.label;label.append(caption);
  let input;
  if(field.type==='select'){input=document.createElement('select');field.options.forEach(([value,title])=>{const option=document.createElement('option');option.value=value;option.textContent=title;input.append(option);});}
  else{input=document.createElement('input');input.type=field.type||'range';if(field.type!=='checkbox'){input.min=field.min;input.max=field.max;input.step=field.step??1;}}
  input.name=field.key;input.setAttribute('aria-label',field.label);
  if(field.type==='checkbox')input.checked=Boolean(state.params[field.key]);else input.value=state.params[field.key];
  label.append(input);const output=document.createElement('output');output.textContent=field.type==='range'?input.value:'';label.append(output);
  input.addEventListener('input',()=>{state.params[field.key]=field.type==='checkbox'?input.checked:field.type==='select'?input.value:Number(input.value);if(field.type==='range')output.textContent=input.value;state.time=0;state.playing=false;state.buttonState();state.draw();});element.append(label);
}
function buildGuideModal(def, panel){
  const dialog = document.createElement('dialog');
  dialog.className = 'scene-guide-dialog';
  dialog.setAttribute('aria-labelledby', `guide-title-${def.id}`);

  const header = document.createElement('div');
  header.className = 'guide-dialog-header';
  const title = document.createElement('h3');
  title.id = `guide-title-${def.id}`;
  title.textContent = `💡 ${def.title} · 操作與動作解說`;
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'guide-close-btn';
  closeBtn.setAttribute('aria-label', '關閉解說');
  closeBtn.textContent = '✕';
  header.append(title, closeBtn);

  const body = document.createElement('div');
  body.className = 'guide-dialog-body';

  if(def.guide && def.guide.length){
    const sec1 = document.createElement('div');
    sec1.className = 'guide-dialog-section';
    const h4 = document.createElement('h4');
    h4.textContent = '🎯 推薦操作與觀察步驟';
    const ol = document.createElement('ol');
    def.guide.forEach(step => {
      const li = document.createElement('li');
      li.textContent = step;
      ol.append(li);
    });
    sec1.append(h4, ol);
    body.append(sec1);
  }

  if(def.fields && def.fields.length){
    const sec2 = document.createElement('div');
    sec2.className = 'guide-dialog-section';
    const h4 = document.createElement('h4');
    h4.textContent = '🎛️ 本場景控制參數（調整它會改變什麼？）';
    const ul = document.createElement('ul');
    def.fields.forEach(f => {
      const li = document.createElement('li');
      const name = document.createElement('strong');
      name.textContent = f.label;
      li.append(name, '：');
      if(f.type === 'select'){
        const opts = (f.options || []).map(o => o[1]).join('、');
        li.append(`下拉選單，可切換不同模式（包含：${opts}）。`);
      } else if(f.type === 'checkbox'){
        li.append('勾選方塊，可開啟／關閉該項特徵條件或顯示項目。');
      } else {
        li.append(`滑桿，可拖曳調整數值（範圍 ${f.min} ~ ${f.max}），觀察幾何圖形與數據的即時變化。`);
      }
      ul.append(li);
    });
    sec2.append(h4, ul);
    body.append(sec2);
  }

  const sec3 = document.createElement('div');
  sec3.className = 'guide-dialog-section';
  const h4 = document.createElement('h4');
  h4.textContent = '🎮 底部控制按鈕功能說明';
  const actionList = document.createElement('ul');
  const actions = [
    ['▶ 播放動畫', '自動連續播放動畫，動態展示數據計算、演算法推演與邊界形成過程。'],
    ['下一步 →', '單步推進關鍵步驟（每按一次前進一步），適合細細觀察每階段的圖形與數值變化。'],
    ['↺ 重設', '將動畫進度歸零並復原初始狀態，方便重新看起。'],
    ['放大動畫', '展開全螢幕視角，查看更多幾何座標細節（按 Esc 鍵即可返回）。'],
    ['時間軸／速度', '按住時間軸可自由手動倒帶或快進；速度選單可調慢（0.5×）或調快（2×）。']
  ];
  if(def.resample){
    actions.push(['重抽資料', '隨機生成一組新的樣本數據，驗證演算法規律是否依然成立。']);
  }
  if(def.pointer){
    actions.push(['畫布拖曳', '可直接在畫布上以滑鼠或手指拖曳查詢點／焦點，直觀感受位置變化。']);
  }
  actions.forEach(([badge, desc]) => {
    const li = document.createElement('li');
    const b = document.createElement('span');
    b.className = 'guide-action-badge';
    b.textContent = badge;
    li.append(b, ' ', desc);
    actionList.append(li);
  });
  sec3.append(h4, actionList);
  body.append(sec3);

  const footer = document.createElement('div');
  footer.className = 'guide-dialog-footer';
  const confirmBtn = document.createElement('button');
  confirmBtn.type = 'button';
  confirmBtn.className = 'button primary';
  confirmBtn.textContent = '我知道了，開始體驗';
  footer.append(confirmBtn);

  dialog.append(header, body, footer);
  panel.append(dialog);

  const openModal = () => {
    if(typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  };
  const closeModal = () => {
    if(typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
  };

  closeBtn.addEventListener('click', closeModal);
  confirmBtn.addEventListener('click', closeModal);
  dialog.addEventListener('click', (e) => {
    if(e.target === dialog) closeModal();
  });

  return openModal;
}

export function mountScenes(definitions){
  definitions.forEach(def=>{
    const panel=document.querySelector(`[data-scene="${def.id}"]`);if(!panel)return;
    panel.classList.add('ready');
    const openGuide = buildGuideModal(def, panel);
    const head = panel.querySelector('.scene-head');
    if(head){
      const headBtn = document.createElement('button');
      headBtn.type = 'button';
      headBtn.className = 'guide-chip';
      headBtn.textContent = '💡 操作解說';
      headBtn.title = '點擊查看本場景完整操作指南與按鈕解說';
      headBtn.addEventListener('click', openGuide);
      head.append(headBtn);
    }
    if(def.note)panel.querySelector('.scene-note').textContent=def.note+' '+panel.querySelector('.scene-note').textContent;
    const canvas=panel.querySelector('canvas'),ctx=canvas.getContext('2d');
    const stage=document.createElement('div');stage.className='scene-stage';stage.tabIndex=0;stage.setAttribute('role','region');stage.setAttribute('aria-label',`${def.title}畫布，可左右捲動`);canvas.before(stage);stage.append(canvas);

    if(def.viewSize)stage.classList.add('responsive-stage');
    const hint=document.createElement('p');hint.className='canvas-scroll-hint';if(def.viewSize)hint.hidden=true;hint.textContent='畫布可左右滑動或用方向鍵捲動；按「放大動畫」可查看更多細節。';stage.before(hint);
    if(!ctx){panel.querySelector('.scene-description').textContent='此瀏覽器不支援 Canvas 2D，請參考本節文字與原圖。';return;}
    const state={def,panel,canvas,ctx,time:0,playing:false,visible:false,speed:1,params:Object.fromEntries((def.fields||[]).map(f=>[f.key,f.value])),lastResult:null};
    state.params.seed=42;
    const controls=panel.querySelector('.scene-parameters');(def.fields||[]).forEach(field=>fieldControl(field,state,controls));
    const play=panel.querySelector('[data-action="play"]'),timeline=panel.querySelector('.timeline');
    state.buttonState=()=>{play.textContent=state.playing?'Ⅱ 暫停':'▶ 播放動畫';play.setAttribute('aria-pressed',String(state.playing));};
    state.draw=()=>{
      const r=canvas.getBoundingClientRect();if(!r.width||!r.height)return;
      const [viewWidth,viewHeight]=def.viewSize?.(r.width)??[760,450];
      if(def.viewSize)canvas.style.aspectRatio=`${viewWidth}/${viewHeight}`;
      const dpr=Math.min(devicePixelRatio||1,2),w=Math.round(r.width*dpr),h=Math.round(r.height*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);const scale=Math.min(w/viewWidth,h/viewHeight);ctx.setTransform(scale,0,0,scale,(w-viewWidth*scale)/2,(h-viewHeight*scale)/2);
      state.progress=state.time/16;const steps=typeof def.steps==='function'?def.steps(state):def.steps??12;state.step=Math.min(steps,Math.floor(state.progress*steps+1e-6));state.phase=state.progress*4;
      const result=def.render(ctx,state)||{};
      if(result.message&&panel.querySelector('.scene-result').textContent!==result.message)panel.querySelector('.scene-result').textContent=result.message;
      state.lastResult=result;panel.dataset.result=JSON.stringify(result.metrics??{});panel.dataset.time=state.time.toFixed(3);
      timeline.value=state.time;panel.querySelector('.time-label').textContent=`${Math.round(state.progress*100)}%`;
    };
    play.addEventListener('click',()=>{if(state.time>=16)state.time=0;state.playing=!state.playing;state.buttonState();state.draw();});
    panel.querySelector('[data-action="reset"]').addEventListener('click',()=>{state.time=0;state.playing=false;state.buttonState();state.draw();});
    panel.querySelector('[data-action="step"]').addEventListener('click',()=>{state.playing=false;const steps=typeof def.steps==='function'?def.steps(state):def.steps??12;state.time=state.time>=16?0:Math.min(16,(Math.floor(state.time/16*steps+1e-6)+1)*16/steps);state.buttonState();state.draw();});
    panel.querySelector('[data-action="resample"]').addEventListener('click',()=>{state.params.seed++;state.time=0;state.playing=false;state.buttonState();state.draw();});
    panel.querySelector('[data-action="resample"]').hidden=!def.resample;
    panel.querySelector('[data-action="zoom"]').addEventListener('click',event=>{const zoom=panel.classList.toggle('zoomed');event.currentTarget.textContent=zoom?'收回動畫':'放大動畫';document.body.classList.toggle('scene-open',zoom);state.draw();});
    const guideBtn=document.createElement('button');guideBtn.type='button';guideBtn.className='btn-guide-toggle';guideBtn.setAttribute('data-action','guide');guideBtn.textContent='💡 操作解說';guideBtn.title='查看本場景完整操作指南與按鈕解說';guideBtn.addEventListener('click',openGuide);panel.querySelector('[data-action="zoom"]').after(guideBtn);
    panel.querySelector('.speed').addEventListener('change',event=>state.speed=Number(event.target.value));
    timeline.addEventListener('input',event=>{state.time=Number(event.target.value);state.playing=false;state.buttonState();state.draw();});
    if(def.pointer){let dragging=false;const move=event=>{if(!dragging)return;const r=canvas.getBoundingClientRect(),scale=Math.min(r.width/760,r.height/450),x=(event.clientX-r.left-(r.width-760*scale)/2)/scale,y=(event.clientY-r.top-(r.height-450*scale)/2)/scale;def.pointer(state,x,y);state.time=0;state.playing=false;state.buttonState();panel.querySelectorAll('.parameter input[type=range]').forEach(input=>{if(input.name in state.params){input.value=state.params[input.name];input.parentNode.querySelector('output').textContent=input.value;}});state.draw();};canvas.addEventListener('pointerdown',event=>{if(event.pointerType==='touch'&&stage.scrollWidth>stage.clientWidth)return;dragging=true;canvas.setPointerCapture(event.pointerId);move(event);});canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',()=>dragging=false);canvas.classList.add('draggable');}
    new ResizeObserver(state.draw).observe(canvas);
    new IntersectionObserver(entries=>{state.visible=entries[0].isIntersecting;if(!state.visible){state.playing=false;state.buttonState();}else state.draw();},{threshold:.05}).observe(panel);
    scenes.push(state);state.draw();
  });
}
let previous=performance.now();
function frame(now){const dt=Math.min(.1,(now-previous)/1000);previous=now;scenes.forEach(s=>{if(s.playing&&s.visible&&!document.hidden){s.time=clamp(s.time+dt*s.speed,0,16);if(s.time>=16){s.playing=false;s.buttonState();}s.draw();}});requestAnimationFrame(frame);}
requestAnimationFrame(frame);
document.addEventListener('visibilitychange',()=>{if(document.hidden)scenes.forEach(s=>{s.playing=false;s.buttonState();});});
document.addEventListener('keydown',event=>{if(event.key==='Escape'){document.querySelectorAll('.zoomed').forEach(panel=>{panel.classList.remove('zoomed');panel.querySelector('[data-action="zoom"]').textContent='放大動畫';});document.body.classList.remove('scene-open');}});
document.querySelector('[data-print]')?.addEventListener('click',()=>window.print());
const demoLinks=[...document.querySelectorAll('.lesson-toc a')];
const tocObserver=new IntersectionObserver(entries=>{const first=entries.find(e=>e.isIntersecting);if(first)demoLinks.forEach(a=>a.toggleAttribute('aria-current',a.hash===`#${first.target.id}`));},{rootMargin:'-10% 0px -60% 0px'});
demoLinks.forEach(a=>{const target=document.getElementById(a.hash.slice(1));if(target)tocObserver.observe(target);});
