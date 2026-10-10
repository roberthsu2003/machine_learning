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
export function mountScenes(definitions){
  definitions.forEach(def=>{
    const panel=document.querySelector(`[data-scene="${def.id}"]`);if(!panel)return;
    panel.classList.add('ready');if(def.note)panel.querySelector('.scene-note').textContent=def.note+' '+panel.querySelector('.scene-note').textContent;const canvas=panel.querySelector('canvas'),ctx=canvas.getContext('2d');
    if(!ctx){panel.querySelector('.scene-description').textContent='此瀏覽器不支援 Canvas 2D，請參考本節文字與原圖。';return;}
    const state={def,panel,canvas,ctx,time:0,playing:false,visible:false,speed:1,params:Object.fromEntries((def.fields||[]).map(f=>[f.key,f.value])),lastResult:null};
    state.params.seed=42;
    const controls=panel.querySelector('.scene-parameters');(def.fields||[]).forEach(field=>fieldControl(field,state,controls));
    const play=panel.querySelector('[data-action="play"]'),timeline=panel.querySelector('.timeline');
    state.buttonState=()=>{play.textContent=state.playing?'Ⅱ 暫停':'▶ 播放動畫';play.setAttribute('aria-pressed',String(state.playing));};
    state.draw=()=>{
      const r=canvas.getBoundingClientRect();if(!r.width||!r.height)return;
      const dpr=Math.min(devicePixelRatio||1,2),w=Math.round(r.width*dpr),h=Math.round(r.height*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);const scale=Math.min(w/760,h/450);ctx.setTransform(scale,0,0,scale,(w-760*scale)/2,(h-450*scale)/2);
      state.progress=state.time/16;state.step=Math.min(def.steps??12,Math.floor(state.progress*(def.steps??12)));state.phase=state.progress*4;
      const result=def.render(ctx,state)||{};
      if(result.message&&panel.querySelector('.scene-result').textContent!==result.message)panel.querySelector('.scene-result').textContent=result.message;
      state.lastResult=result;panel.dataset.result=JSON.stringify(result.metrics??{});panel.dataset.time=state.time.toFixed(3);
      timeline.value=state.time;panel.querySelector('.time-label').textContent=`${Math.round(state.progress*100)}%`;
    };
    play.addEventListener('click',()=>{if(state.time>=16)state.time=0;state.playing=!state.playing;state.buttonState();state.draw();});
    panel.querySelector('[data-action="reset"]').addEventListener('click',()=>{state.time=0;state.playing=false;state.buttonState();state.draw();});
    panel.querySelector('[data-action="step"]').addEventListener('click',()=>{state.playing=false;const steps=def.steps??12;state.time=state.time>=16?0:Math.min(16,(Math.floor(state.time/16*steps+1e-6)+1)*16/steps);state.buttonState();state.draw();});
    panel.querySelector('[data-action="resample"]').addEventListener('click',()=>{state.params.seed++;state.time=0;state.playing=false;state.buttonState();state.draw();});
    panel.querySelector('[data-action="resample"]').hidden=!def.resample;
    panel.querySelector('[data-action="zoom"]').addEventListener('click',event=>{const zoom=panel.classList.toggle('zoomed');event.currentTarget.textContent=zoom?'收回動畫':'放大動畫';document.body.classList.toggle('scene-open',zoom);state.draw();});
    panel.querySelector('.speed').addEventListener('change',event=>state.speed=Number(event.target.value));
    timeline.addEventListener('input',event=>{state.time=Number(event.target.value);state.playing=false;state.buttonState();state.draw();});
    if(def.pointer){let dragging=false;const move=event=>{if(!dragging)return;const r=canvas.getBoundingClientRect(),scale=Math.min(r.width/760,r.height/450),x=(event.clientX-r.left-(r.width-760*scale)/2)/scale,y=(event.clientY-r.top-(r.height-450*scale)/2)/scale;def.pointer(state,x,y);state.time=0;state.playing=false;state.buttonState();panel.querySelectorAll('.parameter input[type=range]').forEach(input=>{if(input.name in state.params){input.value=state.params[input.name];input.parentNode.querySelector('output').textContent=input.value;}});state.draw();};canvas.addEventListener('pointerdown',event=>{dragging=true;canvas.setPointerCapture(event.pointerId);move(event);});canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',()=>dragging=false);canvas.classList.add('draggable');}
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
