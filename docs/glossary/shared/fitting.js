import * as M from './math.js';import * as G from './geometry.js';
export function fitPlot(c,s,{degree=2,lambda=.0001,n=12,noise=.3,seed=42,reveal=true}={}){
 const train=M.samples(seed,n,noise),validation=M.samples(97,35,.12),weights=M.polynomial(train,degree,lambda),fn=x=>M.predict(weights,x),a=G.axes(c,{ymin:-.7,ymax:3.2,xlabel:'特徵 X₁',ylabel:'標籤 y'});
 G.curve(c,a,M.truth,G.colors.line,2);G.curve(c,a,x=>M.lerp(.7,fn(x),Math.min(1,.2+s.progress)),G.colors.orange,3);G.dots(c,a,train);
 G.legend(c,[['訓練資料',G.colors.green],['驗證資料',G.colors.purple,'diamond'],['已擬合模型',G.colors.orange],['生成規律',G.colors.line]],90,422,170);
 if(reveal)G.dots(c,a,validation.slice(0,Math.floor(s.progress*validation.length)).map(p=>({...p,label:2})),'label','diamond');
 return {train,validation,weights,fn,plot:a,trainError:M.mse(train,fn),validationError:M.mse(validation,fn)};
}
export function gapBars(c,train,validation){G.text(c,'共用誤差尺：0 → 1（100%），情境之間可直接比較',380,80,14,G.colors.muted);const max=1; /* A shared 0–1 scale keeps scenarios comparable. */[['訓練誤差',train],['驗證誤差',validation]].forEach(([label,v],i)=>{G.text(c,label,150,135+i*125,17,G.colors.muted);G.rect(c,260,110+i*125,380*v/max,52,G.palette[i],null,5);G.text(c,M.fmt(v,3),680,136+i*125,22,G.palette[i]);});}
