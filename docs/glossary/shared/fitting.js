import * as M from './math.js';import * as G from './geometry.js';
export function fitPlot(c,s,{degree=2,lambda=.0001,n=12,noise=.3,seed=42,reveal=true}={}){
 const train=M.samples(seed,n,noise),validation=M.samples(97,35,.12),weights=M.polynomial(train,degree,lambda),fn=x=>M.predict(weights,x),a=G.axes(c,{ymin:-.7,ymax:3.2});
 G.curve(c,a,M.truth,G.colors.line,2);G.curve(c,a,x=>M.lerp(.7,fn(x),Math.min(1,.2+s.progress)),G.colors.orange,3);G.dots(c,a,train);
 if(reveal)G.dots(c,a,validation.slice(0,Math.floor(s.progress*validation.length)),'label','diamond');
 return {train,validation,weights,fn,plot:a,trainError:M.mse(train,fn),validationError:M.mse(validation,fn)};
}
export function gapBars(c,train,validation){const max=Math.max(train,validation,.01);[['訓練誤差',train],['驗證誤差',validation]].forEach(([label,v],i)=>{G.text(c,label,150,135+i*125,17,G.colors.muted);G.rect(c,260,110+i*125,Math.max(2,380*v/max),52,G.palette[i],null,5);G.text(c,M.fmt(v,3),680,136+i*125,22,G.palette[i]);});}
