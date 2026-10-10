export const range=(key,label,min,max,value,step=1)=>({key,label,min,max,value,step,type:'range'});
export const select=(key,label,options,value=options[0][0])=>({key,label,type:'select',options,value});
export const check=(key,label,value=false)=>({key,label,type:'checkbox',value});
export const demo=(id,title,fields,render,note='固定示範資料；播放或單步操作會改變幾何圖形，數值依目前設定計算。',options={})=>({id,title,fields,render,note,...options});
