(function(root){
const stops=[[0,'#b3d76d'],[1,'#d8de62'],[2,'#f6ba4e'],[4,'#f57b38'],[6,'#ce3d32']];
function color(d){let i=0;while(i<stops.length-2&&d>stops[i+1][0])i++;const [a,ca]=stops[i],[b,cb]=stops[i+1],t=Math.max(0,Math.min(1,(d-a)/(b-a)));return'#'+[1,3,5].map(k=>Math.round(parseInt(ca.slice(k,k+2),16)*(1-t)+parseInt(cb.slice(k,k+2),16)*t).toString(16).padStart(2,'0')).join('')}
function build(agents,scale){const map=new Map(),unit=2/scale;for(const a of agents){let x=Math.min(999.999999,Math.max(0,a.x)),y=Math.min(699.999999,Math.max(0,a.y)),ix=Math.floor(x/unit),iy=Math.floor(y/unit),key=ix+','+iy;if(!map.has(key)){let w=Math.min(unit,1000-ix*unit),h=Math.min(unit,700-iy*unit);map.set(key,{key,x:ix*unit,y:iy*unit,w,h,count:0,area:w*h*scale*scale})}map.get(key).count++}return[...map.values()].map(c=>({...c,density:c.count/c.area,color:color(c.count/c.area)}))}
const api={build,color,stops};root.EHDensity=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
