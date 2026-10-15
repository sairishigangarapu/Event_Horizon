'use strict';
const $=id=>document.getElementById(id),num=v=>v===null||v===undefined?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:3});
let data=null,current=new URLSearchParams(location.search).get('session')||'',followLatest=!current,version=-1;
async function api(url){const r=await fetch(url);if(!r.ok)throw Error('Runtime request failed ('+r.status+')');return r.json();}
function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
function textRow(values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td);}return tr;}
function filtered(){return data.comparisons.filter(r=>r.zone_id===$('zone').value&&r.variable===$('variable').value&&r.horizon_s===+$('horizon').value).sort((a,b)=>a.issued_s-b.issued_s);}
function render(){
 if(!data)return;
 const z=$('zone').value,v=$('variable').value,e=data.evidence.find(e=>e.zone_id===z&&e.variable===v),a=data.actuals.find(a=>a.zone_id===z&&a.variable===v);
 $('clock').textContent=data.time_s+' / '+data.duration+' s';$('count').textContent=num(data.observation_count);
 const scores=data.evidence.map(e=>e.trust_score).filter(x=>x!==null);$('trust').textContent=scores.length?num(scores.reduce((a,b)=>a+b,0)/scores.length)+' / 100':'—';
 $('conflicts').textContent=data.evidence.filter(e=>e.flags.conflict).length;
 $('readiness').textContent=(data.readiness.production_ready?'Validated model. ':'EXPERIMENTAL MODEL — ')+(data.readiness.notice||'')+' Stream A is a synthetic central provider (EWMA), not an external live authority.';
 $('observed').textContent=num(e?.observed_value);$('actual').textContent=num(a?.value);$('actualAt').textContent=a?'At '+a.time_s+'s · held outside inference':'No reference label available at this clock';
 $('evidence').textContent=e?.observation_age_s!==null?'Age '+num(e?.observation_age_s)+'s · independent-source coverage '+num(100*(e?.coverage||0))+'%':'No received evidence';
 $('selectedTrust').textContent=e?.trust_score===null?'Unavailable':num(e?.trust_score)+' / 100';$('trustDetail').textContent='Evidence index at '+data.time_s+'s. It is not forecast accuracy.';
 $('flags').replaceChildren();for(const [name,on] of Object.entries(e?.flags||{})){if(!on||['missing','stale','conflict'].includes(name))continue;const t=document.createElement('span');t.textContent=name.replaceAll('_',' ');if(name!=='recently_verified')t.className='warn';$('flags').append(t);}
 const rows=filtered();$('rows').replaceChildren(...rows.slice(-200).reverse().map(r=>textRow([r.issued_s+'s → '+r.target_s+'s',num(r.central),num(r.architecture),r.actual===null?'Pending':num(r.actual),num(r.difference),r.trust_score===null?'Unavailable':num(r.trust_score)+' / 100'])));
 $('rowCount').textContent=rows.length+' forecasts · latest 200 displayed';$('chartInfo').textContent=($('zone').selectedOptions[0]?.textContent||'')+' / '+v.replaceAll('_',' ')+' / '+$('horizon').value+'s horizon';
 chart(rows);
 $('metrics').replaceChildren();for(const [key,m] of Object.entries(data.metrics)){if(!key.includes('/'+v+'/')||!key.endsWith('/'+$('horizon').value+'s'))continue;const d=document.createElement('div');d.textContent=key+' · MAE '+num(m.mae)+' · RMSE '+num(m.rmse)+' · n='+m.count;$('metrics').append(d);}
 if(!$('metrics').children.length)$('metrics').textContent='Waiting for forecast target times to arrive.';
 $('method').textContent=data.trust_method;$('trees').textContent=JSON.stringify(data.tree_vectors.filter(t=>t.zone_id===z),null,2);
 $('alerts').replaceChildren();for(const alert of data.alerts||[]){const p=document.createElement('p');p.textContent=alert.kind.replaceAll('_',' ')+' / '+alert.zone_id+' / '+alert.variable+' / '+alert.reason;$('alerts').append(p);}if(!data.alerts?.length)$('alerts').textContent='No current alerts. Lack of an alert is not proof of safety.';
 $('updated').textContent='Clock '+data.time_s+'s / '+data.mode.toUpperCase();
}
function chart(rows){
 const c=$('chart'),rect=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=Math.round(rect.width*dpr);c.height=Math.round(rect.height*dpr);
 const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,p=40;
 const vals=rows.flatMap(r=>[r.central,r.architecture,r.actual]).filter(x=>x!==null);if(!rows.length||!vals.length){ctx.fillStyle='#65736a';ctx.font='14px Arial';ctx.fillText('Waiting for forecasts…',p,80);return;}
 const maxX=Math.max(...rows.map(r=>r.target_s),1),minY=Math.min(0,...vals),maxY=Math.max(1,...vals),x=t=>p+t/maxX*(w-p*2),y=v=>h-p-(v-minY)/(maxY-minY)*(h-p*2);
 ctx.font='11px monospace';for(let i=0;i<=4;i++){const v=minY+(maxY-minY)*i/4;ctx.strokeStyle='#e0e6da';ctx.beginPath();ctx.moveTo(p,y(v));ctx.lineTo(w-p,y(v));ctx.stroke();ctx.fillStyle='#65736a';ctx.fillText(num(v),0,y(v)+4);ctx.fillText(Math.round(maxX*i/4)+'s',x(maxX*i/4),h-12);}
 if(data.time_s<maxX){ctx.strokeStyle='#b7c0b1';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x(data.time_s),10);ctx.lineTo(x(data.time_s),h-p);ctx.stroke();ctx.setLineDash([]);ctx.fillText('now',x(data.time_s)+4,20);}
 for(const [key,color] of [['central','#d58132'],['architecture','#416bb7'],['actual','#257653']]){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();let pen=false;for(const r of rows){if(r[key]===null){pen=false;continue;}if(!pen)ctx.moveTo(x(r.target_s),y(r[key]));else ctx.lineTo(x(r.target_s),y(r[key]));pen=true;}ctx.stroke();for(const r of rows){if(r[key]===null)continue;ctx.beginPath();ctx.arc(x(r.target_s),y(r[key]),2.5,0,Math.PI*2);ctx.fill();}}
}
async function poll(){try{
 if(window.EHDashboard?.live)return;
 const sessions=await api('/api/runtime/sessions');$('session').replaceChildren(...sessions.map(s=>option(s.session_id,s.mode+' / '+s.run_id.slice(0,8)+' / '+s.time_s+'s')));
 if(followLatest&&sessions.length&&sessions.at(-1).session_id!==current){current=sessions.at(-1).session_id;version=-1;}
 if(!current&&sessions.length){current=sessions.at(-1).session_id;version=-1;}
 if(!sessions.some(s=>s.session_id===current)){current='';data=null;$('empty').hidden=false;$('content').hidden=true;}
 if(current){$('session').value=current;const next=await api('/api/runtime/sessions/'+current);if(next.version!==version||!data){data=next;version=next.version;const old=$('zone').value;$('zone').replaceChildren(...data.venue.zones.map(z=>option(z.zone_id,z.name)));if(data.venue.zones.some(z=>z.zone_id===old))$('zone').value=old;$('empty').hidden=true;$('content').hidden=false;$('download').disabled=false;render();}}
 $('connection').textContent=current?'● Connected · simulator clock':'● Waiting for Studio';$('connection').dataset.error=false;
 }catch(e){$('connection').textContent=e.message+' · retrying';$('connection').dataset.error=true;}finally{setTimeout(poll,1000);}}
$('session').onchange=()=>{window.EHDashboard.live=false;window.dispatchEvent(new Event('eh-world-detach'));followLatest=false;current=$('session').value;data=null;version=-1;history.replaceState(null,'','?session='+current);};
for(const id of ['zone','variable','horizon'])$(id).onchange=render;
$('download').onclick=()=>{if(current)location.href='/api/runtime/sessions/'+current+'/report.csv';};$('print').onclick=()=>window.print();window.addEventListener('resize',()=>{if(data)chart(filtered());});window.EHDashboard={live:false,apply(next){
 this.live=true;current=next.session_id;followLatest=false;data=next;version=next.version;
 $('session').replaceChildren(option(current,'Live twin / '+next.time_s+'s'));
 const old=$('zone').value;$('zone').replaceChildren(...next.venue.zones.map(z=>option(z.zone_id,z.name)));
 if(next.venue.zones.some(z=>z.zone_id===old))$('zone').value=old;
 $('empty').hidden=true;$('content').hidden=false;$('download').disabled=false;
 $('connection').textContent='● Connected · live twin';$('connection').dataset.error=false;render();
}};poll();
