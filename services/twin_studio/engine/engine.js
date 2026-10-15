(function(root){
// Phase 2 (Twin Studio integration): gate rectangles, `gate_closure` incidents, completion
// classification and spawn/completion/gate logs. Mirrors simulation.py operation by operation;
// the original archive engine is kept byte-identical in original/engine.js.
const GATE_STOP_MARGIN_M=0.5;
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
function rng(seed){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function path(route,scale){let cum=[0];for(let i=1;i<route.points.length;i++)cum.push(cum[i-1]+distance(route.points[i],route.points[i-1])*scale);return{cum,length:cum.at(-1)}}
function position(route,p,s){let i=1;while(i<p.cum.length-1&&s>p.cum[i])i++;let f=Math.min(1,Math.max(0,(s-p.cum[i-1])/(p.cum[i]-p.cum[i-1]||1)));return{x:route.points[i-1].x+(route.points[i].x-route.points[i-1].x)*f,y:route.points[i-1].y+(route.points[i].y-route.points[i-1].y)*f}}
function insideGate(g,pos){return g.x<=pos.x&&pos.x<g.x+g.w&&g.y<=pos.y&&pos.y<g.y+g.d}
function gateSpans(route,p,g){
  const spans=[],pts=route.points;
  for(let i=1;i<pts.length;i++){
    const a=pts[i-1],b=pts[i],dx=b.x-a.x,dy=b.y-a.y,seg=p.cum[i]-p.cum[i-1];
    if(seg<=0)continue;
    let t0=0,t1=1,ok=true;
    for(const [q,r] of [[-dx,a.x-g.x],[dx,g.x+g.w-a.x],[-dy,a.y-g.y],[dy,g.y+g.d-a.y]]){
      if(q===0){if(r<0){ok=false;break}}
      else{const u=r/q;if(q<0){if(u>t0)t0=u}else{if(u<t1)t1=u}}
    }
    if(!ok||t0>t1)continue;
    const s0=p.cum[i-1]+t0*seg,s1=p.cum[i-1]+t1*seg;
    if(spans.length&&s0<=spans[spans.length-1][1])spans[spans.length-1][1]=Math.max(spans[spans.length-1][1],s1);
    else spans.push([s0,s1]);
  }
  return spans}
function validate(c){let errors=[];if(!Number.isInteger(c.duration)||c.duration<10||c.duration>900)errors.push('Duration: 10–900 whole seconds.');if(!(c.scale>0&&c.scale<=2))errors.push('Scale: greater than 0 and at most 2 m/unit.');if(!Number.isInteger(c.seed)||c.seed<0||c.seed>4294967295)errors.push('Seed must be a non-negative integer.');if(!c.routes.length)errors.push('Draw at least one route.');if(c.routes.length>12)errors.push('Maximum 12 routes.');if(c.events.length>100)errors.push('Maximum 100 incidents.');const ids=c.routes.map(r=>r.id);if(ids.some(id=>typeof id!=='string'||!id)||new Set(ids).size!==ids.length)errors.push('Route IDs must be unique non-empty strings.');
  const gates=c.gates??[];
  if(!Array.isArray(gates)||gates.length>20)errors.push('Gates must be a list of at most 20 rectangles.');
  const gids=new Set();
  for(const g of Array.isArray(gates)?gates:[]){
    if(typeof g?.id!=='string'||!g.id||gids.has(g.id))errors.push('Gate IDs must be unique non-empty strings.');
    gids.add(g?.id);
    if(!['x','y','w','d'].every(k=>Number.isFinite(g?.[k]))||g.w<=0||g.d<=0||g.x<0||g.y<0||g.x+g.w>1000||g.y+g.d>700)errors.push('Gate rectangles must be inside the map with positive size.');
  }
  let total=0;for(const r of c.routes){total+=r.count;if(!r.name?.trim())errors.push('Every route needs a name.');if(r.points.length<2||r.points.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x<0||p.x>1000||p.y<0||p.y>700)||path(r,c.scale).length<1)errors.push('Routes need two distinct points inside the map.');if(!Number.isInteger(r.count)||r.count<0||r.count>1500)errors.push('Route population: 0–1500 people.');if(!(r.speed>=0.2&&r.speed<=3))errors.push('Walking speed: 0.2–3 m/s.');if(!(r.width>=0.5&&r.width<=20))errors.push('Path width: 0.5–20 m.')}
  for(const e of c.events){total+=e.type==='surge'?e.people:0;if(!e.name?.trim())errors.push('Name every incident.');if(!['surge','collapse','gate_closure'].includes(e.type))errors.push('Unknown incident type.');
    if(e.type==='gate_closure'){
      if(!gids.has(e.gateId))errors.push('Gate closure must name a configured gate.');
      if((e.routeId!==undefined&&e.routeId!==null)||(e.people??0)!==0||'at' in e)errors.push('Gate closure takes gateId, start and end only.');
      if(!Number.isInteger(e.start)||!Number.isInteger(e.end)||e.start<0||e.start>=e.end||e.end>c.duration)errors.push(`${e.name||'Incident'}: require 0 ≤ start < end ≤ ${c.duration}s.`);
      continue}
    if(!c.routes.some(r=>r.id===e.routeId))errors.push('Place each incident on a route.');if(!Number.isFinite(e.at)||e.at<0||e.at>1)errors.push('Incident location is invalid.');if(!Number.isInteger(e.start)||!Number.isInteger(e.end)||e.start<0||e.start>=e.end||e.end>c.duration)errors.push(`${e.name||'Incident'}: require 0 ≤ start < end ≤ ${c.duration}s.`);if(e.type==='surge'&&(!Number.isInteger(e.people)||e.people<1||e.people>1500))errors.push('Surge: 1–1500 additional people.')}
  if(total>3000)errors.push('Maximum 3000 people per run.');if(!Number.isInteger(c.randomCount)||c.randomCount<1||c.randomCount>10)errors.push('Random collapses: 1–10.');if(!Number.isFinite(total))errors.push('Population is invalid.');
  if(!errors.length){
    const closedIds=new Set(c.events.filter(e=>e.type==='gate_closure').map(e=>e.gateId));
    for(const g of gates){if(!closedIds.has(g.id))continue;
      for(const r of c.routes){const p=path(r,c.scale);
        const starts=[0,...c.events.filter(e=>e.type==='surge'&&e.routeId===r.id).map(e=>e.at*p.length)];
        for(const [s0,s1] of gateSpans(r,p,g))if(starts.some(s=>s0<=s&&s<=s1))errors.push(`Route '${r.name}' spawns inside gate ${g.id}, which has a closure.`)}}
  }
  return errors}
function randomEvents(c){let r=rng(c.seed+9182);return Array.from({length:c.randomCount},(_,i)=>{let start=Math.floor(c.duration*(.15+r()*.55));return{id:'auto'+i,name:'Random collapse '+(i+1),type:'collapse',routeId:c.routes[Math.floor(r()*c.routes.length)].id,at:.15+r()*.7,start,end:Math.min(c.duration,start+Math.max(3,Math.floor(c.duration*(.08+r()*.15)))),people:0}})}
function simulate(c,mode){const errors=validate(c);if(errors.length)throw Error(errors.join('\n'));
  if(!['ideal','custom','random'].includes(mode))throw Error('Unknown run mode.');
  const random=rng(c.seed),events=mode==='ideal'?[]:mode==='custom'?c.events:randomEvents(c),paths=c.routes.map(r=>path(r,c.scale)),agents=[],frames=[],rows=[];
  const gates=c.gates??[],closures=events.filter(e=>e.type==='gate_closure');
  const spans=c.routes.map((r,ri)=>gates.flatMap(g=>gateSpans(r,paths[ri],g).map(([s0,s1])=>[g.id,s0,s1])));
  const spawnLog=[],completionLog=[],gateOpen=[];
  let id=0;const schedules=[];c.routes.forEach((r,ri)=>{for(let i=0;i<r.count;i++)schedules.push({time:Math.floor(i/Math.max(1,r.count)*c.duration*.65),ri,at:0,speed:r.speed*(.85+random()*.3),id:++id})});events.filter(e=>e.type==='surge').forEach(e=>{const ri=c.routes.findIndex(r=>r.id===e.routeId);for(let i=0;i<e.people;i++)schedules.push({time:e.start+Math.floor(i/e.people*(e.end-e.start)),ri,at:e.at*paths[ri].length,speed:c.routes[ri].speed*(.85+random()*.3),id:++id})});schedules.sort((a,b)=>a.time-b.time||a.id-b.id);let cursor=0;const completed=c.routes.map(()=>0);
  const gateExits=c.routes.map(()=>0),notExit=c.routes.map(()=>0);
  for(let t=0;t<=c.duration;t++){const incoming=c.routes.map(()=>0),outgoing=c.routes.map(()=>0),stepGate=c.routes.map(()=>0),stepOther=c.routes.map(()=>0);
    const closed=new Set(closures.filter(e=>e.start<=t&&t<e.end).map(e=>e.gateId));
    gateOpen.push(Object.fromEntries(gates.map(g=>[g.id,!closed.has(g.id)])));
    while(cursor<schedules.length&&schedules[cursor].time===t){const s=schedules[cursor++];agents.push({...s,s:s.at,v:0,done:false});incoming[s.ri]++;spawnLog.push({t,id:s.id,ri:s.ri,...position(c.routes[s.ri],paths[s.ri],s.at)})}
    const active=agents.filter(a=>!a.done),cells=new Map();for(const a of active){a.pos=position(c.routes[a.ri],paths[a.ri],a.s);let k=Math.floor(a.pos.x*c.scale/2)+','+Math.floor(a.pos.y*c.scale/2);a.cell=k;cells.set(k,(cells.get(k)||0)+1)}
    for(let ri=0;ri<c.routes.length;ri++){const route=c.routes[ri],p=paths[ri],group=active.filter(a=>a.ri===ri).sort((a,b)=>b.s-a.s||a.id-b.id);let lead=Infinity;
      for(const a of group){let v=a.speed/(1+Math.max(0,(cells.get(a.cell)||0)/4-.5)*1.8);const blocks=events.filter(e=>e.type==='collapse'&&e.routeId===route.id&&t>=e.start&&t<e.end);for(const b of blocks){let loc=b.at*p.length;if(a.s<=loc+1&&a.s+v>=loc-1)v=Math.max(0,loc-1-a.s)}
        for(const [gid,s0,s1] of spans[ri]){if(!closed.has(gid))continue;if(s0<=a.s&&a.s<=s1)v=0;else if(a.s<s0)v=Math.min(v,Math.max(0,s0-GATE_STOP_MARGIN_M-a.s))}
        v=Math.min(v,Math.max(0,lead-a.s-.65));a.v=v;a.s+=v;lead=a.s;
        if(a.s>=p.length){a.done=true;outgoing[ri]++;completed[ri]++;
          const end=position(route,p,a.s),gate=gates.find(g=>insideGate(g,end))?.id??null;
          if(gate===null){notExit[ri]++;stepOther[ri]++}else{gateExits[ri]++;stepGate[ri]++}
          completionLog.push({t,id:a.id,ri,x:end.x,y:end.y,gateId:gate})}}
      let remaining=group.filter(a=>!a.done);rows.push({run:mode,seed:c.seed,time_s:t,route_id:route.id,route_name:route.name,spawned:incoming[ri],exited:outgoing[ri],active:remaining.length,completed:completed[ri],mean_speed_m_s:remaining.length?remaining.reduce((n,a)=>n+a.v,0)/remaining.length:0,mean_density_p_m2:remaining.length/(p.length*route.width),peak_local_density_p_m2:remaining.length?Math.max(...remaining.map(a=>(cells.get(a.cell)||0)/4)):0,queued:remaining.filter(a=>a.v<.1).length,net_flow:incoming[ri]-outgoing[ri],gate_exits:stepGate[ri],route_completed_not_exit:stepOther[ri]})}
    frames.push(active.filter(a=>!a.done).map(a=>{let pos=position(c.routes[a.ri],paths[a.ri],a.s);return{id:a.id,ri:a.ri,x:pos.x,y:pos.y,v:a.v}}))}
  return{mode,events,frames,rows,total:id,completed:completed.reduce((a,b)=>a+b,0),peakQueue:Math.max(...rows.map(r=>r.queued)),
    gateExits:gateExits.reduce((a,b)=>a+b,0),routeCompletedNotExit:notExit.reduce((a,b)=>a+b,0),spawns:spawnLog,completions:completionLog,gateOpen}}
function csv(result){const keys=Object.keys(result.rows[0]);const escape=v=>'"'+String(v).replaceAll('"','""')+'"';return keys.join(',')+'\n'+result.rows.map(r=>keys.map(k=>escape(typeof r[k]==='number'&&!Number.isInteger(r[k])?r[k].toFixed(4):r[k])).join(',')).join('\n')}
const api={validate,simulate,csv,randomEvents,path,position,gateSpans,insideGate,GATE_STOP_MARGIN_M};
if(typeof module!=='undefined')module.exports=api;root.TwinEngine=api;
})(globalThis);
