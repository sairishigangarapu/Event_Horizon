/* Ordered transport. This module never passes a simulator config to inference. */
window.EHLive = (() => {
  let session=null, transport=Promise.resolve(), epoch=0, sent=-1, lastQueued=-1, failed=false, runtimeUrl='http://127.0.0.1:8001';
  const badge=()=>document.getElementById('liveStatus');
  function status(text,error=false){const b=badge();b.textContent=text;b.style.color=error?'#a52b25':'';}
  async function api(path,data){const r=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:d.error||JSON.stringify(d.detail)||'Runtime request failed');return d;}
  fetch('/api/runtime-config').then(r=>r.json()).then(d=>{runtimeUrl=d.url;document.getElementById('openDashboard').href=runtimeUrl+'/dashboard/';}).catch(()=>{});
  function reset(){epoch++;session=null;sent=-1;lastQueued=-1;failed=false;status('Runtime · waiting for render');}
  async function start(runId,mode,c,time=0){
    reset();if(!runId){status('Runtime requires the Python server',true);return;}
    const token=epoch;
    status('Connecting model runtime…');
    transport=transport.catch(()=>{}).then(async()=>{
      if(token!==epoch)return;
      session=await api('/api/runtime/sessions',{run_id:runId,mode,scale:c.scale,duration:c.duration,
        routes:c.routes.map(r=>({id:r.id,name:r.name,points:r.points,width:r.width}))});
      if(token!==epoch)return;
      document.getElementById('openDashboard').href=runtimeUrl+'/dashboard/';
      session.runId=runId;session.mode=mode;
      status('Runtime connected · '+(session.readiness.production_ready?'validated':'experimental model'));
    }).catch(e=>{if(token===epoch){failed=true;status(e.message,true);}});
    advance(time);
  }
  function advance(time){
    if(failed||time<=lastQueued)return;
    lastQueued=time;const token=epoch;
    transport=transport.then(async()=>{
      if(token!==epoch||!session||failed)return;
      const id=session.session_id;
      for(let t=sent+1;t<=time;t++){
        if(token!==epoch)return;
        const records=await api(`/api/observations/${session.runId}/${session.mode}?time_s=${t}`);
        await api(`/api/runtime/sessions/${id}/advance`,{time_s:t,observations:records.observations});
        // A separate retrospective endpoint accepts ONLY labels at this reached cutoff.
        const labels=await api('/api/runtime-labels',{run_id:session.runId,mode:session.mode,time_s:t});
        await api(`/api/runtime/sessions/${id}/evaluation`,labels);
        if(token!==epoch)return;
        sent=t;status(`Runtime · ${sent}s sent · ${records.observations.length} reports available`);
      }
    }).catch(e=>{if(token===epoch){failed=true;status('Feed stopped: '+e.message+' · use Reconnect runtime',true);}});
  }
  document.getElementById('retryRuntime').onclick=()=>window.dispatchEvent(new Event('eh-reconnect'));
  return {start,advance,reset,get time(){return sent}};
})();
