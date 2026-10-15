// Ordered transport and causal payload regression.
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const elements=Object.fromEntries(['liveStatus','openDashboard','retryRuntime'].map(id=>[id,{style:{}}]));
const calls=[],listeners={};let time=-1,fail=false,sessionCounter=0;
const ctx={document:{getElementById:id=>elements[id]},console,Event:class{constructor(type){this.type=type}},fetch:async(url,opts)=>{
 const d=opts?.body?JSON.parse(opts.body):undefined;calls.push([url,d]);let result={};
 if(url==='/api/runtime-config')result={url:'http://127.0.0.1:8001'};
 else if(url==='/api/runtime/sessions'){assert(!('events' in d));assert(!('seed' in d));assert.deepEqual(Object.keys(d.routes[0]).sort(),['id','name','points','width']);time=-1;result={session_id:'session'+(++sessionCounter),readiness:{production_ready:false}};}
 else if(url.includes('/advance')){if(fail){fail=false;return{ok:false,json:async()=>({detail:'test outage'})};}assert.equal(d.time_s,time+1);time=d.time_s;result={};}
 else if(url.includes('/api/observations/')){const t=+url.split('=')[1];result={observations:[{observed_at_s:Math.max(0,t-1),available_at_s:t}]};}
 else if(url==='/api/runtime-labels')result={actuals:[]};
 else if(url.endsWith('/evaluation'))result={};else throw Error('Unexpected '+url);
 return{ok:true,json:async()=>result};
}};ctx.window=ctx;ctx.dispatchEvent=e=>listeners[e.type]?.();ctx.addEventListener=(name,fn)=>listeners[name]=fn;
vm.createContext(ctx);vm.runInContext(fs.readFileSync('dist/live.js','utf8'),ctx);
const settle=async()=>{for(let i=0;i<12;i++)await new Promise(r=>setImmediate(r));};
const config={scale:.1,duration:60,events:[{secret:true}],seed:42,routes:[{id:'r1',name:'Route',width:3,count:100,speed:1,points:[{x:10,y:10},{x:900,y:10}]}]};
(async()=>{
 await ctx.EHLive.start('a'.repeat(32),'ideal',config,0);ctx.EHLive.advance(5);await settle();assert.equal(ctx.EHLive.time,5);assert.equal(time,5);
 ctx.EHLive.advance(5);await settle();assert.equal(calls.filter(x=>x[0].endsWith('/advance')).length,6);
 await ctx.EHLive.start('a'.repeat(32),'custom',config,2);await settle();assert.equal(time,2);
 fail=true;ctx.EHLive.advance(3);await settle();assert(elements.liveStatus.textContent.includes('Feed stopped'));
 await ctx.EHLive.start('a'.repeat(32),'custom',config,4);await settle();assert.equal(time,4);
 assert.equal(elements.openDashboard.href,'http://127.0.0.1:8001/dashboard/');
 console.log('PASS ordered cutoff transport, idempotency, seek replay, causal payload, visible failure and reconnect');
})().catch(e=>{console.error(e);process.exitCode=1});
