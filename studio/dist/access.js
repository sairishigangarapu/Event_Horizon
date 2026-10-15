/* Optional API-key mode. Credentials stay in session storage and never in URLs. */
(()=>{'use strict';
const nativeFetch=window.fetch.bind(window),storageKey='event_horizon_api_key';let key='';try{key=sessionStorage.getItem(storageKey)||'';}catch{}
let button;
window.fetch=async(input,init)=>{const url=new URL(typeof input==='string'||input instanceof URL?input:input.url,location.href);let options=init;
 if(key&&url.origin===location.origin&&url.pathname.startsWith('/api/')){const headers=new Headers(init?.headers||(input instanceof Request?input.headers:undefined));headers.set('X-API-Key',key);options={...init,headers};}
 const response=await nativeFetch(input,options);if(response.status===401&&button)button.textContent='API key needed';return response;};
function install(){const header=document.querySelector('header');if(!header)return;button=document.createElement('button');button.type='button';button.textContent=key?'API key set':'API access';button.title='Optional access key for a protected local or hosted API';button.onclick=()=>{const next=prompt('API key for this session (leave empty to clear):','');if(next===null)return;key=next.trim();try{if(key)sessionStorage.setItem(storageKey,key);else sessionStorage.removeItem(storageKey);}catch{}button.textContent=key?'API key set':'API access';};header.append(button);}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',install);else install();
})();
