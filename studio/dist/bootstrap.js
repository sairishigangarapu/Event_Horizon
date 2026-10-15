(function(){
function report(message){let banner=document.getElementById('bootError');if(!banner){banner=document.createElement('div');banner.id='bootError';banner.setAttribute('role','alert');banner.style.cssText='position:fixed;top:0;left:0;right:0;z-index:100;background:#923b28;color:white;padding:18px;font:13px Arial';document.body.prepend(banner)}banner.textContent=message}
window.addEventListener('error',e=>{if(e.target?.tagName==='SCRIPT')report('A studio file did not load. Extract the entire ZIP and run START_WINDOWS.bat.');else if(e.message)report('Studio error: '+e.message+' — reload using the printed local server URL.')},true);
if(window.location.protocol==='file:')report('Run START_WINDOWS.bat and open http://127.0.0.1:8000. Opening index.html directly is unsupported.');
setTimeout(()=>{if(!document.body.dataset.studioReady)report('Studio did not start. Extract all files and launch START_WINDOWS.bat. Check that engine.js and app.js are inside dist.')},4000);
})();
