(function(root){
const venue={name:'Event Horizon venue',description:'Schematic reconstruction from your supplied sketch. Dimensions and heights are assumed.',scale:.1,
areas:[
{type:'entrance',name:'Main entrance',x:310,y:55,w:135,d:145,h:2,color:'#c8d4b1'},
{type:'connector',name:'Entry connection',x:445,y:125,w:80,d:75,h:2,color:'#c8d4b1'},
{type:'walkway',name:'North walkway',x:190,y:200,w:545,d:90,h:1,color:'#d3d9c5'},
{type:'walkway',name:'West walkway',x:190,y:290,w:90,d:360,h:1,color:'#d3d9c5'},
{type:'walkway',name:'South walkway',x:280,y:575,w:455,d:75,h:1,color:'#d3d9c5'},
{type:'walkway',name:'East walkway',x:645,y:290,w:90,d:285,h:1,color:'#d3d9c5'},
{type:'courtyard',name:'Vendor courtyard',x:280,y:290,w:365,d:285,h:.5,color:'#e7ddc5'},
{type:'annex',name:'Eastern annex',x:735,y:435,w:230,d:215,h:1,color:'#ccd6bc'},
{type:'pond',name:'Pond',x:415,y:410,w:78,d:70,h:2,color:'#7fbdc7'},
{type:'gate',name:'Exit · north-west',x:305,y:55,w:60,d:12,h:4,color:'#acc989'},
{type:'gate',name:'Exit · south-west',x:180,y:590,w:16,d:50,h:4,color:'#acc989'},
{type:'gate',name:'Exit · east',x:958,y:555,w:14,d:50,h:4,color:'#acc989'}],
stalls:[{x:300,y:310,w:42,d:30,h:24},{x:365,y:310,w:42,d:30,h:24},{x:430,y:310,w:42,d:30,h:24},{x:495,y:310,w:42,d:30,h:24},{x:560,y:310,w:42,d:30,h:24},{x:300,y:525,w:42,d:30,h:24},{x:365,y:525,w:42,d:30,h:24},{x:430,y:525,w:42,d:30,h:24},{x:495,y:525,w:42,d:30,h:24},{x:560,y:525,w:42,d:30,h:24}],
trees:[{x:155,y:270},{x:145,y:380},{x:145,y:480},{x:260,y:690},{x:450,y:690},{x:630,y:690},{x:790,y:400},{x:860,y:400},{x:990,y:475}]
};
function drawPlan(ctx){for(const a of venue.areas){ctx.fillStyle=a.color;ctx.fillRect(a.x,a.y,a.w,a.d);ctx.strokeStyle='#98a48c';ctx.lineWidth=1;ctx.strokeRect(a.x,a.y,a.w,a.d)}for(const s of venue.stalls){ctx.fillStyle='#bf957c';ctx.fillRect(s.x,s.y,s.w,s.d);ctx.fillStyle='#7b6851';ctx.fillRect(s.x,s.y,s.w,4)}for(const t of venue.trees){ctx.beginPath();ctx.arc(t.x,t.y,14,0,Math.PI*2);ctx.fillStyle='#a3b68c';ctx.fill()}ctx.font='11px monospace';ctx.fillStyle='#4b5c41';ctx.fillText('MAIN ENTRANCE',305,40);ctx.fillText('VENDOR COURTYARD',343,374);ctx.fillStyle='#306876';ctx.fillText('POND',438,453);ctx.fillStyle='#4b5c41';ctx.fillText('EASTERN ANNEX',765,495);ctx.fillText('EXIT',128,620);ctx.fillText('EXIT',940,545)}
root.EHVenue={venue,drawPlan};if(typeof module!=='undefined')module.exports={venue,drawPlan};
})(globalThis);
