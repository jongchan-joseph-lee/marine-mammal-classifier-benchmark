const { chromium } = require('playwright');
(async()=>{const b=await chromium.launch({});
const p=await b.newPage({viewport:{width:1280,height:900}}); const errs=[]; p.on('console',m=>{if(m.type()==='error')errs.push(m.text())}); p.on('pageerror',e=>errs.push(e.message));
await p.goto('http://localhost:8765/test.html',{waitUntil:'networkidle'});
const tabs=['overview','data','probe','maps','ct6','signal','phys','ref'];
for(const t of tabs){await p.click(`[data-t="${t}"]`); await p.waitForTimeout(600);
  if(t==='maps'){const c=await p.$('#m-canvas'); await c.scrollIntoViewIfNeeded(); const bb=await c.boundingBox(); const pt=await p.evaluate(()=>{const cv=document.querySelector('#m-canvas');const ctx=cv.getContext('2d');const d=ctx.getImageData(0,0,cv.width,cv.height).data;const dpr=window.devicePixelRatio||1;for(let y=Math.floor(cv.height/2);y<cv.height;y+=3)for(let x=0;x<cv.width;x+=3){if(d[(y*cv.width+x)*4+3]>200)return [x/dpr,y/dpr];}return null;}); await p.mouse.click(bb.x+pt[0],bb.y+pt[1]); await p.waitForTimeout(2500); await p.evaluate(()=>window.scrollTo(0,0));
    console.log('insp',(await p.$eval('#m-insp',e=>e.innerText)).slice(0,80));}
  await p.screenshot({path:`s_${t}.png`,fullPage:t!=='maps'});}
// audio fetch test
const res=await p.evaluate(async()=>{const r=await fetch('audio_0.bin'); const b=await r.arrayBuffer(); return b.byteLength;}); console.log('bundle bytes',res);
await p.setViewportSize({width:400,height:800}); await p.click('[data-t="phys"]'); await p.waitForTimeout(500); await p.screenshot({path:'s_phone.png'});
const sw=await p.evaluate(()=>document.documentElement.scrollWidth); console.log('phone scrollWidth',sw);
console.log('errors',errs.slice(0,10)); await b.close();})();
