const { chromium } = require('playwright');
(async()=>{const b=await chromium.launch({});const p=await b.newPage({viewport:{width:1280,height:900}});const errs=[];p.on('pageerror',e=>errs.push(e.message));
await p.goto('http://localhost:8765/test.html',{waitUntil:'networkidle'});await p.click('[data-t="check2"]');await p.waitForTimeout(500);await p.screenshot({path:'s_check2.png',fullPage:true});
await p.setViewportSize({width:400,height:800});const sw=await p.evaluate(()=>document.documentElement.scrollWidth);console.log('sw',sw,'errs',errs);await b.close();})();
