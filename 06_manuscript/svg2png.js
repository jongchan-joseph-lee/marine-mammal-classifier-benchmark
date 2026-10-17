const { chromium } = require('playwright'); const fs = require('fs');
(async () => { const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' }).catch(() => chromium.launch());
  const p = await b.newPage({ viewport: { width: 624, height: 504 }, deviceScaleFactor: 4 });
  await p.setContent('<body style="margin:0;background:#fff">' + fs.readFileSync('fig/fig1.svg', 'utf8').replace(/<\?xml[^>]*>|<!DOCTYPE[^>]*>/g, '').replace('width="468pt" height="378pt"', 'width="624" height="504"') + '</body>');
  await p.screenshot({ path: 'fig/fig1.png' }); await b.close(); })();
