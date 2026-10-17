const { chromium } = require(process.env.PW || 'playwright');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' }).catch(() => chromium.launch());
  const errs = [];
  for (const [name, w, scheme, tab] of [['sup', 1280, 'light', 'sup'], ['sup_m', 400, 'dark', 'sup']]) {
    const p = await b.newPage({ viewport: { width: w, height: 900 }, colorScheme: scheme });
    p.on('pageerror', e => errs.push(name + ': ' + e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(name + ': ' + m.text()); });
    await p.goto('file:///home/claude/watkins_maps/real/page.html#' + tab); await p.waitForTimeout(1500);
    const ov = await p.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (ov > 0) errs.push(name + ': horizontal overflow ' + ov);
    await p.screenshot({ path: `real/shot_${name}.png`, fullPage: true }); await p.close();
  }
  console.log(errs.length ? errs.join('\n') : 'no errors'); await b.close();
})();
