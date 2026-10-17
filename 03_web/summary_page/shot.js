const { chromium } = require(process.env.PW);
(async () => { const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' }).catch(() => chromium.launch()); const errs = [];
  for (const [name, w, scheme] of [['light', 1100, 'light'], ['phone', 400, 'dark']]) { const p = await b.newPage({ viewport: { width: w, height: 900 }, colorScheme: scheme });
    p.on('pageerror', e => errs.push(name + ': ' + e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(name + ': ' + m.text()); });
    await p.goto('file:///home/claude/summary/page.html'); await p.waitForTimeout(1500);
    const ov = await p.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth); if (ov > 0) errs.push(name + ': overflow ' + ov);
    await p.screenshot({ path: `shot_${name}.png`, fullPage: true });
    if (name === 'light') for (const id of ['figA','figB','figC','figD','figE']) await (await p.$('#' + id)).screenshot({ path: `shot_${id}.png` });
    await p.close(); }
  console.log(errs.length ? errs.join('\n') : 'no errors'); await b.close(); })();
