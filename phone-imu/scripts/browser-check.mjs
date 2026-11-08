import {chromium} from 'playwright';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
const config=JSON.parse(await readFile(process.env.IMU_CONFIG||new URL('../.local/config.json',import.meta.url),'utf8'));
const base=`https://localhost:${config.httpsPort}`;
const browser=process.env.IMU_BROWSER_CDP?await chromium.connectOverCDP(process.env.IMU_BROWSER_CDP):await chromium.launch({headless:true});
const contexts=[],errors=[];
try{
  const dashboardContext=await browser.newContext({ignoreHTTPSErrors:true});contexts.push(dashboardContext);
  const dashboard=await dashboardContext.newPage();dashboard.on('pageerror',e=>errors.push(e.message));
  await dashboard.goto(`${base}/#token=${config.token}`);
  assert.equal(await dashboard.title(),'IMU • Recording console');
  const phones=[];
  for(let i=1;i<=2;i++){
    const context=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:390,height:844}});contexts.push(context);
    await context.addInitScript(()=>{
      Object.defineProperty(DeviceMotionEvent,'requestPermission',{value:async()=> 'granted'});
      let n=0,last=0;
      // Synthetic events exercise the exact production browser path, not iPhone hardware.
      window.syntheticTimer=setInterval(()=>{const event=new DeviceMotionEvent('devicemotion',{
        accelerationIncludingGravity:{x:Math.sin(performance.now()/100),y:0,z:9.81},
        rotationRate:{alpha:4,beta:2,gamma:3},interval:10});
        if(++n%30===0)Object.defineProperty(event,'timeStamp',{value:last});
        last=event.timeStamp;window.dispatchEvent(event);},10);
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`${base}/phone#token=${config.token}`);await page.selectOption('#device',`phone_${i}`);
    await page.click('#enable');phones.push({page,context});
  }
  await dashboard.waitForFunction(()=>!document.getElementById('start').disabled,{},{timeout:15000});
  await dashboard.click('#start');await dashboard.waitForFunction(()=>document.getElementById('duration').textContent.includes('recording'),{},{timeout:10000});
  await phones[1].context.setOffline(true);await phones[1].page.waitForTimeout(800);
  await phones[1].context.setOffline(false);await dashboard.waitForTimeout(2500);
  await dashboard.click('#stop');
  await dashboard.waitForFunction(()=>document.getElementById('export-status').textContent.includes('Both phone'),{},{timeout:20000});
  const downloadPromise=dashboard.waitForEvent('download');await dashboard.getByRole('button',{name:'combined.csv',exact:true}).click();
  const download=await downloadPromise;await mkdir(new URL('../.local/browser-check/',import.meta.url),{recursive:true});
  await download.saveAs(fileURLToPath(new URL('../.local/browser-check/combined.csv',import.meta.url)));
  const firstSession=await dashboard.locator('#session-list').inputValue();
  const firstCSV=await readFile('.local/browser-check/combined.csv','utf8');assert.match(firstCSV,/,duplicate/);
  for(let cycle=0;cycle<2;cycle++){
    await dashboard.waitForFunction(()=>!document.getElementById('start').disabled);
    await dashboard.click('#start');
    await dashboard.waitForFunction(()=>document.getElementById('duration').textContent.startsWith('recording'),{},{timeout:10000});
    await dashboard.waitForTimeout(600);await dashboard.click('#stop');
    await dashboard.waitForFunction(()=>document.getElementById('export-status').textContent.includes('Session saved'),{},{timeout:15000});
  }
  assert.equal(await dashboard.locator('#session-list option').count(),3);
  await dashboard.selectOption('#session-list',firstSession);
  const oldDownloadPromise=dashboard.waitForEvent('download');await dashboard.getByRole('button',{name:'combined.csv',exact:true}).click();
  await (await oldDownloadPromise).saveAs('.local/browser-check/first-session-again.csv');
  assert.equal(await readFile('.local/browser-check/first-session-again.csv','utf8'),firstCSV);
  await dashboard.screenshot({path:'.local/browser-check/dashboard.png',fullPage:true});
  await phones[0].page.screenshot({path:'.local/browser-check/phone.png',fullPage:true});
  assert.deepEqual(errors,[]);
  const report={passed:true,synthetic_browser_events:true,phones:2,offline_recovery:true,csv_download:true,repeated_sessions:3,
    duplicate_timestamps_preserved:true,older_session_download:true,console_errors:errors};
  await writeFile('.local/browser-check/result.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{for(const c of contexts)await c.close();await browser.close();}
