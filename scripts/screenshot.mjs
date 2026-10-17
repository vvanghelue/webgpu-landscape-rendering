// Headless smoke test: loads the app in Chrome (WebGPU), prints console errors + FPS, and saves screenshots.
// usage: node scripts/screenshot.mjs [url=http://localhost:5173] [views=overview,forest] [outDir=tmp]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const url = process.argv[ 2 ] || 'http://localhost:5173';
const views = ( process.argv[ 3 ] || 'overview' ).split( ',' );
const out = process.argv[ 4 ] || 'tmp';
fs.mkdirSync( out, { recursive: true } );

const browser = await puppeteer.launch( {
	executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	headless: 'new',
	args: [ '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=metal', '--ignore-gpu-blocklist', '--window-size=1600,900' ],
	defaultViewport: { width: 1600, height: 900 },
} );

const page = await browser.newPage();
page.on( 'console', ( m ) => console.log( `[${ m.type() }]`, m.text().slice( 0, 2000 ) ) );
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );

for ( const view of views ) {

	const [ name, query = '' ] = view.split( '?' );
	await page.goto( `${ url }/?view=${ name }&${ query }`, { waitUntil: 'load' } );
	await page.waitForFunction( () => window.__app || document.getElementById( 'loading' )?.textContent.startsWith( 'Error' ), { timeout: 60000 } );
	await new Promise( ( r ) => setTimeout( r, 4000 ) );
	console.log( view, '|', await page.$eval( '#hud', ( e ) => e.textContent.replace( /\n/g, ' | ' ) ).catch( () => 'no hud' ) );
	await page.screenshot( { path: `${ out }/${ name }${ query ? '-' + query.replace( /[^a-z0-9]/gi, '_' ) : '' }.png` } );

}

await browser.close();
