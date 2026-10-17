// Per-system GPU cost probe: toggles systems in the running app and reports GPU ms (timestamp queries).
// usage: node scripts/perf.mjs [view=forest] [url=http://localhost:5173]
import puppeteer from 'puppeteer-core';

const view = process.argv[ 2 ] || 'forest';
const url = process.argv[ 3 ] || 'http://localhost:5173';
const browser = await puppeteer.launch( {
	executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	headless: 'new',
	args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--window-size=1600,900' ],
	defaultViewport: { width: 1600, height: 900 },
} );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );
await page.goto( `${ url }/?view=${ view }&${ process.argv[ 4 ] || '' }` );
await page.waitForFunction( () => window.__app, { timeout: 60000 } );

const sample = async () => {

	const vals = [];
	for ( let i = 0; i < 6; i ++ ) {

		await new Promise( ( r ) => setTimeout( r, 500 ) );
		vals.push( parseFloat( await page.evaluate( () => document.getElementById( 'hud' ).textContent.match( /GPU ([\d.]+)/ )?.[ 1 ] ) ) );

	}

	vals.sort( ( a, b ) => a - b );
	return vals[ 2 ];

};

const measure = async ( label, setup ) => {

	await page.evaluate( setup );
	await new Promise( ( r ) => setTimeout( r, 1500 ) );
	console.log( label.padEnd( 30 ), ( await sample() ).toFixed( 2 ), 'ms' );

};

await measure( 'all', () => {} );
await measure( 'no procedural tree shadows', () => { __app.uniforms.treeShadowDistance.value = 0; } );
await measure( '+ no near meshes', () => { __app.trees.enabledNear = false; } );
await measure( '+ no impostors', () => { __app.trees.enabledFar = false; } );
await measure( '+ no terrain', () => { __app.terrain.mesh.visible = false; } );
await measure( 'only sky', () => { __app.scene.children.forEach( ( c ) => c.visible = c === __app.scene.children[ 0 ] ); } );
await browser.close();
