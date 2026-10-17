import * as THREE from 'three/webgpu';
import GUI from 'lil-gui';
import * as TSL from 'three/tsl';
import { World } from './world/World.js';
import { Terrain } from './terrain/Terrain.js';
import { ShadowBaker } from './terrain/ShadowBaker.js';
import { Trees } from './vegetation/Trees.js';
import { Water } from './water/Water.js';
import { Sky } from './sky/Sky.js';
import { FlyControls } from './controls/FlyControls.js';
import {
	initShading, sunDir, fogDensity, nearTreeRadius, treeDrawDistance, treeTierRange0, treeShadowDistance,
} from './shading/common.js';
import { TREE_CELL, WORLD_SIZE } from './world/config.js';

const loading = document.getElementById( 'loading' );
const hud = document.getElementById( 'hud' );

async function main() {

	if ( ! navigator.gpu ) {

		loading.textContent = 'WebGPU is not available in this browser.\nUse a recent Chrome / Edge / Safari 26+.';
		return;

	}

	const params = new URLSearchParams( location.search );
	const renderer = new THREE.WebGPURenderer( { antialias: params.get( 'aa' ) !== '0', reversedDepthBuffer: true, powerPreference: 'high-performance', trackTimestamp: true } );
	renderer.setPixelRatio( Math.min( window.devicePixelRatio, 1.5 ) );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 0.9;
	document.body.appendChild( renderer.domElement );
	await renderer.init();

	const t0 = performance.now();
	const world = await World.generate( ( p ) => loading.textContent = `Generating island… ${ Math.round( p * 100 ) }%` );
	console.log( `world generated in ${ ( performance.now() - t0 ).toFixed( 0 ) } ms, ~${ ( world.expectedTrees / 1e6 ).toFixed( 2 ) }M trees, lake level ${ world.lakeLevel.toFixed( 1 ) } m` );

	const shadows = new ShadowBaker( 2048 );
	initShading( world, shadows.texture, shadows.normalTexture );
	shadows.build();

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 1, 300000 );

	const sky = new Sky();
	const terrain = new Terrain( world );
	const trees = new Trees( world );
	const water = new Water( world );
	// Draw order front-to-back-ish so early-z rejects hidden terrain/water fragments.
	trees.nearMeshes.forEach( ( m ) => m.renderOrder = 0 );
	trees.impostors.forEach( ( m ) => m.renderOrder = 1 );
	terrain.mesh.renderOrder = 2;
	water.group.children.forEach( ( m ) => m.renderOrder = 3 );
	scene.add( sky.mesh, terrain.mesh, trees.group, water.group );

	const controls = new FlyControls( camera, renderer.domElement, world );

	// ---------------------------------------------------------------------------------------------- views
	const findForest = () => {

		const N = world.res;
		let best = null;
		for ( let k = 0; k < 4000; k ++ ) {

			const i = ( k * 7919 ) % N, j = ( k * 104729 ) % N;
			const d = world.density[ j * N + i ];
			const x = ( i / N - 0.5 ) * WORLD_SIZE, z = ( j / N - 0.5 ) * WORLD_SIZE;
			if ( d > 200 && Math.hypot( x, z ) < 5000 && ( ! best || world.heights[ j * N + i ] < best.h ) ) best = { x, z, h: world.heights[ j * N + i ] };

		}

		return best || { x: 0, z: 0, h: world.heightAt( 0, 0 ) };

	};

	const views = {
		overview: () => controls.setView( new THREE.Vector3( 0, 4200, 12500 ), 0, - 0.38 ),
		highAltitude: () => controls.setView( new THREE.Vector3( - 5000, 9000, 12000 ), - 0.35, - 0.7 ),
		forest: () => {

			const f = findForest();
			controls.setView( new THREE.Vector3( f.x, f.h + 25, f.z ), 0.6, - 0.12 );

		},
		lake: () => {

			const L = world.lake;
			// hover over the lake near its shore, looking across it
			const x = L.x + L.radius * 0.7, z = L.z + L.radius * 0.45;
			const yaw = Math.atan2( - ( L.x - x ), - ( L.z - z ) );
			controls.setView( new THREE.Vector3( x, L.level + 30, z ), yaw, - 0.06 );

		},
		coast: () => controls.setView( new THREE.Vector3( 5200, 140, 7200 ), Math.atan2( 5200, 7200 ) + 0.25, - 0.06 ),
	};

	( views[ params.get( 'view' ) ] || views.overview )();

	// ---------------------------------------------------------------------------------------------- GUI
	const settings = {
		sunAzimuth: 135,
		sunElevation: 22,
		lodRange: terrain.lodRange0.value,
		wireframe: false,
		nearTrees: true,
		farTrees: true,
		nearRadius: nearTreeRadius.value,
		tierRange: treeTierRange0.value,
		treeDistance: treeDrawDistance.value,
		treeShadowDistance: treeShadowDistance.value,
		fog: fogDensity.value * 1e5,
		autoSun: false,
	};

	const updateSun = () => {

		const az = THREE.MathUtils.degToRad( settings.sunAzimuth ), el = THREE.MathUtils.degToRad( settings.sunElevation );
		sunDir.value.set( Math.cos( el ) * Math.sin( az ), Math.sin( el ), Math.cos( el ) * Math.cos( az ) );
		shadows.dirty = true;

	};

	if ( params.has( 'el' ) ) settings.sunElevation = parseFloat( params.get( 'el' ) );
	if ( params.has( 'az' ) ) settings.sunAzimuth = parseFloat( params.get( 'az' ) );
	updateSun();

	const gui = new GUI();
	const fSun = gui.addFolder( 'Sun' );
	fSun.add( settings, 'sunAzimuth', 0, 360, 1 ).onChange( updateSun );
	fSun.add( settings, 'sunElevation', 1, 85, 0.5 ).onChange( updateSun );
	fSun.add( settings, 'autoSun' ).name( 'animate sun' );
	const fT = gui.addFolder( 'Terrain' );
	fT.add( settings, 'lodRange', 200, 2000, 10 ).name( 'LOD range (m)' ).onChange( ( v ) => terrain.lodRange0.value = v );
	fT.add( settings, 'wireframe' ).onChange( ( v ) => terrain.material.wireframe = v );
	fT.add( settings, 'fog', 0, 20, 0.1 ).name( 'fog' ).onChange( ( v ) => fogDensity.value = v / 1e5 );
	const fTr = gui.addFolder( 'Trees' );
	fTr.add( settings, 'nearTrees' ).name( 'mesh trees' ).onChange( ( v ) => trees.enabledNear = v );
	fTr.add( settings, 'farTrees' ).name( 'impostors' ).onChange( ( v ) => trees.enabledFar = v );
	fTr.add( settings, 'nearRadius', 40, 300, 5 ).name( 'mesh radius (m)' ).onChange( ( v ) => nearTreeRadius.value = v );
	fTr.add( settings, 'tierRange', 500, 6000, 50 ).name( 'full density to (m)' ).onChange( ( v ) => treeTierRange0.value = v );
	fTr.add( settings, 'treeDistance', 1000, 20000, 100 ).name( 'draw distance (m)' ).onChange( ( v ) => treeDrawDistance.value = v );
	fTr.add( settings, 'treeShadowDistance', 0, 4000, 50 ).name( 'tree shadow dist (m)' ).onChange( ( v ) => treeShadowDistance.value = v );
	const fV = gui.addFolder( 'Views' );
	for ( const name of Object.keys( views ) ) fV.add( views, name );

	// ---------------------------------------------------------------------------------------------- loop
	window.addEventListener( 'resize', () => {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( window.innerWidth, window.innerHeight );

	} );

	loading.remove();

	const clock = new THREE.Timer();
	let frames = 0, acc = 0, fps = 0, ms = 0, hudTimer = 0, gpuMs = 0, gpuPending = false;

	renderer.setAnimationLoop( () => {

		clock.update();
		const dt = Math.min( clock.getDelta(), 0.1 );

		if ( settings.autoSun ) {

			settings.sunAzimuth = ( settings.sunAzimuth + dt * 6 ) % 360;
			updateSun();
			gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

		}

		controls.update( dt );
		sky.update( camera );
		terrain.update( camera );
		trees.update( camera, renderer );
		shadows.update( renderer );
		renderer.render( scene, camera );

		if ( renderer.backend.trackTimestamp && ! gpuPending ) {

			gpuPending = true;
			renderer.resolveTimestampsAsync( 'render' ).then( ( d ) => {

				if ( d ) gpuMs = gpuMs * 0.9 + d * 0.1;
				gpuPending = false;

			} );

		}

		frames ++; acc += dt; hudTimer += dt;
		if ( hudTimer > 0.5 ) {

			fps = frames / acc; ms = 1000 * acc / frames; frames = 0; acc = 0; hudTimer = 0;
			const info = renderer.info.render;
			hud.textContent = [
				`${ fps.toFixed( 0 ) } fps  ${ ms.toFixed( 2 ) } ms   GPU ${ gpuMs ? gpuMs.toFixed( 2 ) + ' ms' : 'n/a' }`,
				`direct tris    ${ ( info.triangles / 1e6 ).toFixed( 2 ) } M (+ indirect trees)`,
				`terrain patches ${ terrain.stats.patches }`,
				`tree tiles     ${ trees.stats.tiles }  (${ ( trees.stats.cells / 1e6 ).toFixed( 2 ) } M cells culled on GPU)`,
				`visible trees  ${ ( trees.stats.visibleImpostors / 1e6 ).toFixed( 3 ) } M impostors + ${ trees.stats.visibleMeshes } meshes`,
				`trees on map   ~${ ( world.expectedTrees / 1e6 ).toFixed( 2 ) } M (${ TREE_CELL } m grid)`,
				`camera         ${ camera.position.x.toFixed( 0 ) }, ${ camera.position.y.toFixed( 0 ) }, ${ camera.position.z.toFixed( 0 ) }  speed ${ controls.speed.toFixed( 0 ) } m/s`,
			].join( '\n' );

		}

	} );

	window.__app = { THREE, TSL, uniforms: { treeShadowDistance, fogDensity, treeTierRange0, nearTreeRadius }, renderer, scene, camera, world, terrain, trees, shadows, controls, views, settings, updateSun };

}

main().catch( ( e ) => {

	console.error( e );
	loading.textContent = 'Error: ' + e.message;

} );
