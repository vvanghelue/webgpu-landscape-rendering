// Shared TSL building blocks: global uniforms, world sampling, procedural tree placement, lighting, sky and fog.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, texture, vec2, vec3, vec4, float, fract, dot, floor, max, min, mix, clamp, exp, pow,
	normalize, length, smoothstep, step, select, cameraPosition, abs, sqrt,
} from 'three/tsl';
import { WORLD_SIZE, TEXEL, TREE_CELL } from '../world/config.js';

// ---- global uniforms -------------------------------------------------------------------------------------------
export const sunDir = uniform( new THREE.Vector3( 0.5, 0.4, 0.3 ).normalize() ); // points towards the sun
export const sunColor = uniform( new THREE.Color( 1.0, 0.92, 0.8 ).multiplyScalar( 3.2 ) );
export const skyZenith = uniform( new THREE.Color( 0.16, 0.34, 0.72 ) );
export const skyHorizon = uniform( new THREE.Color( 0.62, 0.74, 0.88 ) );
export const ambientSky = uniform( new THREE.Color( 0.32, 0.42, 0.6 ) );
export const ambientGround = uniform( new THREE.Color( 0.16, 0.14, 0.11 ) );
export const fogDensity = uniform( 1 / 40000 );
export const fogFalloff = uniform( 1 / 900 );
export const nearTreeRadius = uniform( 180 );
export const treeDrawDistance = uniform( 20000 );
export const treeTierRange0 = uniform( 2000 );
export const impostorFarDistance = uniform( 600 ); // beyond: 1-triangle conifer cards
export const treeShadowDistance = uniform( 1000 );

// ---- world textures (bound in initShading) ---------------------------------------------------------------------
export const G = { heightTexture: null, densityTexture: null, shadowTexture: null, normalTexture: null };

export function initShading( world, shadowTexture, normalTexture ) {

	G.normalTexture = normalTexture;

	G.heightTexture = world.heightTexture;
	G.densityTexture = world.densityTexture;
	G.shadowTexture = shadowTexture;

}

// Order-independent smoothstep (WGSL/Metal smoothstep is undefined when edge0 > edge1).
export const ss = ( a, b, x ) => {

	const t = clamp( float( x ).sub( a ).div( float( b ).sub( a ) ), 0.0, 1.0 );
	return t.mul( t ).mul( float( 3.0 ).sub( t.mul( 2.0 ) ) );

};

export const worldUV = ( xz ) => xz.div( WORLD_SIZE ).add( 0.5 );

// Vertex/compute safe samplers (explicit LOD 0).
export const heightAt = ( xz ) => texture( G.heightTexture, worldUV( xz ) ).level( 0 ).r;
export const densityAt = ( xz ) => texture( G.densityTexture, worldUV( xz ) ).level( 0 ).r;
export const shadowAt = ( xz ) => texture( G.shadowTexture, worldUV( xz ) ).level( 0 );

export const terrainNormal = Fn( ( [ xz ] ) => {

	const e = float( TEXEL );
	const hL = heightAt( xz.sub( vec2( e, 0 ) ) );
	const hR = heightAt( xz.add( vec2( e, 0 ) ) );
	const hD = heightAt( xz.sub( vec2( 0, e ) ) );
	const hU = heightAt( xz.add( vec2( 0, e ) ) );
	return normalize( vec3( hL.sub( hR ), e.mul( 2 ), hD.sub( hU ) ) );

} );

// ---- hashing / procedural tree placement -----------------------------------------------------------------------
// "Hash without sine" (Dave Hoskins) – stable across vertex/fragment/compute stages.
export const hash22 = Fn( ( [ p ] ) => {

	const p3 = fract( vec3( p.x, p.y, p.x ).mul( vec3( 0.1031, 0.1030, 0.0973 ) ) ).toVar();
	p3.addAssign( dot( p3, p3.yzx.add( 33.33 ) ) );
	return fract( p3.xx.add( p3.yz ).mul( p3.zy ) );

} );

// Every TREE_CELL² cell may hold one tree. Returns vec4(worldX, worldZ, height (0 = no tree), random).
// Cheap 2D value noise in [0, 1] (4 hashes) – much lighter than Perlin for per-pixel albedo variation.
export const valueNoise = Fn( ( [ p ] ) => {

	const i = floor( p );
	const f = fract( p );
	const u = f.mul( f ).mul( float( 3.0 ).sub( f.mul( 2.0 ) ) );
	const a = hash22( i ).x;
	const b = hash22( i.add( vec2( 1, 0 ) ) ).x;
	const c = hash22( i.add( vec2( 0, 1 ) ) ).x;
	const d = hash22( i.add( vec2( 1, 1 ) ) ).x;
	return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );

} );

export const treeAt = Fn( ( [ cell ] ) => {

	const r1 = hash22( cell );
	const r2 = hash22( cell.add( vec2( 37.17, 113.91 ) ) );
	const pos = cell.add( r1.mul( 0.8 ).add( 0.1 ) ).mul( TREE_CELL );
	const dens = densityAt( pos );
	const present = step( r2.x, dens.mul( 0.999 ) );
	const h = mix( 9.0, 21.0, r2.y.mul( r2.y ) ).mul( mix( 0.75, 1.0, dens ) );
	return vec4( pos, h.mul( present ), fract( r1.x.mul( 13.17 ).add( r2.y.mul( 7.31 ) ) ) );

} );

export const treeIsBroadleaf = ( rnd ) => step( rnd, 0.22 ); // ~22% broadleaf, rest conifers

// ---- lighting ---------------------------------------------------------------------------------------------------
export const shade = Fn( ( [ albedo, N, shadow, ao ] ) => {

	const ndl = max( dot( N, sunDir ), 0.0 );
	const direct = sunColor.mul( ndl.mul( shadow ) );
	const amb = mix( ambientGround, ambientSky, N.y.mul( 0.5 ).add( 0.5 ) ).mul( ao );
	return albedo.mul( direct.add( amb ) );

} );

// ---- sky + aerial perspective ----------------------------------------------------------------------------------
export const skyColor = Fn( ( [ dir, withSun ] ) => {

	const up = max( dir.y, 0.0 );
	const base = mix( skyHorizon, skyZenith, pow( up, 0.45 ) ).toVar();
	const below = ss( 0.0, - 0.25, dir.y );
	base.assign( mix( base, skyHorizon.mul( 0.55 ), below ) );
	const mu = max( dot( dir, sunDir ), 0.0 );
	const glow = pow( mu, 8.0 ).mul( 0.35 ).add( pow( mu, 64.0 ).mul( 0.6 ) );
	const disc = smoothstep( 0.99993, 0.99997, mu ).mul( 40.0 ).mul( withSun );
	const sunTint = vec3( 1.0, 0.85, 0.6 );
	// Warm the horizon when the sun is low.
	const lowSun = ss( 0.35, 0.0, sunDir.y );
	const horizonWarm = sunTint.mul( lowSun.mul( 0.4 ).mul( pow( mu, 3.0 ) ).mul( ss( 0.4, 0.0, up ) ) );
	return base.add( sunTint.mul( glow.add( disc ) ) ).add( horizonWarm );

} );

// Returns vec4( fogColor, fogAmount ). Cheap enough per-pixel, but small objects (impostors) evaluate it per-vertex.
export const fogParams = Fn( ( [ wpos ] ) => {

	const v = wpos.sub( cameraPosition );
	const d = length( v );
	const dir = v.div( max( d, 1e-3 ) );
	// Analytic height fog: density(y) = fogDensity * exp(-y * falloff), integrated along the view ray.
	const k = fogFalloff;
	const camY = max( cameraPosition.y, 0.0 );
	const x = k.mul( v.y );
	const integral = select( abs( x ).greaterThan( 1e-4 ), float( 1.0 ).sub( exp( x.negate() ) ).div( x ), float( 1.0 ) );
	const optical = fogDensity.mul( d ).mul( exp( camY.mul( k ).negate() ) ).mul( integral );
	const fogAmt = clamp( float( 1.0 ).sub( exp( optical.negate().mul( 3.0 ) ) ), 0.0, 1.0 );
	const fogCol = skyColor( normalize( vec3( dir.x, max( dir.y, 0.0 ).mul( 0.3 ).add( 0.02 ), dir.z ) ), 0.0 );
	return vec4( fogCol, fogAmt );

} );

export const applyFog = ( color, wpos ) => {

	const f = fogParams( wpos );
	return mix( color, f.xyz, f.w );

};

// Approximate tree canopy shadow onto a ground point: ray-sphere test against the trees of the 2x2 cells around
// the point where the sun ray crosses the average canopy height. Fully procedural – no shadow map, any distance.
export const proceduralTreeShadow = Fn( ( [ p ] ) => {

	const sy = max( sunDir.y, 0.12 );
	const q = p.xz.add( sunDir.xz.div( sy ).mul( 8.0 ) );
	const base = floor( q.div( TREE_CELL ).sub( 0.5 ) ); // the 2x2 cells closest to q
	const vis = float( 1.0 ).toVar();

	for ( let j = 0; j <= 1; j ++ ) {

		for ( let i = 0; i <= 1; i ++ ) {

			const t = treeAt( base.add( vec2( i, j ) ) );
			const c = vec3( t.x, p.y.add( t.z.mul( 0.58 ) ), t.y );
			const v = c.sub( p );
			const tca = dot( v, sunDir );
			const d2 = max( dot( v, v ).sub( tca.mul( tca ) ), 0.0 );
			const r = max( t.z.mul( 0.24 ), 0.05 );
			const occ = float( 1.0 ).sub( smoothstep( r.mul( 0.55 ), r, sqrt( d2 ) ) ).mul( step( 0.0, tca ) ).mul( step( 0.01, t.z ) );
			vis.assign( min( vis, float( 1.0 ).sub( occ.mul( 0.88 ) ) ) );

		}

	}

	return vis;

} );
