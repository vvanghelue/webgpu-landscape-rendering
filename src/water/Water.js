import * as THREE from 'three/webgpu';
import {
	Fn, positionWorld, cameraPosition, time, vec2, vec3, float, sin, cos, dot, normalize, length, max, mix, exp, pow,
	reflect, smoothstep, texture, uniform,
} from 'three/tsl';
import { heightAt, skyColor, applyFog, sunDir, sunColor, ambientSky, worldUV, G, ss } from '../shading/common.js';

// Opaque stylised water: depth-based absorption (from the terrain heightmap), analytic wave normals, Fresnel sky
// reflection incl. sun glint, shore foam, and the baked terrain shadow so mountains shadow lakes and the sea too.

const WAVES = [
	// dirX, dirZ, wavelength, amplitude, speed
	[ 1.0, 0.2, 60, 0.35, 1.0 ], [ 0.7, - 0.7, 31, 0.18, 1.1 ], [ - 0.3, 1.0, 17, 0.09, 1.3 ],
	[ 0.9, 0.5, 9, 0.045, 1.5 ], [ - 0.8, 0.4, 5.3, 0.025, 1.7 ], [ 0.2, - 1.0, 3.1, 0.014, 2.0 ],
];

export function createWaterMaterial( level ) {

	const mat = new THREE.MeshBasicNodeMaterial();
	const waterLevel = uniform( level );

	mat.colorNode = Fn( () => {

		const p = positionWorld;
		const v = p.sub( cameraPosition );
		const dist = length( v );
		const viewDir = v.div( dist );

		// analytic wave slope; each wave fades out once its wavelength gets close to pixel size (no moiré)
		const grad = vec2( 0, 0 ).toVar();
		for ( const [ dx, dz, wl, amp, speed ] of WAVES ) {

			const len = Math.hypot( dx, dz );
			const k = ( 2 * Math.PI ) / wl;
			const d = vec2( dx / len, dz / len );
			const w = Math.sqrt( 9.81 * k ) * speed * 0.5;
			const phase = dot( d, p.xz ).mul( k ).add( time.mul( w ) );
			const fade = ss( wl * 25, wl * 4, dist );
			grad.addAssign( d.mul( cos( phase ).mul( fade.mul( amp * k ) ) ) );

		}

		const N = normalize( vec3( grad.x.negate(), 1.0, grad.y.negate() ) );

		const depth = max( waterLevel.sub( texture( G.heightTexture, worldUV( p.xz ) ).r ), 0.0 ); // implicit LOD: mip-mapped
		const sh = texture( G.shadowTexture, worldUV( p.xz ) ).r;

		const deep = vec3( 0.005, 0.03, 0.06 );
		const shallow = vec3( 0.04, 0.26, 0.28 );
		const light = sunColor.mul( max( sunDir.y, 0.0 ).mul( sh ).mul( 0.25 ) ).add( ambientSky.mul( 0.5 ) );
		const body = mix( deep, shallow, exp( depth.mul( - 0.12 ) ) ).mul( light );
		const sandBottom = vec3( 0.55, 0.5, 0.38 ).mul( light ).mul( 0.6 );
		const bodyCol = mix( body, sandBottom, exp( depth.mul( - 0.9 ) ).mul( 0.8 ) );

		const R = reflect( viewDir, N );
		const refl = skyColor( normalize( vec3( R.x, max( R.y, 0.01 ), R.z ) ), sh );
		const cosT = max( dot( N, viewDir.negate() ), 0.0 );
		const fresnel = float( 0.02 ).add( pow( float( 1.0 ).sub( cosT ), 5.0 ).mul( 0.98 ) );
		const col = mix( bodyCol, refl, fresnel ).toVar();

		const foam = ss( 1.2, 0.0, depth ).mul( sin( depth.mul( 6.0 ).sub( time.mul( 1.5 ) ) ).mul( 0.3 ).add( 0.7 ) );
		col.assign( mix( col, vec3( 0.9, 0.95, 1.0 ).mul( light ), foam.mul( 0.7 ) ) );

		return applyFog( col, p );

	} )();

	return mat;

}

export class Water {

	constructor( world ) {

		this.group = new THREE.Group();

		const sea = new THREE.Mesh( new THREE.PlaneGeometry( 400000, 400000, 8, 8 ).rotateX( - Math.PI / 2 ), createWaterMaterial( 0 ) );
		sea.frustumCulled = false;
		this.group.add( sea );

		const L = world.lake;
		const lake = new THREE.Mesh(
			new THREE.CircleGeometry( L.radius * 1.6, 96 ).rotateX( - Math.PI / 2 ).translate( L.x, L.level, L.z ),
			createWaterMaterial( L.level ),
		);
		this.group.add( lake );

	}

}
