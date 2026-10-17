import * as THREE from 'three/webgpu';
import { smoothstep } from './noise.js';
import { WORLD_SIZE, HEIGHTMAP_RES, TEXEL, LAKE } from './config.js';

// Holds the CPU-side terrain data (heights, tree density, min/max pyramid) and the matching GPU textures.
export class World {

	static async generate( onProgress = () => {} ) {

		const N = HEIGHTMAP_RES;
		const heights = new Float32Array( N * N );
		const forest = new Float32Array( N * N );
		const workers = Math.min( 8, Math.max( 2, navigator.hardwareConcurrency || 4 ) );
		const rowsPer = Math.ceil( N / workers );
		let done = 0, lakeLevel = 0;

		await Promise.all( Array.from( { length: workers }, ( _, w ) => new Promise( ( resolve, reject ) => {

			const worker = new Worker( new URL( './generator.worker.js', import.meta.url ), { type: 'module' } );
			worker.onerror = reject;
			worker.onmessage = ( e ) => {

				const { rowStart, heights: h, forest: f } = e.data;
				heights.set( h, rowStart * N );
				forest.set( f, rowStart * N );
				lakeLevel = e.data.lakeLevel;
				worker.terminate();
				onProgress( ++ done / workers );
				resolve();

			};

			worker.postMessage( { rowStart: w * rowsPer, rowEnd: Math.min( N, ( w + 1 ) * rowsPer ) } );

		} ) ) );

		return new World( heights, forest, lakeLevel );

	}

	constructor( heights, forest, lakeLevel ) {

		this.size = WORLD_SIZE;
		this.res = HEIGHTMAP_RES;
		this.heights = heights;
		this.lakeLevel = lakeLevel;
		this.lake = { ...LAKE, level: lakeLevel };

		this.density = this.buildDensity( forest );
		this.buildPyramid();

		const heightTexture = new THREE.DataTexture( heights, this.res, this.res, THREE.RedFormat, THREE.FloatType );
		heightTexture.magFilter = THREE.LinearFilter;
		heightTexture.minFilter = THREE.LinearMipmapLinearFilter;
		heightTexture.generateMipmaps = true;
		heightTexture.wrapS = heightTexture.wrapT = THREE.ClampToEdgeWrapping;
		heightTexture.needsUpdate = true;
		this.heightTexture = heightTexture;

		const densityTexture = new THREE.DataTexture( this.density, this.res, this.res, THREE.RedFormat, THREE.UnsignedByteType );
		densityTexture.magFilter = THREE.LinearFilter;
		densityTexture.minFilter = THREE.LinearMipmapLinearFilter;
		densityTexture.generateMipmaps = true;
		densityTexture.wrapS = densityTexture.wrapT = THREE.ClampToEdgeWrapping;
		densityTexture.needsUpdate = true;
		this.densityTexture = densityTexture;

	}

	buildDensity( forest ) {

		const N = this.res, h = this.heights, L = this.lake;
		const out = new Uint8Array( N * N );
		let sum = 0;

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const o = j * N + i;
				const hx = h[ j * N + Math.min( N - 1, i + 1 ) ] - h[ j * N + Math.max( 0, i - 1 ) ];
				const hz = h[ Math.min( N - 1, j + 1 ) * N + i ] - h[ Math.max( 0, j - 1 ) * N + i ];
				const slope = Math.hypot( hx, hz ) / ( 2 * TEXEL );
				const y = h[ o ];

				const env = ( 1 - smoothstep( 0.45, 0.85, slope ) ) * smoothstep( 1.5, 7, y ) * ( 1 - smoothstep( 520, 780, y ) );
				let d = smoothstep( - 0.3, 0.12, forest[ o ] ) * 0.95 * env;
				d = Math.max( d, 0.08 * env ); // scattered trees in meadows

				const x = ( ( i + 0.5 ) / N - 0.5 ) * this.size, z = ( ( j + 0.5 ) / N - 0.5 ) * this.size;
				if ( Math.hypot( x - L.x, z - L.z ) < L.radius * 1.6 && y < L.level + 2.5 ) d = 0;

				out[ o ] = Math.round( d * 255 );
				sum += d;

			}

		}

		this.expectedTrees = Math.round( sum * ( this.size * this.size ) / ( N * N ) / 16 ); // 4x4 m cells
		return out;

	}

	// Min/max mip pyramid for terrain/tree tile bounding boxes. Level k has (res >> k)² entries.
	buildPyramid() {

		const levels = [ { size: this.res, min: this.heights, max: this.heights } ];
		let size = this.res;
		while ( size > 1 ) {

			const prev = levels[ levels.length - 1 ];
			const s = size >> 1;
			const min = new Float32Array( s * s ), max = new Float32Array( s * s );
			for ( let j = 0; j < s; j ++ ) for ( let i = 0; i < s; i ++ ) {

				const a = ( 2 * j ) * size + 2 * i, b = a + size;
				min[ j * s + i ] = Math.min( prev.min[ a ], prev.min[ a + 1 ], prev.min[ b ], prev.min[ b + 1 ] );
				max[ j * s + i ] = Math.max( prev.max[ a ], prev.max[ a + 1 ], prev.max[ b ], prev.max[ b + 1 ] );

			}

			levels.push( { size: s, min, max } );
			size = s;

		}

		this.pyramid = levels;

	}

	// Min/max height of an axis-aligned world-space square (x0,z0,size). Size must be a power-of-two multiple of TEXEL.
	rangeOf( x0, z0, size, out ) {

		const texels = Math.max( 1, size / TEXEL );
		const k = Math.min( this.pyramid.length - 1, Math.round( Math.log2( texels ) ) );
		const lvl = this.pyramid[ k ];
		const i = Math.floor( ( x0 / this.size + 0.5 ) * lvl.size );
		const j = Math.floor( ( z0 / this.size + 0.5 ) * lvl.size );
		if ( i < 0 || j < 0 || i >= lvl.size || j >= lvl.size ) {

			out.min = - 130; out.max = 0;
			return out;

		}

		out.min = lvl.min[ j * lvl.size + i ] - 2;
		out.max = lvl.max[ j * lvl.size + i ] + 2;
		return out;

	}

	heightAt( x, z ) {

		const N = this.res;
		const fx = Math.min( N - 1, Math.max( 0, ( x / this.size + 0.5 ) * N - 0.5 ) );
		const fz = Math.min( N - 1, Math.max( 0, ( z / this.size + 0.5 ) * N - 0.5 ) );
		const i = Math.min( N - 2, Math.floor( fx ) ), j = Math.min( N - 2, Math.floor( fz ) );
		const tx = fx - i, tz = fz - j;
		const h = this.heights;
		const a = h[ j * N + i ], b = h[ j * N + i + 1 ], c = h[ ( j + 1 ) * N + i ], d = h[ ( j + 1 ) * N + i + 1 ];
		return ( a * ( 1 - tx ) + b * tx ) * ( 1 - tz ) + ( c * ( 1 - tx ) + d * tx ) * tz;

	}

}
