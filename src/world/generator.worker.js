// Generates a band of rows of the island heightmap + forest mask.
import { createNoise, smoothstep } from './noise.js';
import { WORLD_SIZE, HEIGHTMAP_RES, SEED, LAKE } from './config.js';

const { fbm, ridged } = createNoise( SEED );

function rawHeight( x, z ) {

	// Domain warp so features don't look grid/noise-aligned.
	const wx = x + 450 * fbm( x * 0.00025 + 3.1, z * 0.00025 + 7.7, 3 );
	const wz = z + 450 * fbm( x * 0.00025 - 5.3, z * 0.00025 + 1.9, 3 );

	// Elliptic island with a noisy coastline.
	const r = Math.hypot( wx, wz * 1.12 ) / 7000;
	const coast = 1 - smoothstep( 0.5, 1.05, r + 0.2 * fbm( x * 0.0003, z * 0.0003, 5 ) );

	const hills = fbm( wx * 0.0007, wz * 0.0007, 6 ) * 0.5 + 0.5;
	const ridge = ridged( wx * 0.00042, wz * 0.00042, 8 );
	const mMask = smoothstep( 0.45, 0.9, 0.6 * ( 1 - r ) + 0.5 * ( fbm( x * 0.0002 + 11, z * 0.0002 - 4, 3 ) * 0.5 + 0.5 ) );

	const land = 6 + hills * 85 + Math.pow( ridge, 1.7 ) * 1250 * mMask;
	return land * coast - 120 * ( 1 - coast );

}

const lakeLevel = Math.max( 20, rawHeight( LAKE.x, LAKE.z ) - 6 );

function height( x, z ) {

	const hRaw = rawHeight( x, z );
	const dx = x - LAKE.x, dz = z - LAKE.z;
	let d = Math.hypot( dx, dz ) / LAKE.radius;
	if ( d > 2.6 ) return hRaw;
	d += 0.12 * fbm( x * 0.004, z * 0.004, 3 );

	const bowl = d < 1
		? lakeLevel + 3 - 30 * ( 1 - smoothstep( 0, 1, d ) )
		: lakeLevel + 3 + 25 * ( d - 1 );
	const outer = hRaw + ( Math.max( hRaw, lakeLevel + 3 ) - hRaw ) * ( 1 - smoothstep( 1.9, 2.6, d ) );
	const w = smoothstep( 1.0, 1.9, d );
	return bowl + ( outer - bowl ) * w;

}

self.onmessage = ( e ) => {

	const { rowStart, rowEnd } = e.data;
	const N = HEIGHTMAP_RES;
	const rows = rowEnd - rowStart;
	const heights = new Float32Array( rows * N );
	const forest = new Float32Array( rows * N );

	for ( let j = rowStart; j < rowEnd; j ++ ) {

		const z = ( ( j + 0.5 ) / N - 0.5 ) * WORLD_SIZE;
		for ( let i = 0; i < N; i ++ ) {

			const x = ( ( i + 0.5 ) / N - 0.5 ) * WORLD_SIZE;
			const o = ( j - rowStart ) * N + i;
			heights[ o ] = height( x, z );
			forest[ o ] = fbm( x * 0.0011 + 50, z * 0.0011 - 20, 4 ) + 0.25 * fbm( x * 0.006, z * 0.006, 2 );

		}

	}

	self.postMessage( { rowStart, rowEnd, heights, forest, lakeLevel }, [ heights.buffer, forest.buffer ] );

};
