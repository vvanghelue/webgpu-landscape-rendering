// Seeded 2D simplex noise + fractal helpers (CPU side, used by the generator worker).

const F2 = 0.5 * ( Math.sqrt( 3 ) - 1 );
const G2 = ( 3 - Math.sqrt( 3 ) ) / 6;
const GRAD = new Float32Array( [ 1, 1, - 1, 1, 1, - 1, - 1, - 1, 1, 0, - 1, 0, 0, 1, 0, - 1 ] );

export function createNoise( seed = 1337 ) {

	const perm = new Uint8Array( 512 );
	const p = new Uint8Array( 256 );
	for ( let i = 0; i < 256; i ++ ) p[ i ] = i;
	let s = seed >>> 0;
	for ( let i = 255; i > 0; i -- ) {

		s = ( Math.imul( s, 1664525 ) + 1013904223 ) >>> 0;
		const j = s % ( i + 1 );
		const t = p[ i ]; p[ i ] = p[ j ]; p[ j ] = t;

	}

	for ( let i = 0; i < 512; i ++ ) perm[ i ] = p[ i & 255 ];

	function simplex( x, y ) {

		const s = ( x + y ) * F2;
		const i = Math.floor( x + s ), j = Math.floor( y + s );
		const t = ( i + j ) * G2;
		const x0 = x - ( i - t ), y0 = y - ( j - t );
		const i1 = x0 > y0 ? 1 : 0, j1 = 1 - i1;
		const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
		const x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
		const ii = i & 255, jj = j & 255;
		let n = 0;

		let t0 = 0.5 - x0 * x0 - y0 * y0;
		if ( t0 > 0 ) {

			const g = ( perm[ ii + perm[ jj ] ] & 7 ) * 2;
			t0 *= t0; n += t0 * t0 * ( GRAD[ g ] * x0 + GRAD[ g + 1 ] * y0 );

		}

		let t1 = 0.5 - x1 * x1 - y1 * y1;
		if ( t1 > 0 ) {

			const g = ( perm[ ii + i1 + perm[ jj + j1 ] ] & 7 ) * 2;
			t1 *= t1; n += t1 * t1 * ( GRAD[ g ] * x1 + GRAD[ g + 1 ] * y1 );

		}

		let t2 = 0.5 - x2 * x2 - y2 * y2;
		if ( t2 > 0 ) {

			const g = ( perm[ ii + 1 + perm[ jj + 1 ] ] & 7 ) * 2;
			t2 *= t2; n += t2 * t2 * ( GRAD[ g ] * x2 + GRAD[ g + 1 ] * y2 );

		}

		return 70 * n; // ~[-1, 1]

	}

	function fbm( x, y, octaves, lacunarity = 2.0, gain = 0.5 ) {

		let sum = 0, amp = 1, norm = 0;
		for ( let o = 0; o < octaves; o ++ ) {

			sum += simplex( x, y ) * amp;
			norm += amp;
			amp *= gain;
			x = x * lacunarity + 17.13;
			y = y * lacunarity - 9.71;

		}

		return sum / norm;

	}

	// Ridged multifractal with weight feedback -> sharp crests, eroded-looking valleys. Returns ~[0, 1].
	function ridged( x, y, octaves ) {

		let sum = 0, amp = 0.5, weight = 1, norm = 0;
		for ( let o = 0; o < octaves; o ++ ) {

			let n = 1 - Math.abs( simplex( x, y ) );
			n *= n;
			n *= weight;
			weight = Math.min( 1, Math.max( 0, n * 2 ) );
			sum += n * amp;
			norm += amp;
			amp *= 0.5;
			x = x * 2.03 + 31.7;
			y = y * 2.03 - 11.3;

		}

		return sum / norm;

	}

	return { simplex, fbm, ridged };

}

export const smoothstep = ( a, b, x ) => {

	const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) );
	return t * t * ( 3 - 2 * t );

};
