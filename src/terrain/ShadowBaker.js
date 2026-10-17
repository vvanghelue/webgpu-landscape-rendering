import * as THREE from 'three/webgpu';
import { Fn, instanceIndex, vec2, vec4, float, uvec2, normalize, length, max, min, smoothstep, clamp, textureStore, Loop } from 'three/tsl';
import { sunDir, heightAt, terrainNormal } from '../shading/common.js';
import { WORLD_SIZE, HEIGHTMAP_RES } from '../world/config.js';

// GPU compute pass that bakes, for the whole island, a soft sun-visibility term (R) by ray-marching the heightmap
// towards the sun over several kilometres, plus a cheap large-scale ambient occlusion term (G).
// Re-run only when the sun moves. Everything (terrain, trees, water) samples it -> mountain shadows at any distance.
export class ShadowBaker {

	constructor( res = 2048 ) {

		this.res = res;
		this.texture = new THREE.StorageTexture( res, res );
		this.texture.magFilter = THREE.LinearFilter;
		this.texture.minFilter = THREE.LinearMipmapLinearFilter; // mips matter: distant pixels would thrash the cache
		this.texture.generateMipmaps = true;
		this.texture.wrapS = this.texture.wrapT = THREE.ClampToEdgeWrapping;
		this.dirty = true;

		// Terrain normals baked once at full heightmap resolution: 1 RGBA8 fetch per pixel instead of 4 float32 taps.
		this.normalTexture = new THREE.StorageTexture( HEIGHTMAP_RES, HEIGHTMAP_RES );
		this.normalTexture.magFilter = THREE.LinearFilter;
		this.normalTexture.minFilter = THREE.LinearMipmapLinearFilter;
		this.normalTexture.generateMipmaps = true;
		this.normalTexture.wrapS = this.normalTexture.wrapT = THREE.ClampToEdgeWrapping;
		this.normalsBaked = false;

	}

	build() {

		const N = this.res;
		const bake = Fn( ( { tex } ) => {

			const ix = instanceIndex.mod( N );
			const iy = instanceIndex.div( N );
			const uv = vec2( float( ix ).add( 0.5 ), float( iy ).add( 0.5 ) ).div( N );
			const p = uv.sub( 0.5 ).mul( WORLD_SIZE );
			const h0 = heightAt( p ).toVar();

			// --- sun visibility
			const horiz = max( length( sunDir.xz ), 1e-4 );
			const dir = normalize( sunDir.xz );
			const tanE = sunDir.y.div( horiz );
			const vis = float( 1.0 ).toVar();
			const t = float( 3.0 ).toVar();

			Loop( 100, () => {

				const hq = heightAt( p.add( dir.mul( t ) ) );
				const rayH = h0.add( t.mul( tanE ) ).add( 1.5 );
				const w = t.mul( 0.012 ).add( 1.5 ); // penumbra widens with distance
				vis.assign( min( vis, smoothstep( w.negate(), w, rayH.sub( hq ) ) ) );
				t.mulAssign( 1.075 );

			} );

			vis.mulAssign( smoothstep( - 0.02, 0.06, sunDir.y ) );

			// --- large-scale AO: how much the surrounding terrain rises above this texel.
			const occl = float( 0.0 ).toVar();
			for ( const r of [ 12, 40, 120, 320 ] ) {

				for ( let a = 0; a < 8; a ++ ) {

					const ang = ( a + ( r % 3 ) * 0.33 ) / 8 * Math.PI * 2;
					const hq = heightAt( p.add( vec2( Math.cos( ang ) * r, Math.sin( ang ) * r ) ) );
					occl.addAssign( clamp( hq.sub( h0 ).div( r ), 0.0, 1.0 ) );

				}

			}

			const ao = clamp( float( 1.0 ).sub( occl.div( 32.0 ).mul( 1.6 ) ), 0.25, 1.0 );
			textureStore( tex, uvec2( ix, iy ), vec4( vis, ao, 0.0, 1.0 ) ).toWriteOnly();

		} );

		this.computeNode = bake( { tex: this.texture } ).compute( N * N );

		const NR = HEIGHTMAP_RES;
		this.normalNode = Fn( ( { tex } ) => {

			const ix = instanceIndex.mod( NR );
			const iy = instanceIndex.div( NR );
			const p = vec2( float( ix ).add( 0.5 ), float( iy ).add( 0.5 ) ).div( NR ).sub( 0.5 ).mul( WORLD_SIZE );
			textureStore( tex, uvec2( ix, iy ), vec4( terrainNormal( p ).mul( 0.5 ).add( 0.5 ), 1.0 ) ).toWriteOnly();

		} )( { tex: this.normalTexture } ).compute( NR * NR );

	}

	update( renderer ) {

		if ( ! this.normalsBaked ) {

			this.normalsBaked = true;
			renderer.compute( this.normalNode );

		}

		if ( ! this.dirty ) return;
		this.dirty = false;
		renderer.compute( this.computeNode );

	}

}
