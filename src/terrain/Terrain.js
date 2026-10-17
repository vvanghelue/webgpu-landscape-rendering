import * as THREE from 'three/webgpu';
import {
	Fn, attribute, positionLocal, positionWorld, cameraPosition, vec2, vec3, float, fract, clamp, mix, exp2, select,
	distance, smoothstep, max, uniform, texture, If, normalize, varyingProperty,
} from 'three/tsl';
import {
	heightAt, ss, valueNoise, fogParams, shade, applyFog, worldUV, G, proceduralTreeShadow, treeShadowDistance,
} from '../shading/common.js';
import { WORLD_SIZE } from '../world/config.js';

// CDLOD terrain (Strugar 2009): a quadtree is selected on the CPU every frame; each selected quadrant is drawn as an
// instance of one shared 16x16 grid patch. Vertices morph towards the next-coarser grid near the LOD range boundary,
// giving crack-free, pop-free continuous LOD over the whole island with a few hundred instances / one draw call.

const vFog = varyingProperty( 'vec4', 'vTerrainFog' );

const PATCH = 16; // cells per drawn quadrant
const LEAF_NODE = 128; // m, finest node size -> 4 m cells (matches heightmap)
const MAX_INSTANCES = 4096;

function createPatchGeometry() {

	const n = PATCH + 1;
	const pos = new Float32Array( n * n * 3 );
	for ( let j = 0; j < n; j ++ ) for ( let i = 0; i < n; i ++ ) pos.set( [ i, 0, j ], ( j * n + i ) * 3 );
	const idx = [];
	for ( let j = 0; j < PATCH; j ++ ) for ( let i = 0; i < PATCH; i ++ ) {

		const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
		// alternate the diagonal for nicer morph shapes
		if ( ( i + j ) & 1 ) idx.push( a, c, b, b, c, d );
		else idx.push( a, c, d, a, d, b );

	}

	const geo = new THREE.InstancedBufferGeometry();
	geo.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
	geo.setIndex( idx );
	return geo;

}

export class Terrain {

	constructor( world ) {

		this.world = world;
		this.levels = Math.round( Math.log2( WORLD_SIZE / LEAF_NODE ) ); // root level
		this.lodRange0 = uniform( LEAF_NODE * 5.0 );
		this.morphStart = 0.6;

		this.instanceData = new Float32Array( MAX_INSTANCES * 4 );
		this.instanceAttr = new THREE.InstancedBufferAttribute( this.instanceData, 4 );
		this.instanceAttr.setUsage( THREE.DynamicDrawUsage );
		this.geometry = createPatchGeometry();
		this.geometry.setAttribute( 'inst', this.instanceAttr );
		this.geometry.instanceCount = 0;

		this.material = this.createMaterial();
		this.mesh = new THREE.Mesh( this.geometry, this.material );
		this.mesh.frustumCulled = false;

		this.frustum = new THREE.Frustum();
		this._box = new THREE.Box3();
		this._range = { min: 0, max: 0 };
		this.stats = { patches: 0 };

	}

	createMaterial() {

		const mat = new THREE.MeshBasicNodeMaterial();
		const inst = attribute( 'inst', 'vec4' );
		const lodRange0 = this.lodRange0;
		const morphStart = this.morphStart;

		mat.positionNode = Fn( () => {

			const g = positionLocal.xz;
			const cell = inst.z.div( PATCH );
			const wxz = inst.xy.add( g.mul( cell ) );
			const h0 = heightAt( wxz );
			const d = distance( cameraPosition, vec3( wxz.x, h0, wxz.y ) );
			const range = lodRange0.mul( exp2( inst.w ) );
			const prev = select( inst.w.greaterThan( 0.5 ), range.mul( 0.5 ), float( 0 ) );
			const start = mix( prev, range, morphStart );
			const k = clamp( d.sub( start ).div( range.sub( start ) ), 0.0, 1.0 );
			const m = wxz.sub( fract( g.mul( 0.5 ) ).mul( 2.0 ).mul( cell ).mul( k ) );
			const wp = vec3( m.x, heightAt( m ), m.y ).toVar();
			vFog.assign( fogParams( wp ) ); // fog varies smoothly -> per vertex is plenty
			return wp;

		} )();

		mat.colorNode = Fn( () => {

			const p = positionWorld;
			const uv = worldUV( p.xz );
			const N = normalize( texture( G.normalTexture, uv ).xyz.mul( 2.0 ).sub( 1.0 ) ).toVar();
			const h = p.y;
			const slope = float( 1.0 ).sub( N.y );
			const dist = distance( cameraPosition, p );

			const sh = texture( G.shadowTexture, uv );
			const dens = texture( G.densityTexture, uv ).r;

			// --- albedo layers
			const pw = vec2( p.x.add( p.y.mul( 0.7 ) ), p.z.sub( p.y.mul( 0.7 ) ) ); // skew by height: no stretching on cliffs
			const n1 = valueNoise( pw.mul( 0.013 ) ).mul( 0.7 ).add( valueNoise( pw.mul( 0.041 ) ).mul( 0.3 ) );
			const n2 = valueNoise( pw.mul( 0.11 ) ).mul( 0.6 ).add( valueNoise( pw.mul( 0.37 ) ).mul( 0.4 ) );
			const grass = mix( vec3( 0.13, 0.2, 0.05 ), vec3( 0.26, 0.3, 0.1 ), n1.mul( 0.7 ).add( n2.mul( 0.3 ) ) );
			const forestFloor = vec3( 0.07, 0.09, 0.04 );
			const sand = mix( vec3( 0.55, 0.5, 0.38 ), vec3( 0.7, 0.64, 0.5 ), n2 );
			const wetSand = vec3( 0.32, 0.29, 0.22 );
			const rock = mix( vec3( 0.25, 0.23, 0.21 ), vec3( 0.42, 0.4, 0.37 ), n1.mul( 0.6 ).add( n2.mul( 0.4 ) ) );
			const snow = vec3( 0.86, 0.89, 0.93 );

			const col = grass.toVar();
			col.assign( mix( col, forestFloor, smoothstep( 0.05, 0.7, dens ) ) );
			col.assign( mix( col, sand, ss( 4.5, 2.0, h.add( n1.mul( 2.0 ) ) ) ) );
			col.assign( mix( col, wetSand, ss( 0.6, - 1.5, h ) ) );
			const rockAmt = smoothstep( 0.22, 0.42, slope.add( n2.mul( 0.08 ) ) ).max( smoothstep( 650, 900, h.add( n1.mul( 120 ) ) ) );
			col.assign( mix( col, rock, rockAmt ) );
			const snowAmt = smoothstep( 820, 960, h.add( n1.mul( 160 ) ) ).mul( ss( 0.55, 0.3, slope ) );
			col.assign( mix( col, snow, snowAmt ) );

			// --- tree canopy shadows: exact procedural ones nearby, statistical darkening far away.
			const treeVis = float( 1.0 ).toVar();
			If( dist.lessThan( treeShadowDistance ), () => {

				treeVis.assign( proceduralTreeShadow( p ) );

			} );
			const farVis = float( 1.0 ).sub( dens.mul( 0.55 ) );
			treeVis.assign( mix( treeVis, farVis, smoothstep( treeShadowDistance.mul( 0.6 ), treeShadowDistance, dist ) ) );

			const ao = sh.g.mul( mix( 1.0, 0.55, dens ) );
			const lit = shade( col, N, sh.r.mul( treeVis ), ao );
			return mix( lit, vFog.xyz, vFog.w );

		} )();

		return mat;

	}

	update( camera ) {

		camera.updateMatrixWorld();
		this._projView = this._projView || new THREE.Matrix4();
		this._projView.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
		this.frustum.setFromProjectionMatrix( this._projView, camera.coordinateSystem, camera.reversedDepth );
		this.camPos = camera.position;
		this.count = 0;

		const half = WORLD_SIZE / 2;
		this.select( - half, - half, WORLD_SIZE, this.levels );

		this.geometry.instanceCount = this.count;
		this.instanceAttr.needsUpdate = true;
		this.stats.patches = this.count;

	}

	range( level ) {

		return this.lodRange0.value * Math.pow( 2, level );

	}

	nodeBox( x, z, size ) {

		this.world.rangeOf( x, z, size, this._range );
		return this._box.set(
			this._box.min.set( x, this._range.min, z ),
			this._box.max.set( x + size, this._range.max, z + size ),
		);

	}

	select( x, z, size, level ) {

		const box = this.nodeBox( x, z, size );
		if ( ! this.frustum.intersectsBox( box ) ) return;
		const dist = box.distanceToPoint( this.camPos );

		if ( level === 0 || dist >= this.range( level - 1 ) ) {

			this.emitNode( x, z, size, level );
			return;

		}

		const h = size / 2;
		for ( let c = 0; c < 4; c ++ ) {

			const cx = x + ( c & 1 ) * h, cz = z + ( c >> 1 ) * h;
			const cbox = this.nodeBox( cx, cz, h );
			if ( cbox.distanceToPoint( this.camPos ) < this.range( level - 1 ) ) {

				this.select( cx, cz, h, level - 1 );

			} else if ( this.frustum.intersectsBox( cbox ) ) {

				this.emit( cx, cz, h, level ); // child area drawn at this node's resolution

			}

		}

	}

	emitNode( x, z, size, level ) {

		const h = size / 2;
		for ( let c = 0; c < 4; c ++ ) {

			const cx = x + ( c & 1 ) * h, cz = z + ( c >> 1 ) * h;
			if ( this.frustum.intersectsBox( this.nodeBox( cx, cz, h ) ) ) this.emit( cx, cz, h, level );

		}

	}

	emit( x, z, size, level ) {

		if ( this.count >= MAX_INSTANCES ) return;
		const o = this.count * 4;
		this.instanceData[ o ] = x;
		this.instanceData[ o + 1 ] = z;
		this.instanceData[ o + 2 ] = size;
		this.instanceData[ o + 3 ] = level;
		this.count ++;

	}

}
