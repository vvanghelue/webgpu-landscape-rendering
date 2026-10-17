import * as THREE from 'three/webgpu';
import {
	Fn, If, attribute, positionLocal, positionWorld, normalLocal, cameraPosition, instanceIndex, vertexIndex, varyingProperty, uniform,
	vec2, vec3, vec4, float, uint, abs, mod, floor, fract, max, mix, step, select, sqrt, sin, cos, atan, exp2, length,
	normalize, smoothstep, clamp, dot, instancedArray, storage, atomicAdd, atomicStore,
} from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import {
	treeAt, treeIsBroadleaf, heightAt, shadowAt, shade, applyFog, fogParams, hash22, proceduralTreeShadow, ss,
	nearTreeRadius, treeCrossfade, crossfadeIn, crossfadeOut, treeDrawDistance, treeTierRange0, impostorFarDistance,
} from '../shading/common.js';
import { WORLD_SIZE, TREE_CELL } from '../world/config.js';

// Millions of trees with zero per-tree CPU data. Tree positions are a pure function of a 4 m cell grid
// (hash + density texture), so every system (terrain shadows, impostors, meshes) agrees on where trees are.
//
// GPU-driven pipeline, every frame:
//   1. CPU quadtree picks tiles of 32x32 cells (frustum + distance tiers) -> small storage buffer.
//   2. Compute "cull" pass: one thread per cell evaluates the tree, applies distance fades + sphere/frustum test and
//      appends survivors to compact instance buffers with an atomic counter that is also the indirect draw args.
//   3. Indirect draws: impostor quads (far) and two low-poly mesh species (near).
// Each tier further away visits every 2nd cell per axis; trees that vanish in the next tier are shrunk smoothly before
// the switch, so there is no popping.
//
// Mesh <-> impostor is a real alpha crossfade across a wide band around nearTreeRadius, and one of the two is always
// opaque so the terrain never shows through a half-faded tree:
//   crossfadeIn .. nearTreeRadius   opaque mesh, card fading out on top of it
//   nearTreeRadius .. crossfadeOut  opaque card, mesh fading in on top of it
// No dither or noise, so the transition is just as smooth while moving as when standing still.

const TILE = 32; // cells per tile side
const TIERS = 5;
const MAX_TILES = 4096;
const MAX_IMPOSTORS = 2_000_000;
const MAX_NEAR = 32768;
const MAX_FADE = 16384;

const vUv = varyingProperty( 'vec2', 'vTreeUv' );
const vInfo = varyingProperty( 'vec4', 'vTreeInfo' ); // x: broadleaf, y: rnd, z: sun visibility, w: ao
const vRight = varyingProperty( 'vec3', 'vTreeRight' );
const vToCam = varyingProperty( 'vec3', 'vTreeToCam' );
const vNormal = varyingProperty( 'vec3', 'vTreeNormal' );
const vLocalY = varyingProperty( 'float', 'vTreeLocalY' );
const vFog = varyingProperty( 'vec4', 'vTreeFog' );
const vFade = varyingProperty( 'float', 'vTreeFade' );

// Instance record: vec4( x, groundY, z, floor(height * 100) + rnd )
const pack = ( H, rnd ) => floor( H.mul( 100.0 ) ).add( rnd.mul( 0.999 ) );
const unpackH = ( w ) => floor( w ).mul( 0.01 );
const unpackRnd = ( w ) => fract( w );

// ---------------------------------------------------------------------------------------------------------------
function colored( geo, color ) {

	geo = geo.index ? geo.toNonIndexed() : geo;
	const n = geo.attributes.position.count;
	const c = new Float32Array( n * 3 );
	for ( let i = 0; i < n; i ++ ) c.set( [ color.r, color.g, color.b ], i * 3 );
	geo.setAttribute( 'color', new THREE.BufferAttribute( c, 3 ) );
	geo.deleteAttribute( 'uv' );
	return geo;

}

function createConiferGeometry() {

	const bark = new THREE.Color( 0.22, 0.14, 0.08 );
	const parts = [ colored( new THREE.CylinderGeometry( 0.018, 0.03, 0.3, 5, 1, true ).translate( 0, 0.15, 0 ), bark ) ];
	const tiers = [ [ 0.16, 0.42, 0.25 ], [ 0.34, 0.38, 0.2 ], [ 0.52, 0.32, 0.15 ], [ 0.68, 0.32, 0.1 ] ];
	tiers.forEach( ( [ y, h, r ], i ) => {

		const g = new THREE.ConeGeometry( r, h, 8, 1, false ).translate( 0, y + h / 2, 0 ).rotateY( i * 0.7 );
		parts.push( colored( g, new THREE.Color( 0.05, 0.12 + i * 0.012, 0.045 ) ) );

	} );
	return mergeGeometries( parts );

}

function createBroadleafGeometry() {

	const bark = new THREE.Color( 0.25, 0.18, 0.11 );
	const leaf = new THREE.Color( 0.12, 0.2, 0.05 );
	const parts = [ colored( new THREE.CylinderGeometry( 0.025, 0.045, 0.5, 5, 1, true ).translate( 0, 0.25, 0 ), bark ) ];
	const blobs = [ [ 0, 0.62, 0, 0.3 ], [ 0.14, 0.52, 0.06, 0.22 ], [ - 0.12, 0.55, - 0.08, 0.22 ], [ 0.02, 0.8, - 0.03, 0.2 ] ];
	for ( const [ x, y, z, r ] of blobs ) parts.push( colored( new THREE.IcosahedronGeometry( r, 0 ).translate( x, y, z ), leaf ) );
	return mergeGeometries( parts );

}

// Opaque silhouette cards (no alpha test: Apple/mobile TBDR GPUs lose hidden-surface removal with discard).
// Vertex positions are in billboard uv space ([0,1]², y up); position.z flags broadleaf rim vertices for jitter.
function createConiferCardGeometry() {

	const pos = [ 0.46, 0, 0, 0.54, 0, 0, 0.46, 0.2, 0, 0.54, 0.2, 0 ];
	const idx = [ 0, 1, 2, 2, 1, 3 ];
	const tiers = [ [ 0.1, 0.5, 0.56 ], [ 0.3, 0.4, 0.73 ], [ 0.5, 0.3, 0.89 ], [ 0.68, 0.19, 1.0 ] ];
	for ( const [ y, hw, top ] of tiers ) {

		const b = pos.length / 3;
		pos.push( 0.5 - hw, y, 0, 0.5 + hw, y, 0, 0.5, top, 0 );
		idx.push( b, b + 1, b + 2 );

	}

	return cardGeometry( pos, idx );

}

// Far LOD: a single triangle (trees are a few pixels tall there; sub-pixel triangles are what costs).
function createConiferFarCardGeometry() {

	return cardGeometry( [ 0.0, 0.05, 0, 1.0, 0.05, 0, 0.5, 1.0, 0 ], [ 0, 1, 2 ] );

}

function createBroadleafCardGeometry( RIM = 12, trunk = true ) {

	const pos = [ 0.45, 0, 0, 0.55, 0, 0, 0.45, 0.4, 0, 0.55, 0.4, 0, 0.5, 0.62, 0 ];
	const idx = trunk ? [ 0, 1, 2, 2, 1, 3 ] : [];
	for ( let k = 0; k < RIM; k ++ ) {

		const a = ( k / RIM ) * Math.PI * 2;
		pos.push( 0.5 + 0.5 * Math.cos( a ), 0.62 + 0.38 * Math.sin( a ), k + 1 );
		idx.push( 4, 5 + k, 5 + ( ( k + 1 ) % RIM ) );

	}

	return cardGeometry( pos, idx );

}

function cardGeometry( pos, idx ) {

	const geo = new THREE.InstancedBufferGeometry();
	geo.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( pos ), 3 ) );
	geo.setIndex( idx );
	return geo;

}

// All indirect draw args live in ONE buffer (5 uints per draw, offset per draw) to stay under the
// 8-storage-buffers-per-stage limit in the cull passes. instanceCount (word 1) is the atomic append counter.
const MAX_DRAWS = 10;
const drawArgsAttr = new THREE.IndirectStorageBufferAttribute( new Uint32Array( MAX_DRAWS * 5 ), 1 );
const drawArgsNode = storage( drawArgsAttr, 'uint', MAX_DRAWS * 5 ).toAtomic();
let drawSlots = 0;

function indirectArgs( geometry, countPerInstance ) {

	const slot = drawSlots ++;
	drawArgsAttr.array.set( [ countPerInstance, 0, 0, 0, 0 ], slot * 5 );
	geometry.setIndirect( drawArgsAttr, slot * 20 );
	geometry.instanceCount = 1; // ignored by the indirect draw, but must be non-zero
	return { slot, counter: drawArgsNode.element( slot * 5 + 1 ) };

}

// ---------------------------------------------------------------------------------------------------------------
export class Trees {

	constructor( world ) {

		this.world = world;
		this.group = new THREE.Group();
		this.frustum = new THREE.Frustum();
		this._box = new THREE.Box3();
		this._range = { min: 0, max: 0 };
		this._pv = new THREE.Matrix4();
		this.stats = { tiles: 0, cells: 0, nearCells: 0, visibleImpostors: 0, visibleMeshes: 0, crossfading: 0 };
		this.enabledNear = true;
		this.enabledFar = true;

		this.planes = Array.from( { length: 6 }, () => uniform( new THREE.Vector4() ) );
		this.cullCameraPosition = uniform( new THREE.Vector3() );
		this.gridOrigin = uniform( new THREE.Vector2() );
		this.gridN = uniform( 0, 'uint' );

		// --- buffers
		this.tiles = instancedArray( MAX_TILES, 'vec4' );
		// lists: [ conifer near, broadleaf near, conifer far, broadleaf far ]
		this.impostorInstances = [ 0.3, 0.15, 0.7, 0.3 ].map( ( f ) => instancedArray( MAX_IMPOSTORS * f, 'vec4' ) );
		this.nearInstances = [ instancedArray( MAX_NEAR, 'vec4' ), instancedArray( MAX_NEAR, 'vec4' ) ];
		this.meshFadeInstances = [ instancedArray( MAX_FADE, 'vec4' ), instancedArray( MAX_FADE, 'vec4' ) ];
		this.cardFadeInstances = [ instancedArray( MAX_FADE, 'vec4' ), instancedArray( MAX_FADE, 'vec4' ) ];

		// --- impostors: one opaque card draw per species, drawn indirectly
		this.impostorArgs = [];
		const cards = [ createConiferCardGeometry(), createBroadleafCardGeometry(), createConiferFarCardGeometry(), createBroadleafCardGeometry( 6, false ) ];
		this.impostors = cards.map( ( geo, list ) => {

			const species = list & 1;

			this.impostorArgs.push( indirectArgs( geo, geo.index.count ) );
			const mesh = new THREE.Mesh( geo, this.createImpostorMaterial( species, this.impostorInstances[ list ] ) );
			mesh.frustumCulled = false;
			this.group.add( mesh );
			return mesh;

		} );

		// --- near meshes, one indirect draw per species
		this.nearArgs = [];
		this.nearMeshes = [ createConiferGeometry(), createBroadleafGeometry() ].map( ( src, species ) => {

			const geo = new THREE.InstancedBufferGeometry().copy( src );
			this.nearArgs.push( indirectArgs( geo, src.attributes.position.count ) );
			const mesh = new THREE.Mesh( geo, this.createMeshMaterial( species, this.nearInstances[ species ] ) );
			mesh.frustumCulled = false;
			this.group.add( mesh );
			return mesh;

		} );

		// --- crossfade band, drawn after all opaque geometry (transparent queue, in this renderOrder)
		// Fading meshes: a depth-only pass first, so the blended pass only shades each tree's front-most surface
		// (otherwise its overlapping cones/blobs would show through each other).
		this.meshFadeArgs = [];
		this.fadeMeshes = [ createConiferGeometry(), createBroadleafGeometry() ].flatMap( ( src, species ) => {

			const geo = new THREE.InstancedBufferGeometry().copy( src );
			this.meshFadeArgs.push( indirectArgs( geo, src.attributes.position.count ) );
			const color = this.createMeshMaterial( species, this.meshFadeInstances[ species ], true );
			// Same node graph -> identical vertex shader, so the LessEqual test in the colour pass hits the prepass depth exactly.
			const depth = color.clone();
			depth.colorWrite = false;
			depth.depthWrite = true;
			return [ [ depth, 4 ], [ color, 5 ] ].map( ( [ mat, order ] ) => {

				const mesh = new THREE.Mesh( geo, mat );
				mesh.frustumCulled = false;
				mesh.renderOrder = order;
				this.group.add( mesh );
				return mesh;

			} );

		} );

		this.cardFadeArgs = [];
		this.fadeCards = [ createConiferCardGeometry(), createBroadleafCardGeometry() ].map( ( geo, species ) => {

			this.cardFadeArgs.push( indirectArgs( geo, geo.index.count ) );
			const mesh = new THREE.Mesh( geo, this.createImpostorMaterial( species, this.cardFadeInstances[ species ], true ) );
			mesh.frustumCulled = false;
			mesh.renderOrder = 6;
			this.group.add( mesh );
			return mesh;

		} );

		this.createComputePasses();

	}

	allArgs() {

		return [ ...this.impostorArgs, ...this.nearArgs, ...this.meshFadeArgs, ...this.cardFadeArgs ];

	}

	// ----------------------------------------------------------------------------------------- compute culling
	createComputePasses() {

		const planes = this.planes;
		const camPos = this.cullCameraPosition; // compute passes have no camera bound
		const inFrustum = ( c, r ) => {

			let inside = float( 1.0 );
			for ( const p of planes ) inside = inside.mul( step( r.negate(), dot( p.xyz, c ).add( p.w ) ) );
			return inside.greaterThan( 0.5 );

		};

		const counters = this.allArgs().map( ( a ) => a.counter );
		this.resetPass = Fn( () => {

			for ( const c of counters ) atomicStore( c, uint( 0 ) );

		} )().compute( 1 );

		// Append a record to the per-species instance list; the counter doubles as the indirect instanceCount.
		const append = ( args, lists, isBroad, record, isFar = null ) => {

			// Assign the record before branching: TSL caches a node reused across sibling If branches into a var that is
			// only assigned in the first branch, so the other branches would write zeros.
			const rec = record.toVar();
			const emit = ( species ) => {

				const slot = atomicAdd( args[ species ].counter, uint( 1 ) ).toVar();
				If( slot.lessThan( uint( lists[ species ].value.count ) ), () => {

					lists[ species ].element( slot ).assign( rec );

				} );

			};

			if ( isFar === null ) {

				If( isBroad, () => emit( 1 ) ).Else( () => emit( 0 ) );

			} else {

				If( isFar, () => {

					If( isBroad, () => emit( 3 ) ).Else( () => emit( 2 ) );

				} ).Else( () => {

					If( isBroad, () => emit( 1 ) ).Else( () => emit( 0 ) );

				} );

			}

		};

		const impostorArgs = this.impostorArgs, nearArgs = this.nearArgs;

		const tiles = this.tiles;
		const impostorInstances = this.impostorInstances;

		this.impostorCullPass = Fn( () => {

			const tile = tiles.element( instanceIndex.div( TILE * TILE ) );
			const local = instanceIndex.mod( TILE * TILE );
			const stride = tile.z;
			const cell = tile.xy.add( vec2( float( local.mod( TILE ) ), float( local.div( TILE ) ) ).mul( stride ) );
			const t = treeAt( cell ).toVar();

			If( t.z.greaterThan( 0.01 ), () => {

				const base = vec3( t.x, heightAt( t.xy ), t.y ).toVar();
				const dist = length( camPos.sub( base ) ).toVar();

				// Trees not kept by the next (coarser) tier shrink away before the tier switch.
				const s2 = stride.mul( 2.0 );
				const lattice = step( abs( mod( cell.x, s2 ) ).add( abs( mod( cell.y, s2 ) ) ), 0.5 );
				const range = treeTierRange0.mul( exp2( tile.w ) );
				const fade = max( lattice, ss( range, range.mul( 0.7 ), dist ) );
				const farFade = ss( treeDrawDistance, treeDrawDistance.mul( 0.85 ), dist );
				// Slightly enlarge distant trees so thinned-out tiers keep similar canopy coverage.
				const boost = mix( 1.0, 1.35, smoothstep( treeTierRange0, treeTierRange0.mul( 4.0 ), dist ) );
				const H = t.z.mul( fade ).mul( farFade ).mul( boost ).toVar();

				If( H.greaterThan( 0.5 ).and( dist.greaterThanEqual( nearTreeRadius ) ).and( inFrustum( base.add( vec3( 0, H.mul( 0.5 ), 0 ) ), H.mul( 0.6 ) ) ), () => {

					append( impostorArgs, impostorInstances, treeIsBroadleaf( t.w ).greaterThan( 0.5 ), vec4( base, pack( H, t.w ) ), dist.greaterThan( impostorFarDistance ) );

				} );

			} );

		} )().compute( 1 );

		const gridOrigin = this.gridOrigin, gridN = this.gridN;
		const nearInstances = this.nearInstances;
		const meshFadeArgs = this.meshFadeArgs, meshFadeInstances = this.meshFadeInstances;
		const cardFadeArgs = this.cardFadeArgs, cardFadeInstances = this.cardFadeInstances;

		// Covers the whole crossfade band. Beyond nearTreeRadius the opaque card comes from the impostor pass.
		this.nearCullPass = Fn( () => {

			const cell = gridOrigin.add( vec2( float( instanceIndex.mod( gridN ) ), float( instanceIndex.div( gridN ) ) ) );
			const t = treeAt( cell ).toVar();

			If( t.z.greaterThan( 0.01 ), () => {

				const base = vec3( t.x, heightAt( t.xy ), t.y ).toVar();
				const dist = length( camPos.sub( base ) ).toVar();

				If( dist.lessThan( crossfadeOut ).and( inFrustum( base.add( vec3( 0, t.z.mul( 0.5 ), 0 ) ), t.z.mul( 0.6 ) ) ), () => {

					// Shared by sibling branches below, so assign before branching (see append).
					const isBroad = treeIsBroadleaf( t.w ).greaterThan( 0.5 ).toVar();
					const rec = vec4( base, pack( t.z, t.w ) ).toVar();

					If( dist.lessThan( nearTreeRadius ), () => {

						append( nearArgs, nearInstances, isBroad, rec );
						If( dist.greaterThan( crossfadeIn ), () => append( cardFadeArgs, cardFadeInstances, isBroad, rec ) );

					} ).Else( () => {

						append( meshFadeArgs, meshFadeInstances, isBroad, rec );

					} );

				} );

			} );

		} )().compute( 1 );

	}

	// ----------------------------------------------------------------------------------------- impostor material
	// fade: card fading out over the opaque mesh in the inner half of the crossfade band.
	createImpostorMaterial( species, instances, fade = false ) {

		const mat = new THREE.MeshBasicNodeMaterial();
		const inst = instances.toAttribute();
		const broad = float( species );

		mat.positionNode = Fn( () => {

			const base = inst.xyz;
			const H = unpackH( inst.w );
			const rnd = unpackRnd( inst.w );
			const toCam = cameraPosition.sub( base ).toVar();
			const dist = length( toCam );

			// broadleaf rim vertices get a per-tree radial jitter -> irregular crowns
			const rim = positionLocal.z;
			const jitter = select( rim.greaterThan( 0.5 ), hash22( vec2( rim, rnd.mul( 97.0 ) ) ).x.mul( 0.22 ).add( 0.8 ), float( 1.0 ) );
			const c = vec2( 0.5, 0.62 );
			const uv = select( rim.greaterThan( 0.5 ), c.add( positionLocal.xy.sub( c ).mul( jitter ) ), positionLocal.xy );

			const viewDir = toCam.div( max( dist, 1e-3 ) );
			const right = normalize( vec3( toCam.z.negate(), 0.0, toCam.x ).add( vec3( 1e-5, 0, 0 ) ) );
			// Lean the card's up-vector towards the camera so trees don't turn into slivers from above.
			const upOrtho = normalize( vec3( 0, 1, 0 ).sub( viewDir.mul( viewDir.y ) ).add( vec3( 0, 1e-4, 0 ) ) );
			const up = normalize( mix( vec3( 0, 1, 0 ), upOrtho, 0.6 ) );
			const width = H.mul( species ? 0.78 : 0.52 );

			const sh = shadowAt( base.xz );
			vUv.assign( uv );
			vInfo.assign( vec4( broad, rnd, sh.r, sh.g ) );
			vRight.assign( right );
			vToCam.assign( viewDir );
			vFog.assign( fogParams( base.add( vec3( 0, H.mul( 0.5 ), 0 ) ) ) );

			const p = base.add( right.mul( uv.x.sub( 0.5 ).mul( width ) ) ).add( up.mul( uv.y.mul( H ).sub( H.mul( 0.04 ) ) ) );
			if ( ! fade ) return p;

			vFade.assign( ss( crossfadeIn, nearTreeRadius, dist ) );
			// The card's triangles overlap and are coplanar. Pushing each later vertex 1 cm further away makes later
			// triangles strictly behind earlier ones, so with depth write + a strict test every pixel is blended once.
			return p.sub( viewDir.mul( float( vertexIndex ).mul( 0.01 ) ) );

		} )();

		// Fragment: fake volumetric normal from the card uv + leafy speckle. Fully opaque.
		mat.colorNode = Fn( () => {

			const uv = vUv;
			const rnd = vInfo.y;
			const y = uv.y;
			const speckle = hash22( floor( uv.mul( vec2( 14.0, 30.0 ) ) ).add( rnd.mul( 91.0 ) ) ).x;
			const isTrunk = step( y, species ? 0.36 : 0.13 );

			let nx, ny;
			if ( species ) {

				const d = uv.sub( vec2( 0.5, 0.62 ) ).div( vec2( 0.5, 0.38 ) );
				nx = clamp( d.x, - 1.0, 1.0 );
				ny = d.y.mul( 0.7 ).add( 0.25 );

			} else {

				nx = clamp( uv.x.sub( 0.5 ).mul( 2.2 ).div( max( float( 1.05 ).sub( y ), 0.1 ) ), - 1.0, 1.0 );
				ny = y.mul( 0.4 ).add( 0.2 );

			}

			const nz = sqrt( max( float( 1.0 ).sub( nx.mul( nx ) ), 0.05 ) );
			const N = normalize( vRight.mul( nx ).add( vec3( 0, ny, 0 ) ).add( vToCam.mul( nz ) ) );

			const leafCol = ( species ? vec3( 0.12, 0.19, 0.05 ) : vec3( 0.045, 0.1, 0.04 ) )
				.mul( mix( 0.75, 1.2, rnd ) ).mul( speckle.mul( 0.35 ).add( 0.8 ) );
			const albedo = mix( leafCol, vec3( 0.2, 0.13, 0.08 ), isTrunk );
			const ao = mix( 0.4, 1.0, y ).mul( vInfo.w );

			return mix( shade( albedo, N, vInfo.z, ao ), vFog.xyz, vFog.w );

		} )();
		mat.side = THREE.DoubleSide;

		if ( fade ) {

			mat.opacityNode = vFade;
			mat.transparent = true;
			mat.depthFunc = THREE.LessDepth;

		}

		return mat;

	}

	// ----------------------------------------------------------------------------------------- mesh material
	// fade: mesh fading in over the opaque card in the outer half of the crossfade band (needs a depth prepass).
	createMeshMaterial( species, instances, fade = false ) {

		const mat = new THREE.MeshBasicNodeMaterial();
		const inst = instances.toAttribute();

		mat.positionNode = Fn( () => {

			const base = inst.xyz;
			const H = unpackH( inst.w );
			const rnd = unpackRnd( inst.w );
			const W = H.mul( mix( 0.85, 1.15, fract( rnd.mul( 31.7 ) ) ) );

			const ang = rnd.mul( 43.98 );
			const c = cos( ang ), s = sin( ang );
			const p = positionLocal;
			const rx = p.x.mul( c ).sub( p.z.mul( s ) );
			const rz = p.x.mul( s ).add( p.z.mul( c ) );
			const n = normalLocal;
			vNormal.assign( vec3( n.x.mul( c ).sub( n.z.mul( s ) ), n.y, n.x.mul( s ).add( n.z.mul( c ) ) ) );
			vLocalY.assign( p.y );
			const sh = shadowAt( base.xz );
			vInfo.assign( vec4( float( species ), rnd, sh.r, sh.g ) );
			if ( fade ) vFade.assign( ss( crossfadeOut, nearTreeRadius, length( cameraPosition.sub( base ) ) ) );

			return base.add( vec3( rx.mul( W ), p.y.mul( H ).sub( 0.3 ), rz.mul( W ) ) );

		} )();

		mat.colorNode = Fn( () => {

			const albedo = attribute( 'color', 'vec3' ).mul( mix( 0.8, 1.2, vInfo.y ) );
			const N = normalize( vNormal );
			const selfShadow = proceduralTreeShadow( positionWorld );
			const ao = mix( 0.35, 1.0, clamp( vLocalY.mul( 1.3 ), 0.0, 1.0 ) ).mul( vInfo.w );
			// a bit of wrap lighting so foliage isn't pitch black on the back side
			const wrapN = normalize( N.add( vec3( 0, 0.35, 0 ) ) );
			return applyFog( shade( albedo, wrapN, vInfo.z.mul( selfShadow ), ao ), positionWorld );

		} )();

		mat.side = THREE.DoubleSide;

		if ( fade ) {

			mat.opacityNode = vFade;
			mat.transparent = true;
			mat.depthWrite = false; // the prepass wrote it; LessEqual keeps only the front-most surface

		}

		return mat;

	}

	// ----------------------------------------------------------------------------------------- per-frame update
	update( camera, renderer ) {

		this._pv.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
		this.frustum.setFromProjectionMatrix( this._pv, camera.coordinateSystem, camera.reversedDepth );
		this.frustum.planes.forEach( ( pl, i ) => this.planes[ i ].value.set( pl.normal.x, pl.normal.y, pl.normal.z, pl.constant ) );
		this.camPos = camera.position;
		this.cullCameraPosition.value.copy( camera.position );

		renderer.compute( this.resetPass );

		// near grid, out to the far edge of the crossfade band
		const n = Math.ceil( ( 2 * ( nearTreeRadius.value + treeCrossfade.value / 2 ) ) / TREE_CELL ) + 2;
		this.gridN.value = n;
		this.gridOrigin.value.set(
			Math.floor( camera.position.x / TREE_CELL ) - Math.floor( n / 2 ),
			Math.floor( camera.position.z / TREE_CELL ) - Math.floor( n / 2 ),
		);
		this.stats.nearCells = n * n;
		if ( this.enabledNear ) {

			this.nearCullPass.count = n * n;
			renderer.compute( this.nearCullPass );

		}

		// impostor tiles, sorted front-to-back so the GPU append order is roughly front-to-back (early-z friendly)
		this.count = 0;
		this.selected = this.selected || [];
		this.selected.length = 0;
		const rootCells = TILE << ( TIERS - 1 );
		const half = WORLD_SIZE / TREE_CELL / 2;
		for ( let cz = - half; cz < half; cz += rootCells ) {

			for ( let cx = - half; cx < half; cx += rootCells ) this.select( cx, cz, TIERS - 1 );

		}

		this.selected.sort( ( a, b ) => a.dist - b.dist );
		const data = this.tiles.value.array;
		for ( let i = 0; i < this.count; i ++ ) {

			const t = this.selected[ i ];
			data.set( [ t.cx, t.cz, 1 << t.tier, t.tier ], i * 4 );

		}

		this.tiles.value.needsUpdate = true;
		this.stats.tiles = this.count;
		this.stats.cells = this.count * TILE * TILE;
		if ( this.enabledFar && this.count > 0 ) {

			this.impostorCullPass.count = this.count * TILE * TILE;
			renderer.compute( this.impostorCullPass );

		}

		this.readbackStats( renderer );

	}

	// Async readback of the GPU counters for the HUD (throttled, never stalls the frame).
	readbackStats( renderer ) {

		const now = performance.now();
		if ( this._readPending || now - ( this._lastRead || 0 ) < 500 ) return;
		this._readPending = true;
		this._lastRead = now;
		renderer.getArrayBufferAsync( drawArgsAttr )
			.then( ( buffer ) => {

				const words = new Uint32Array( buffer );
				const n = this.allArgs().map( ( a ) => words[ a.slot * 5 + 1 ] );
				this.stats.visibleImpostors = n[ 0 ] + n[ 1 ] + n[ 2 ] + n[ 3 ];
				this.stats.visibleMeshes = n[ 4 ] + n[ 5 ];
				this.stats.crossfading = n[ 6 ] + n[ 7 ] + n[ 8 ] + n[ 9 ];

			} )
			.finally( () => this._readPending = false );

	}

	select( cx, cz, tier ) {

		const cells = TILE << tier;
		const size = cells * TREE_CELL;
		const x0 = cx * TREE_CELL, z0 = cz * TREE_CELL;
		this.world.rangeOf( x0, z0, size, this._range );
		if ( this._range.max < 1 ) return; // fully under water -> no trees
		const box = this._box;
		box.min.set( x0, this._range.min, z0 );
		box.max.set( x0 + size, this._range.max + 25, z0 + size );
		if ( ! this.frustum.intersectsBox( box ) ) return;
		const dist = box.distanceToPoint( this.camPos );
		if ( dist > treeDrawDistance.value ) return;

		if ( tier === 0 || dist >= treeTierRange0.value * Math.pow( 2, tier - 1 ) ) {

			if ( this.count >= MAX_TILES ) return;
			this.selected.push( { cx, cz, tier, dist } );
			this.count ++;
			return;

		}

		const h = cells >> 1;
		this.select( cx, cz, tier - 1 );
		this.select( cx + h, cz, tier - 1 );
		this.select( cx, cz + h, tier - 1 );
		this.select( cx + h, cz + h, tier - 1 );

	}

}
