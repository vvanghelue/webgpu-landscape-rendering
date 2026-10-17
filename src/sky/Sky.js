import * as THREE from 'three/webgpu';
import { positionWorld, cameraPosition, normalize } from 'three/tsl';
import { skyColor } from '../shading/common.js';

// Analytic gradient sky with sun disc. Shares skyColor() with the fog and water reflections so the horizon is seamless.
export class Sky {

	constructor() {

		const mat = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide, depthWrite: false } );
		mat.colorNode = skyColor( normalize( positionWorld.sub( cameraPosition ) ), 1.0 );
		this.mesh = new THREE.Mesh( new THREE.SphereGeometry( 1, 48, 24 ), mat );
		this.mesh.scale.setScalar( 150000 );
		this.mesh.frustumCulled = false;
		this.mesh.renderOrder = 10; // last: only shades pixels nothing else covered

	}

	update( camera ) {

		this.mesh.position.copy( camera.position );

	}

}
