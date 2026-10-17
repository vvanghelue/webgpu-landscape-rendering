import * as THREE from 'three/webgpu';

// Minimal pointer-lock fly camera with terrain collision.
export class FlyControls {

	constructor( camera, dom, world ) {

		this.camera = camera;
		this.dom = dom;
		this.world = world;
		this.speed = 60; // m/s
		this.yaw = 0;
		this.pitch = 0;
		this.keys = new Set();
		this.locked = false;

		dom.addEventListener( 'click', () => dom.requestPointerLock() );
		document.addEventListener( 'pointerlockchange', () => this.locked = document.pointerLockElement === dom );
		document.addEventListener( 'mousemove', ( e ) => {

			if ( ! this.locked ) return;
			this.yaw -= e.movementX * 0.0022;
			this.pitch = THREE.MathUtils.clamp( this.pitch - e.movementY * 0.0022, - 1.55, 1.55 );

		} );
		document.addEventListener( 'keydown', ( e ) => this.keys.add( e.code ) );
		document.addEventListener( 'keyup', ( e ) => this.keys.delete( e.code ) );
		dom.addEventListener( 'wheel', ( e ) => {

			this.speed = THREE.MathUtils.clamp( this.speed * ( e.deltaY > 0 ? 0.85 : 1.18 ), 2, 5000 );

		}, { passive: true } );

	}

	setView( pos, yaw, pitch ) {

		this.camera.position.copy( pos );
		this.yaw = yaw;
		this.pitch = pitch;

	}

	update( dt ) {

		const k = this.keys;
		const move = new THREE.Vector3(
			( k.has( 'KeyD' ) ? 1 : 0 ) - ( k.has( 'KeyA' ) ? 1 : 0 ),
			( k.has( 'KeyE' ) || k.has( 'Space' ) ? 1 : 0 ) - ( k.has( 'KeyQ' ) ? 1 : 0 ),
			( k.has( 'KeyS' ) ? 1 : 0 ) - ( k.has( 'KeyW' ) ? 1 : 0 ),
		);

		this.camera.quaternion.setFromEuler( new THREE.Euler( this.pitch, this.yaw, 0, 'YXZ' ) );

		if ( move.lengthSq() > 0 ) {

			const boost = k.has( 'ShiftLeft' ) || k.has( 'ShiftRight' ) ? 6 : 1;
			const up = move.y;
			move.y = 0;
			move.applyQuaternion( this.camera.quaternion );
			move.y += up;
			this.camera.position.addScaledVector( move.normalize(), this.speed * boost * dt );

		}

		const ground = Math.max( this.world.heightAt( this.camera.position.x, this.camera.position.z ), 0 );
		this.camera.position.y = Math.max( this.camera.position.y, ground + 1.8 );

	}

}
