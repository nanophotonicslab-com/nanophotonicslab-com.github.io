/** Three.js 3D viewport for BEM mesh visualization. */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { viridis } from './colormap';

export class BEMViewport {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private meshObj: THREE.Mesh | null = null;
  private wireframe: THREE.LineSegments | null = null;
  private showWireframe = false;

  constructor(container: HTMLElement) {
    const w = container.clientWidth;
    const h = container.clientHeight;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0xf8f7ff);  // --bg-soft
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 10000);
    this.camera.position.set(80, 60, 80);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(50, 80, 60);
    this.scene.add(dir);

    // Grid
    const grid = new THREE.GridHelper(200, 20, 0xa5b4fc, 0xf1f0fb);
    grid.rotation.x = Math.PI / 2;  // XY plane
    this.scene.add(grid);

    // Axes
    this.scene.add(new THREE.AxesHelper(40));

    // Resize observer
    const ro = new ResizeObserver(() => this.resize());
    ro.observe(container);

    this.animate();
  }

  setMesh(vertices: Float64Array, faces: Int32Array, nFaces: number, nVerts: number): void {
    // Remove old mesh
    if (this.meshObj) this.scene.remove(this.meshObj);
    if (this.wireframe) this.scene.remove(this.wireframe);

    // Build non-indexed geometry (for per-face coloring)
    const positions = new Float32Array(nFaces * 3 * 3);
    const colors = new Float32Array(nFaces * 3 * 3);

    for (let f = 0; f < nFaces; f++) {
      const i0 = faces[f * 3], i1 = faces[f * 3 + 1], i2 = faces[f * 3 + 2];
      for (let v = 0; v < 3; v++) {
        const vi = [i0, i1, i2][v];
        positions[f * 9 + v * 3 + 0] = vertices[vi * 3 + 0];
        positions[f * 9 + v * 3 + 1] = vertices[vi * 3 + 1];
        positions[f * 9 + v * 3 + 2] = vertices[vi * 3 + 2];
        // Default: light indigo
        colors[f * 9 + v * 3 + 0] = 0.65;
        colors[f * 9 + v * 3 + 1] = 0.71;
        colors[f * 9 + v * 3 + 2] = 0.95;
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: true,
    });

    this.meshObj = new THREE.Mesh(geom, mat);
    this.scene.add(this.meshObj);

    // Wireframe overlay
    const wireGeom = new THREE.WireframeGeometry(geom);
    const wireMat = new THREE.LineBasicMaterial({ color: 0x4f46e5, opacity: 0.3, transparent: true });
    this.wireframe = new THREE.LineSegments(wireGeom, wireMat);
    this.wireframe.visible = this.showWireframe;
    this.scene.add(this.wireframe);

    // Auto-fit camera
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(80, 60, 80);
    this.controls.update();
  }

  setFaceColors(enhancement: Float64Array, minVal?: number, maxVal?: number): void {
    if (!this.meshObj) return;

    const geom = this.meshObj.geometry;
    const colors = geom.getAttribute('color') as THREE.BufferAttribute;
    const nFaces = enhancement.length;

    const eMin = minVal ?? Math.min(...Array.from(enhancement));
    const eMax = maxVal ?? Math.max(...Array.from(enhancement));
    const range = Math.max(eMax - eMin, 1e-10);

    for (let f = 0; f < nFaces; f++) {
      const t = (enhancement[f] - eMin) / range;
      const [r, g, b] = viridis(Math.max(0, Math.min(1, t)));
      for (let v = 0; v < 3; v++) {
        colors.setXYZ(f * 3 + v, r, g, b);
      }
    }
    colors.needsUpdate = true;
  }

  toggleWireframe(): boolean {
    this.showWireframe = !this.showWireframe;
    if (this.wireframe) this.wireframe.visible = this.showWireframe;
    return this.showWireframe;
  }

  resetView(): void {
    this.camera.position.set(80, 60, 80);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  screenshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  private resize(): void {
    const container = this.renderer.domElement.parentElement;
    if (!container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private animate = (): void => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
