/**
 * Asteroid Trajectory & Earth Risk Analysis - 3D Keplerian Orbit Visualizer
 * Powered by Three.js
 */

class OrbitVisualizer3D {
  constructor(canvasContainerId) {
    this.container = document.getElementById(canvasContainerId);
    if (!this.container) return;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.animationId = null;

    // Meshes and lines
    this.sunMesh = null;
    this.earthOrbitLine = null;
    this.earthMesh = null;
    this.marsOrbitLine = null;
    this.asteroidOrbitLine = null;
    this.asteroidMesh = null;
    this.approachDashedLine = null;

    // Trajectory data
    this.trajectoryData = null;
    this.currentStepIdx = 0;
    this.isPlaying = false;
    this.playInterval = null;

    // Scale factor: 1 AU = 15 units in Three.js world space
    this.AU_SCALE = 15.0;

    this.initScene();
    this.bindEvents();
  }

  initScene() {
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 580;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x030712);

    // Starfield background
    this.createStarfield();

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(0, 32, 42);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls (if available)
    if (typeof THREE.OrbitControls !== "undefined") {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxDistance = 150;
      this.controls.minDistance = 2;
    }

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0x334155, 0.8);
    this.scene.add(ambientLight);

    const sunLight = new THREE.PointLight(0xfffbeb, 2.5, 300, 0.5);
    sunLight.position.set(0, 0, 0);
    this.scene.add(sunLight);

    // 6. Heliocentric Celestial Reference Grid (Ecliptic Plane)
    const gridHelper = new THREE.GridHelper(50, 20, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -0.01;
    this.scene.add(gridHelper);

    // 7. Sun (Origin)
    const sunGeo = new THREE.SphereGeometry(1.2, 32, 32);
    const sunMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
    this.sunMesh = new THREE.Mesh(sunGeo, sunMat);
    this.scene.add(this.sunMesh);

    // Sun Glow Halo
    const glowGeo = new THREE.SphereGeometry(1.6, 32, 32);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.25,
      side: THREE.BackSide
    });
    const sunGlow = new THREE.Mesh(glowGeo, glowMat);
    this.scene.add(sunGlow);

    // 8. Earth & Asteroid Meshes
    const earthGeo = new THREE.SphereGeometry(0.55, 24, 24);
    const earthMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.3,
      metalness: 0.2
    });
    this.earthMesh = new THREE.Mesh(earthGeo, earthMat);
    
    // Earth Atmosphere 3D Glow
    const earthAtmGeo = new THREE.SphereGeometry(0.72, 24, 24);
    const earthAtmMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.25,
      side: THREE.BackSide
    });
    this.earthAtm = new THREE.Mesh(earthAtmGeo, earthAtmMat);
    this.earthMesh.add(this.earthAtm);
    this.scene.add(this.earthMesh);

    const astGeo = new THREE.DodecahedronGeometry(0.44, 1);
    const astMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      roughness: 0.8,
      metalness: 0.3
    });
    this.asteroidMesh = new THREE.Mesh(astGeo, astMat);

    // Asteroid Hazard Glow
    const astGlowGeo = new THREE.SphereGeometry(0.62, 16, 16);
    const astGlowMat = new THREE.MeshBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.3,
      side: THREE.BackSide
    });
    this.astGlow = new THREE.Mesh(astGlowGeo, astGlowMat);
    this.asteroidMesh.add(this.astGlow);
    this.scene.add(this.asteroidMesh);

    // 9. Distance dashed connection line
    const dashedMat = new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 0.6,
      gapSize: 0.4,
      transparent: true,
      opacity: 0.7
    });
    const dashedGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 0)
    ]);
    this.approachDashedLine = new THREE.Line(dashedGeo, dashedMat);
    this.approachDashedLine.computeLineDistances();
    this.scene.add(this.approachDashedLine);

    // Start render loop
    this.animate = this.animate.bind(this);
    this.animate();

    // Resize handler
    window.addEventListener("resize", () => this.handleResize());
  }

  createStarfield() {
    const starsGeo = new THREE.BufferGeometry();
    const starCount = 800;
    const positions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 400;
      positions[i + 1] = (Math.random() - 0.5) * 400;
      positions[i + 2] = (Math.random() - 0.5) * 400;
    }
    starsGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const starsMat = new THREE.PointsMaterial({
      color: 0x94a3b8,
      size: 0.8,
      transparent: true,
      opacity: 0.6
    });
    const starField = new THREE.Points(starsGeo, starsMat);
    this.scene.add(starField);
  }

  bindEvents() {
    const resetBtn = document.getElementById("btn-reset-cam");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => this.setCameraView("perspective"));
    }
    const topBtn = document.getElementById("btn-top-cam");
    if (topBtn) {
      topBtn.addEventListener("click", () => this.setCameraView("top"));
    }
    const sideBtn = document.getElementById("btn-side-cam");
    if (sideBtn) {
      sideBtn.addEventListener("click", () => this.setCameraView("side"));
    }
    const slider = document.getElementById("time-scrub-slider");
    if (slider) {
      slider.addEventListener("input", (e) => {
        this.currentStepIdx = parseInt(e.target.value, 10);
        this.updateStepDisplay();
      });
    }
    const playBtn = document.getElementById("btn-play-orbit");
    if (playBtn) {
      playBtn.addEventListener("click", () => this.togglePlayback());
    }
  }

  setCameraView(view) {
    if (view === "top") {
      this.camera.position.set(0, 50, 0.001);
      this.camera.lookAt(0, 0, 0);
    } else if (view === "side") {
      this.camera.position.set(0, 1, 50);
      this.camera.lookAt(0, 0, 0);
    } else {
      this.camera.position.set(0, 32, 42);
      this.camera.lookAt(0, 0, 0);
    }
    if (this.controls) this.controls.target.set(0, 0, 0);
  }

  handleResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight || 580;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    this.animationId = requestAnimationFrame(this.animate);
    if (this.controls) this.controls.update();

    // Subtle sun pulsing
    if (this.sunMesh) {
      this.sunMesh.rotation.y += 0.002;
    }
    if (this.asteroidMesh) {
      this.asteroidMesh.rotation.x += 0.01;
      this.asteroidMesh.rotation.y += 0.015;
    }

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Load trajectory data returned by /api/orbit/propagate
   */
  loadTrajectory(data) {
    this.trajectoryData = data;
    this.currentStepIdx = 0;

    // Remove existing orbit lines
    if (this.earthOrbitLine) this.scene.remove(this.earthOrbitLine);
    if (this.marsOrbitLine) this.scene.remove(this.marsOrbitLine);
    if (this.asteroidOrbitLine) this.scene.remove(this.asteroidOrbitLine);

    const S = this.AU_SCALE;

    // 1. Draw Earth Orbit Ellipse
    if (data.earth_orbit_path && data.earth_orbit_path.length > 0) {
      const earthPoints = data.earth_orbit_path.map(
        (p) => new THREE.Vector3(p.x * S, p.z * S, -p.y * S) // Y is up in Three.js, so ecliptic z -> y, y -> -z
      );
      const earthGeo = new THREE.BufferGeometry().setFromPoints(earthPoints);
      const earthLineMat = new THREE.LineBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.65,
        linewidth: 1
      });
      this.earthOrbitLine = new THREE.Line(earthGeo, earthLineMat);
      this.scene.add(this.earthOrbitLine);
    }

    // 2. Draw Mars Reference Orbit
    if (data.mars_orbit_path && data.mars_orbit_path.length > 0) {
      const marsPoints = data.mars_orbit_path.map(
        (p) => new THREE.Vector3(p.x * S, p.z * S, -p.y * S)
      );
      const marsGeo = new THREE.BufferGeometry().setFromPoints(marsPoints);
      const marsLineMat = new THREE.LineBasicMaterial({
        color: 0xf97316,
        transparent: true,
        opacity: 0.25
      });
      this.marsOrbitLine = new THREE.Line(marsGeo, marsLineMat);
      this.scene.add(this.marsOrbitLine);
    }

    // 3. Draw Asteroid Orbit Ellipse
    if (data.asteroid_orbit_path && data.asteroid_orbit_path.length > 0) {
      const astPoints = data.asteroid_orbit_path.map(
        (p) => new THREE.Vector3(p.x * S, p.z * S, -p.y * S)
      );
      const astGeo = new THREE.BufferGeometry().setFromPoints(astPoints);
      
      const isCritical = data.closest_approach_in_window && data.closest_approach_in_window.min_distance_au <= 0.05;
      const lineColor = isCritical ? 0xef4444 : 0xf59e0b;
      
      const astLineMat = new THREE.LineBasicMaterial({
        color: lineColor,
        transparent: true,
        opacity: 0.9,
        linewidth: 2
      });
      this.asteroidOrbitLine = new THREE.Line(astGeo, astLineMat);
      this.scene.add(this.asteroidOrbitLine);

      // Color asteroid mesh and glow
      if (this.asteroidMesh) {
        this.asteroidMesh.material.color.setHex(lineColor);
      }
      if (this.astGlow) {
        this.astGlow.material.color.setHex(lineColor);
      }
    }

    // Setup slider max
    const slider = document.getElementById("time-scrub-slider");
    if (slider && data.time_series) {
      slider.max = data.time_series.length - 1;
      slider.value = 0;
    }

    this.updateStepDisplay();
  }

  updateStepDisplay() {
    if (!this.trajectoryData || !this.trajectoryData.time_series) return;
    const series = this.trajectoryData.time_series;
    if (this.currentStepIdx >= series.length) {
      this.currentStepIdx = 0;
    }
    const step = series[this.currentStepIdx];
    const S = this.AU_SCALE;

    // Position Earth
    const ep = step.earth_pos;
    const earthVec = new THREE.Vector3(ep.x * S, ep.z * S, -ep.y * S);
    if (this.earthMesh) {
      this.earthMesh.position.copy(earthVec);
    }

    // Position Asteroid
    const ap = step.asteroid_pos;
    const astVec = new THREE.Vector3(ap.x * S, ap.z * S, -ap.y * S);
    if (this.asteroidMesh) {
      this.asteroidMesh.position.copy(astVec);
    }

    // Update Distance Line
    if (this.approachDashedLine) {
      const positions = this.approachDashedLine.geometry.attributes.position.array;
      positions[0] = earthVec.x;
      positions[1] = earthVec.y;
      positions[2] = earthVec.z;
      positions[3] = astVec.x;
      positions[4] = astVec.y;
      positions[5] = astVec.z;
      this.approachDashedLine.geometry.attributes.position.needsUpdate = true;
      this.approachDashedLine.computeLineDistances();
    }

    // Update Telemetry Panel Elements
    const dateEl = document.getElementById("tel-current-date");
    if (dateEl) dateEl.innerText = step.date;

    const daysEl = document.getElementById("tel-days-offset");
    if (daysEl) daysEl.innerText = `+${step.days_from_now} days`;

    const distAuEl = document.getElementById("tel-dist-au");
    if (distAuEl) distAuEl.innerText = `${step.distance_au.toFixed(4)} AU`;

    const distLdEl = document.getElementById("tel-dist-ld");
    if (distLdEl) {
      distLdEl.innerText = `${step.distance_ld.toFixed(1)} LD`;
      distLdEl.className = step.distance_au <= 0.05 ? "tel-val text-danger" : "tel-val text-success";
    }

    const distKmEl = document.getElementById("tel-dist-km");
    if (distKmEl) distKmEl.innerText = `${step.distance_km.toLocaleString()} km`;

    const anomEl = document.getElementById("tel-true-anomaly");
    if (anomEl) anomEl.innerText = `${step.asteroid_pos.true_anomaly_deg}°`;
  }

  togglePlayback() {
    this.isPlaying = !this.isPlaying;
    const playBtn = document.getElementById("btn-play-orbit");
    if (this.isPlaying) {
      if (playBtn) playBtn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
      this.playInterval = setInterval(() => {
        if (!this.trajectoryData || !this.trajectoryData.time_series) return;
        this.currentStepIdx = (this.currentStepIdx + 1) % this.trajectoryData.time_series.length;
        const slider = document.getElementById("time-scrub-slider");
        if (slider) slider.value = this.currentStepIdx;
        this.updateStepDisplay();
      }, 100);
    } else {
      if (playBtn) playBtn.innerHTML = '<i class="fa-solid fa-play"></i> Animate Orbit';
      clearInterval(this.playInterval);
    }
  }
}

// Global instance attached to window for coordination
window.OrbitVisualizer3D = OrbitVisualizer3D;
