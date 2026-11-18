/**
 * UI 3D Engine - Immersive Spatial Depth, Mouse Parallax, Card Tilts & Background Canvas
 */

(function () {
  // 1. Interactive 3D Card Tilt with Specular Lighting
  function init3DCardTilt() {
    const tiltElements = document.querySelectorAll(
      ".metric-card, .alert-card, .chart-card, .prediction-form-card, .prediction-results-card, .orbit-sidebar-card, .param-card"
    );

    tiltElements.forEach((el) => {
      // Create glare overlay if not present
      if (!el.querySelector(".card-3d-glare")) {
        const glare = document.createElement("div");
        glare.className = "card-3d-glare";
        el.appendChild(glare);
      }

      let bounds = null;

      function onMouseEnter() {
        bounds = el.getBoundingClientRect();
        el.style.transition = "transform 0.1s ease-out, box-shadow 0.2s ease";
      }

      function onMouseMove(e) {
        if (!bounds) bounds = el.getBoundingClientRect();
        const mouseX = e.clientX - bounds.left;
        const mouseY = e.clientY - bounds.top;

        const xPct = mouseX / bounds.width - 0.5; // -0.5 to 0.5
        const yPct = mouseY / bounds.height - 0.5;

        // 3D rotation angles
        const rotateY = xPct * 14; // degrees
        const rotateX = -yPct * 14;

        el.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateZ(10px)`;

        // Dynamic Glare
        const glare = el.querySelector(".card-3d-glare");
        if (glare) {
          const glareX = (mouseX / bounds.width) * 100;
          const glareY = (mouseY / bounds.height) * 100;
          glare.style.background = `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(255, 255, 255, 0.18) 0%, transparent 65%)`;
          glare.style.opacity = "1";
        }
      }

      function onMouseLeave() {
        bounds = null;
        el.style.transition = "transform 0.5s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.5s ease";
        el.style.transform = "perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)";
        const glare = el.querySelector(".card-3d-glare");
        if (glare) {
          glare.style.opacity = "0";
        }
      }

      el.addEventListener("mouseenter", onMouseEnter);
      el.addEventListener("mousemove", onMouseMove);
      el.addEventListener("mouseleave", onMouseLeave);
    });
  }

  // 2. Interactive 3D Space Background Parallax Canvas
  function init3DBackground() {
    let canvas = document.getElementById("bg-3d-canvas");
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.id = "bg-3d-canvas";
      document.body.prepend(canvas);
    }

    const ctx = canvas.getContext("2d");
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    window.addEventListener("resize", () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    });

    // 3D Star / Particle Nodes
    const stars = [];
    const NUM_STARS = 160;
    for (let i = 0; i < NUM_STARS; i++) {
      stars.push({
        x: (Math.random() - 0.5) * width * 2,
        y: (Math.random() - 0.5) * height * 2,
        z: Math.random() * 1000 + 100, // Depth
        size: Math.random() * 1.6 + 0.6,
        color: Math.random() > 0.4 ? "#38bdf8" : Math.random() > 0.5 ? "#818cf8" : "#f43f5e"
      });
    }

    let mouseTargetX = 0;
    let mouseTargetY = 0;
    let mouseCurrentX = 0;
    let mouseCurrentY = 0;

    window.addEventListener("mousemove", (e) => {
      mouseTargetX = (e.clientX - width / 2) * 0.05;
      mouseTargetY = (e.clientY - height / 2) * 0.05;
    });

    function render() {
      // Smooth mouse parallax lerp
      mouseCurrentX += (mouseTargetX - mouseCurrentX) * 0.05;
      mouseCurrentY += (mouseTargetY - mouseCurrentY) * 0.05;

      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;

      for (let i = 0; i < NUM_STARS; i++) {
        const s = stars[i];

        // Move stars forward in 3D
        s.z -= 0.35;
        if (s.z <= 10) {
          s.z = 1000;
          s.x = (Math.random() - 0.5) * width * 2;
          s.y = (Math.random() - 0.5) * height * 2;
        }

        // 3D Perspective Projection
        const fov = 350;
        const scale = fov / (fov + s.z);
        const px = (s.x + mouseCurrentX * 3) * scale + cx;
        const py = (s.y + mouseCurrentY * 3) * scale + cy;

        if (px >= 0 && px <= width && py >= 0 && py <= height) {
          const alpha = Math.min(1, (1000 - s.z) / 700);
          const rad = Math.max(0.6, s.size * scale * 2.2);

          ctx.beginPath();
          ctx.arc(px, py, rad, 0, Math.PI * 2);
          ctx.fillStyle = s.color;
          ctx.globalAlpha = alpha * 0.45;
          ctx.fill();

          // Subtle glow for closest stars
          if (s.z < 350) {
            ctx.beginPath();
            ctx.arc(px, py, rad * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = s.color;
            ctx.globalAlpha = alpha * 0.15;
            ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1.0;
      requestAnimationFrame(render);
    }
    render();
  }

  // Observe DOM for newly added elements to attach 3D tilt
  function observeDynamicElements() {
    const observer = new MutationObserver((mutations) => {
      let needsInit = false;
      mutations.forEach((m) => {
        if (m.addedNodes.length > 0) needsInit = true;
      });
      if (needsInit) {
        init3DCardTilt();
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  document.addEventListener("DOMContentLoaded", () => {
    init3DBackground();
    init3DCardTilt();
    observeDynamicElements();
  });
})();
