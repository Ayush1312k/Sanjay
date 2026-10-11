/**
 * ============================================================================
 * SANJAYA AI — MINIMAL RETRO AMBIENT CANVAS
 * Subtle vintage paper grain & gentle drifting olive motes.
 * Strictly background layer (z-index: -1, pointer-events: none, non-blocking).
 * ============================================================================
 */

(function () {
    'use strict';

    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    // Enforce fixed background styles directly on element
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '-1';
    canvas.style.display = 'block';

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let dpr = 1;
    let motes = [];
    let blobs = [];
    let grainPattern = null;
    let animId = null;
    let isRunning = false;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function isDark() {
        const theme = document.documentElement.getAttribute('data-theme');
        if (theme) return theme === 'dark';
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    function createGrainPattern() {
        const size = 128;
        const offscreen = document.createElement('canvas');
        offscreen.width = size;
        offscreen.height = size;
        const oCtx = offscreen.getContext('2d');
        const imgData = oCtx.createImageData(size, size);
        const data = imgData.data;

        for (let i = 0; i < data.length; i += 4) {
            const noise = (Math.random() - 0.5) * 25;
            data[i] = 107 + noise;     // Olive R
            data[i + 1] = 112 + noise; // Olive G
            data[i + 2] = 72 + noise;  // Olive B
            data[i + 3] = Math.random() < 0.08 ? Math.floor(Math.random() * 18 + 6) : 0;
        }

        oCtx.putImageData(imgData, 0, 0);
        grainPattern = ctx.createPattern(offscreen, 'repeat');
    }

    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        initParticles();
        drawFrame();
    }

    function initParticles() {
        // Subtle ambient motes
        const count = width < 600 ? 12 : 22;
        motes = [];
        for (let i = 0; i < count; i++) {
            motes.push({
                x: Math.random() * width,
                y: Math.random() * height,
                r: Math.random() * 1.5 + 0.6,
                vy: -(Math.random() * 0.25 + 0.08),
                vx: (Math.random() - 0.5) * 0.15,
                alpha: Math.random() * 0.35 + 0.1,
                isClay: Math.random() < 0.25
            });
        }

        // Soft subtle olive ambient glow blobs
        blobs = [
            { x: width * 0.2, y: height * 0.3, r: Math.max(160, width * 0.25), vx: 0.08, vy: 0.05 },
            { x: width * 0.8, y: height * 0.7, r: Math.max(180, width * 0.3), vx: -0.06, vy: -0.04 }
        ];
    }

    function drawFrame() {
        ctx.clearRect(0, 0, width, height);

        const dark = isDark();

        // 1. Draw subtle ambient blobs
        for (let i = 0; i < blobs.length; i++) {
            const b = blobs[i];
            const grad = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
            const rgb = dark ? '84, 89, 54' : '155, 162, 123';
            const a = dark ? 0.07 : 0.045;

            grad.addColorStop(0, `rgba(${rgb}, ${a})`);
            grad.addColorStop(1, `rgba(${rgb}, 0)`);

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
            ctx.fill();
        }

        // 2. Draw subtle drifting motes
        for (let i = 0; i < motes.length; i++) {
            const m = motes[i];
            const rgb = m.isClay ? '207, 154, 111' : (dark ? '198, 203, 178' : '107, 112, 72');
            ctx.fillStyle = `rgba(${rgb}, ${m.alpha})`;
            ctx.beginPath();
            ctx.arc(m.x, m.y, m.r, 0, Math.PI * 2);
            ctx.fill();
        }

        // 3. Draw fine micro paper grain
        if (grainPattern) {
            ctx.save();
            ctx.globalAlpha = dark ? 0.06 : 0.04;
            ctx.fillStyle = grainPattern;
            ctx.fillRect(0, 0, width, height);
            ctx.restore();
        }
    }

    function update() {
        for (let i = 0; i < motes.length; i++) {
            const m = motes[i];
            m.y += m.vy;
            m.x += m.vx;

            if (m.y < -10) {
                m.y = height + 10;
                m.x = Math.random() * width;
            }
            if (m.x < -10) m.x = width + 10;
            if (m.x > width + 10) m.x = -10;
        }

        for (let i = 0; i < blobs.length; i++) {
            const b = blobs[i];
            b.x += b.vx;
            b.y += b.vy;
            if (b.x < 0 || b.x > width) b.vx *= -1;
            if (b.y < 0 || b.y > height) b.vy *= -1;
        }
    }

    function loop() {
        if (!isRunning) return;
        update();
        drawFrame();
        animId = requestAnimationFrame(loop);
    }

    function start() {
        if (isRunning || prefersReducedMotion) return;
        isRunning = true;
        animId = requestAnimationFrame(loop);
    }

    function stop() {
        isRunning = false;
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }
    }

    document.addEventListener('visibilitychange', () => {
        if (document.hidden) stop();
        else start();
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(resize, 100);
    }, { passive: true });

    createGrainPattern();
    resize();

    if (!prefersReducedMotion) {
        start();
    }
})();
