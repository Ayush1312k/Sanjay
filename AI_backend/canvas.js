/**
 * ============================================================================
 * SANJAYA AI GURU — SACRED ORGANIC BACKGROUND CANVAS MODULE
 * Monochrome Olive Palette (#6B7048) & Handcrafted Paper-Grain Aesthetics
 * ============================================================================
 * Features:
 *  - Organic drifting tonal olive blobs
 *  - Procedural micro-texture paper grain (cached offscreen pattern for 60fps)
 *  - Floating sacred leaves and shimmering ambient dust motes
 *  - Dynamic scaling by devicePixelRatio, viewport size, and device tier
 *  - Automatic visibility pause (document.hidden) & memory leak protection
 *  - Full prefers-reduced-motion support (renders single elegant static frame)
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.SanjayaCanvas = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    // Configurable Settings & Tweaks
    const CONFIG = {
        // Base Olive Tones (RGB)
        oliveRgb: [107, 112, 72],       // #6B7048
        oliveLightRgb: [155, 162, 123],  // #9BA27B
        oliveDarkRgb: [62, 66, 40],      // #3E4228
        clayRgb: [207, 154, 111],        // #CF9A6F (Clay accent specks)

        // Counts scaled by viewport width
        getBlobCount: (w) => (w < 600 ? 3 : w < 1200 ? 4 : 5),
        getLeafCount: (w) => (w < 600 ? 6 : w < 1200 ? 10 : 14),
        getMoteCount: (w) => (w < 600 ? 12 : w < 1200 ? 20 : 28),

        // Motion Speeds
        blobSpeed: 0.22,
        leafSpeedY: 0.35,
        leafSpeedX: 0.20,
        moteSpeedY: 0.18,

        // Paper Grain
        grainTileSize: 256,
        grainOpacityLight: 0.045,
        grainOpacityDark: 0.065
    };

    class OrganicCanvas {
        constructor(canvasId) {
            this.canvas = document.getElementById(canvasId);
            if (!this.canvas) return;

            this.ctx = this.canvas.getContext('2d', { alpha: true });
            this.width = 0;
            this.height = 0;
            this.dpr = 1;

            this.blobs = [];
            this.leaves = [];
            this.motes = [];

            this.grainPattern = null;
            this.animationFrameId = null;
            this.lastTimestamp = 0;
            this.isRunning = false;

            this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

            this._handleResize = this._debounce(this._resize.bind(this), 120);
            this._handleVisibilityChange = this._onVisibilityChange.bind(this);
            this._handleMotionChange = (e) => {
                this.reducedMotion = e.matches;
                if (this.reducedMotion) {
                    this.stop();
                    this.renderStaticFrame();
                } else {
                    this.start();
                }
            };

            this.init();
        }

        init() {
            this._setupEventListeners();
            this._generateGrainTexture();
            this._resize();
            this._initElements();

            if (this.reducedMotion) {
                this.renderStaticFrame();
            } else {
                this.start();
            }
        }

        _setupEventListeners() {
            window.addEventListener('resize', this._handleResize, { passive: true });
            window.addEventListener('orientationchange', this._handleResize, { passive: true });
            document.addEventListener('visibilitychange', this._handleVisibilityChange);

            try {
                const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
                if (motionQuery.addEventListener) {
                    motionQuery.addEventListener('change', this._handleMotionChange);
                } else if (motionQuery.addListener) {
                    motionQuery.addListener(this._handleMotionChange);
                }
            } catch (err) {
                // Older browsers fallback
            }
        }

        _onVisibilityChange() {
            if (document.hidden) {
                this.stop();
            } else if (!this.reducedMotion) {
                this.lastTimestamp = performance.now();
                this.start();
            }
        }

        _resize() {
            if (!this.canvas) return;

            this.dpr = Math.min(window.devicePixelRatio || 1, 2); // Cap at 2 for mobile performance
            this.width = window.innerWidth;
            this.height = window.innerHeight;

            this.canvas.width = Math.floor(this.width * this.dpr);
            this.canvas.height = Math.floor(this.height * this.dpr);
            this.canvas.style.width = this.width + 'px';
            this.canvas.style.height = this.height + 'px';

            this.ctx.scale(this.dpr, this.dpr);

            // Re-populate if count changed significantly
            this._initElements();

            if (this.reducedMotion) {
                this.renderStaticFrame();
            }
        }

        _isDarkMode() {
            const docTheme = document.documentElement.getAttribute('data-theme');
            if (docTheme) return docTheme === 'dark';
            return window.matchMedia('(prefers-color-scheme: dark)').matches;
        }

        /**
         * Generates an offscreen paper grain canvas pattern once.
         * Renders at ultra-high efficiency via createPattern.
         */
        _generateGrainTexture() {
            const size = CONFIG.grainTileSize;
            const offscreen = document.createElement('canvas');
            offscreen.width = size;
            offscreen.height = size;
            const offCtx = offscreen.getContext('2d');

            const imgData = offCtx.createImageData(size, size);
            const data = imgData.data;

            for (let i = 0; i < data.length; i += 4) {
                // Subtle tonal noise with olive tint variation
                const noise = (Math.random() - 0.5) * 35;
                const r = Math.min(255, Math.max(0, CONFIG.oliveRgb[0] + noise));
                const g = Math.min(255, Math.max(0, CONFIG.oliveRgb[1] + noise));
                const b = Math.min(255, Math.max(0, CONFIG.oliveRgb[2] + noise));

                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                // High density, very low alpha for subtle paper tactile feel
                data[i + 3] = Math.random() < 0.12 ? Math.floor(Math.random() * 28 + 12) : 0;
            }

            offCtx.putImageData(imgData, 0, 0);
            this.grainPattern = this.ctx.createPattern(offscreen, 'repeat');
        }

        _initElements() {
            const w = this.width;
            const h = this.height;

            // 1. Organic Blobs
            const blobCount = CONFIG.getBlobCount(w);
            this.blobs = [];
            for (let i = 0; i < blobCount; i++) {
                this.blobs.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    radius: Math.random() * 140 + (w < 600 ? 120 : 200),
                    vx: (Math.random() - 0.5) * CONFIG.blobSpeed,
                    vy: (Math.random() - 0.5) * CONFIG.blobSpeed,
                    pulsePhase: Math.random() * Math.PI * 2,
                    pulseSpeed: Math.random() * 0.008 + 0.004
                });
            }

            // 2. Floating Leaves
            const leafCount = CONFIG.getLeafCount(w);
            this.leaves = [];
            for (let i = 0; i < leafCount; i++) {
                this.leaves.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    size: Math.random() * 8 + 8,
                    angle: Math.random() * Math.PI * 2,
                    angularSpeed: (Math.random() - 0.5) * 0.015,
                    swayPhase: Math.random() * Math.PI * 2,
                    swaySpeed: Math.random() * 0.02 + 0.01,
                    vy: Math.random() * CONFIG.leafSpeedY + 0.15,
                    vx: (Math.random() - 0.5) * CONFIG.leafSpeedX,
                    opacity: Math.random() * 0.22 + 0.12
                });
            }

            // 3. Ambient Dust Motes
            const moteCount = CONFIG.getMoteCount(w);
            this.motes = [];
            for (let i = 0; i < moteCount; i++) {
                this.motes.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    radius: Math.random() * 1.5 + 0.8,
                    vy: -Math.random() * CONFIG.moteSpeedY - 0.08,
                    vx: (Math.random() - 0.5) * 0.15,
                    alpha: Math.random() * 0.35 + 0.1,
                    alphaPhase: Math.random() * Math.PI * 2,
                    isClay: Math.random() < 0.2 // Small percentage of clay specks (<5% overall)
                });
            }
        }

        renderStaticFrame() {
            this.ctx.clearRect(0, 0, this.width, this.height);
            this._drawBlobs();
            this._drawLeaves();
            this._drawMotes();
            this._drawGrain();
        }

        start() {
            if (this.isRunning) return;
            this.isRunning = true;
            this.lastTimestamp = performance.now();
            this._loop();
        }

        stop() {
            this.isRunning = false;
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
        }

        _loop(timestamp = performance.now()) {
            if (!this.isRunning) return;

            const delta = Math.min((timestamp - this.lastTimestamp) / 16.666, 3); // Normalise to ~60fps
            this.lastTimestamp = timestamp;

            this._update(delta);
            this.renderStaticFrame();

            this.animationFrameId = requestAnimationFrame(this._loop.bind(this));
        }

        _update(delta) {
            const w = this.width;
            const h = this.height;

            // Update Blobs
            for (let i = 0; i < this.blobs.length; i++) {
                const b = this.blobs[i];
                b.x += b.vx * delta;
                b.y += b.vy * delta;
                b.pulsePhase += b.pulseSpeed * delta;

                // Soft bounce
                if (b.x < -b.radius * 0.5) b.vx = Math.abs(b.vx);
                if (b.x > w + b.radius * 0.5) b.vx = -Math.abs(b.vx);
                if (b.y < -b.radius * 0.5) b.vy = Math.abs(b.vy);
                if (b.y > h + b.radius * 0.5) b.vy = -Math.abs(b.vy);
            }

            // Update Leaves
            for (let i = 0; i < this.leaves.length; i++) {
                const l = this.leaves[i];
                l.swayPhase += l.swaySpeed * delta;
                l.angle += l.angularSpeed * delta;
                l.x += (l.vx + Math.sin(l.swayPhase) * 0.35) * delta;
                l.y += l.vy * delta;

                // Wrap around edges
                if (l.y > h + 20) {
                    l.y = -20;
                    l.x = Math.random() * w;
                }
                if (l.x < -20) l.x = w + 20;
                if (l.x > w + 20) l.x = -20;
            }

            // Update Motes
            for (let i = 0; i < this.motes.length; i++) {
                const m = this.motes[i];
                m.x += m.vx * delta;
                m.y += m.vy * delta;
                m.alphaPhase += 0.02 * delta;

                if (m.y < -10) {
                    m.y = h + 10;
                    m.x = Math.random() * w;
                }
                if (m.x < -10) m.x = w + 10;
                if (m.x > w + 10) m.x = -10;
            }
        }

        _drawBlobs() {
            const isDark = this._isDarkMode();
            const rgb = isDark ? CONFIG.oliveDarkRgb : CONFIG.oliveLightRgb;
            const baseAlpha = isDark ? 0.09 : 0.055;

            for (let i = 0; i < this.blobs.length; i++) {
                const b = this.blobs[i];
                const currentRadius = b.radius * (1 + Math.sin(b.pulsePhase) * 0.08);

                const grad = this.ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, currentRadius);
                grad.addColorStop(0, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${baseAlpha})`);
                grad.addColorStop(0.65, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${baseAlpha * 0.45})`);
                grad.addColorStop(1, `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0)`);

                this.ctx.fillStyle = grad;
                this.ctx.beginPath();
                this.ctx.arc(b.x, b.y, currentRadius, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }

        _drawLeaves() {
            const isDark = this._isDarkMode();
            const rgb = isDark ? CONFIG.oliveLightRgb : CONFIG.oliveRgb;

            for (let i = 0; i < this.leaves.length; i++) {
                const l = this.leaves[i];
                const s = l.size;

                this.ctx.save();
                this.ctx.translate(l.x, l.y);
                this.ctx.rotate(l.angle + Math.sin(l.swayPhase) * 0.25);

                this.ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${l.opacity})`;
                this.ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${l.opacity * 0.6})`;
                this.ctx.lineWidth = 0.75;

                // Draw Sacred Leaf Silhouette via Bezier Curves
                this.ctx.beginPath();
                this.ctx.moveTo(0, -s);
                // Right leaf edge
                this.ctx.bezierCurveTo(s * 0.6, -s * 0.45, s * 0.7, s * 0.4, 0, s);
                // Left leaf edge
                this.ctx.bezierCurveTo(-s * 0.7, s * 0.4, -s * 0.6, -s * 0.45, 0, -s);
                this.ctx.closePath();
                this.ctx.fill();

                // Leaf Center Stem Vein
                this.ctx.beginPath();
                this.ctx.moveTo(0, -s * 0.85);
                this.ctx.lineTo(0, s * 0.85);
                this.ctx.stroke();

                this.ctx.restore();
            }
        }

        _drawMotes() {
            const isDark = this._isDarkMode();

            for (let i = 0; i < this.motes.length; i++) {
                const m = this.motes[i];
                const currentAlpha = Math.max(0.04, m.alpha + Math.sin(m.alphaPhase) * 0.08);
                const rgb = m.isClay
                    ? CONFIG.clayRgb
                    : (isDark ? CONFIG.oliveLightRgb : CONFIG.oliveRgb);

                this.ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${currentAlpha})`;
                this.ctx.beginPath();
                this.ctx.arc(m.x, m.y, m.radius, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }

        _drawGrain() {
            if (!this.grainPattern) return;
            const isDark = this._isDarkMode();
            this.ctx.save();
            this.ctx.globalAlpha = isDark ? CONFIG.grainOpacityDark : CONFIG.grainOpacityLight;
            this.ctx.fillStyle = this.grainPattern;
            this.ctx.fillRect(0, 0, this.width, this.height);
            this.ctx.restore();
        }

        _debounce(fn, delay) {
            let timer = null;
            return function (...args) {
                clearTimeout(timer);
                timer = setTimeout(() => fn.apply(this, args), delay);
            };
        }

        destroy() {
            this.stop();
            window.removeEventListener('resize', this._handleResize);
            window.removeEventListener('orientationchange', this._handleResize);
            document.removeEventListener('visibilitychange', this._handleVisibilityChange);
            this.canvas = null;
            this.ctx = null;
        }
    }

    // Auto-initialize when DOM is ready
    let instance = null;
    function init() {
        if (!instance) {
            instance = new OrganicCanvas('bg-canvas');
        }
        return instance;
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    return {
        init: init,
        getInstance: () => instance,
        CONFIG: CONFIG
    };
}));
