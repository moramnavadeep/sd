// One Euro filter (Casiez et al., CHI 2012): low jitter when still, low lag when moving.
class LowPass {
  constructor() { this.y = null; }
  filter(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; }
}
const alpha = (cutoff, dt) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };

export class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.02, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.x = new LowPass(); this.dx = new LowPass(); this.t = null;
  }
  reset() { this.x = new LowPass(); this.dx = new LowPass(); this.t = null; }
  filter(v, tMs) {
    if (this.t === null) { this.t = tMs; this.dx.filter(0, 1); return this.x.filter(v, 1); }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000); this.t = tMs;
    const d = (v - (this.x.y ?? v)) / dt;
    const ed = this.dx.filter(d, alpha(this.dCutoff, dt));
    return this.x.filter(v, alpha(this.minCutoff + this.beta * Math.abs(ed), dt));
  }
}

// Smooths 17 keypoints in normalised camera space and handles track loss / recovery.
export class PoseTracker {
  constructor({ minConf = 0.3, lostAfterMs = 700, jumpReset = 0.25 } = {}) {
    Object.assign(this, { minConf, lostAfterMs, jumpReset });
    this.filters = null; this.kps = null; this.lastSeen = 0; this.state = 'searching';
  }
  update(kps, tMs) {
    if (!kps) {
      if (this.kps && tMs - this.lastSeen > this.lostAfterMs) { this.state = 'lost'; }
      return this.current(tMs);
    }
    if (!this.filters) this.filters = kps.map(() => [new OneEuro(), new OneEuro()]);
    // Large jump after a gap (user re-entered frame) -> reset instead of sliding across the screen.
    if (this.kps && (tMs - this.lastSeen > this.lostAfterMs || this.meanJump(kps) > this.jumpReset)) {
      this.filters.forEach(f => { f[0].reset(); f[1].reset(); });
    }
    this.kps = kps.map(([x, y, c], i) => {
      if (c < this.minConf && this.kps) return [this.kps[i][0], this.kps[i][1], c];   // hold position, keep low conf
      return [this.filters[i][0].filter(x, tMs), this.filters[i][1].filter(y, tMs), c];
    });
    this.lastSeen = tMs; this.state = 'tracking';
    return this.current(tMs);
  }
  meanJump(kps) {
    let s = 0, n = 0;
    kps.forEach(([x, y, c], i) => { if (c >= this.minConf && this.kps[i][2] >= this.minConf) { s += Math.hypot(x - this.kps[i][0], y - this.kps[i][1]); n++; } });
    return n ? s / n : 0;
  }
  // Fade guides out over 400 ms once the track is lost.
  current(tMs) {
    if (!this.kps) return { kps: null, opacity: 0, state: this.state };
    const since = tMs - this.lastSeen;
    const opacity = since <= this.lostAfterMs ? 1 : Math.max(0, 1 - (since - this.lostAfterMs) / 400);
    return { kps: this.kps, opacity, state: this.state };
  }
}
