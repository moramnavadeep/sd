import { OneEuro } from './smoothing.js';

const SKELETON = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const REGION_COLORS = { waist_wrap: 'rgba(80,180,255,0.28)', pleats: 'rgba(255,190,60,0.30)', pallu: 'rgba(230,90,200,0.28)' };

// Per-point smoothing for guide primitives so they track as smoothly as the skeleton.
export class GuideSmoother {
  constructor() { this.f = new Map(); }
  pt(key, [x, y], t) {
    let f = this.f.get(key);
    if (!f) { f = [new OneEuro(1.0, 0.015), new OneEuro(1.0, 0.015)]; this.f.set(key, f); }
    return [f[0].filter(x, t), f[1].filter(y, t)];
  }
  reset() { this.f.clear(); }
  smooth(guides, t) {
    return guides.map(g => {
      const s = { ...g };
      if (g.at) s.at = this.pt(`${g.id}:at`, g.at, t);
      if (g.points) s.points = g.points.map((p, i) => this.pt(`${g.id}:p${i}`, p, t));
      if (g.polygon) s.polygon = g.polygon.map((p, i) => this.pt(`${g.id}:g${i}`, p, t));
      return s;
    });
  }
}

function label(ctx, text, x, y, color = '#fff') {
  ctx.font = '600 15px system-ui, sans-serif';
  const w = ctx.measureText(text).width + 14;
  ctx.fillStyle = 'rgba(20,16,24,0.72)';
  ctx.beginPath(); ctx.roundRect(x - w / 2, y - 26, w, 22, 6); ctx.fill();
  ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.fillText(text, x, y - 10);
}

function arrowHead(ctx, [x0, y0], [x1, y1], size = 14) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - size * Math.cos(a - 0.45), y1 - size * Math.sin(a - 0.45));
  ctx.lineTo(x1 - size * Math.cos(a + 0.45), y1 - size * Math.sin(a + 0.45));
  ctx.closePath(); ctx.fill();
}

export function drawFrameGuide(ctx, box, ok, tracking) {
  ctx.save();
  ctx.setLineDash([10, 8]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = ok ? 'rgba(90,220,140,0.9)' : 'rgba(255,255,255,0.65)';
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.setLineDash([]);
  if (!ok) label(ctx, tracking === 'tracking' ? 'Fit your whole body inside the box' : 'Step into the box', box.x + box.w / 2, box.y + 34);
  ctx.restore();
}

export function drawSkeleton(ctx, kps, T, minConf, opacity) {
  ctx.save(); ctx.globalAlpha = 0.55 * opacity;
  ctx.lineWidth = 3; ctx.strokeStyle = '#9fe8ff';
  for (const [a, b] of SKELETON) {
    if (kps[a][2] < minConf || kps[b][2] < minConf) continue;
    const p = T.toScreen(kps[a][0], kps[a][1]), q = T.toScreen(kps[b][0], kps[b][1]);
    ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke();
  }
  ctx.fillStyle = '#ffffff';
  kps.forEach(([x, y, c]) => { if (c >= minConf) { const [px, py] = T.toScreen(x, y); ctx.beginPath(); ctx.arc(px, py, 4, 0, 7); ctx.fill(); } });
  ctx.restore();
}

export function drawRegions(ctx, regions, T, opacity) {
  ctx.save(); ctx.globalAlpha = opacity;
  for (const [cls, polys] of Object.entries(regions || {})) {
    ctx.fillStyle = REGION_COLORS[cls] || 'rgba(255,255,255,0.2)';
    for (const poly of polys) {
      ctx.beginPath();
      poly.forEach(([x, y], i) => { const [px, py] = T.toScreen(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath(); ctx.fill();
    }
  }
  ctx.restore();
}

export function drawGuides(ctx, guides, T, opacity, highlightId) {
  ctx.save(); ctx.globalAlpha = opacity;
  for (const g of guides) {
    const focus = g.focus || g.id === highlightId;
    const col = focus ? '#ffcf4a' : '#ffffff';
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = focus ? 5 : 3;
    if (g.type === 'marker') {
      const [x, y] = T.toScreen(...g.at);
      const r = 12 + (focus ? 4 * Math.sin(performance.now() / 180) : 0);
      ctx.beginPath(); ctx.arc(x, y, Math.abs(r), 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
      if (g.label) label(ctx, g.label, x, y - 14, col);
    } else if (g.type === 'arrow' || g.type === 'line') {
      const pts = g.points.map(p => T.toScreen(...p));
      ctx.setLineDash(g.type === 'line' ? [8, 6] : []);
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.stroke();
      ctx.setLineDash([]);
      if (g.type === 'arrow') arrowHead(ctx, pts[pts.length - 2], pts[pts.length - 1]);
      if (g.label) { const m = pts[Math.floor(pts.length / 2)]; label(ctx, g.label, m[0], m[1] - 8, col); }
    } else if (g.type === 'zone') {
      const pts = g.polygon.map(p => T.toScreen(...p));
      ctx.setLineDash([6, 6]);
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.closePath(); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.12 * opacity; ctx.fill(); ctx.globalAlpha = opacity;
      if (g.label) label(ctx, g.label, (pts[0][0] + pts[1][0]) / 2, Math.min(pts[0][1], pts[1][1]) - 6, col);
    } else if (g.type === 'folds') {
      ctx.strokeStyle = 'rgba(255,207,74,0.85)'; ctx.lineWidth = 2;
      for (const [a, b] of g.lines) { const p = T.toScreen(...a), q = T.toScreen(...b); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke(); }
      if (g.experimental) { const p = T.toScreen(...g.lines[0][0]); label(ctx, 'experimental fold detection', p[0], p[1] - 6, '#ffcf4a'); }
    }
  }
  ctx.restore();
}
