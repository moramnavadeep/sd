import { test } from 'node:test';
import assert from 'node:assert/strict';
import { coverTransform } from '../../frontend/js/geometry.js';
import { PoseTracker } from '../../frontend/js/smoothing.js';

test('cover transform: wide video in tall box crops sides, mirror flips x', () => {
  const T = coverTransform(1280, 720, 720, 1280, true);   // portrait screen, landscape camera
  assert.equal(T.scale, 1280 / 720);
  const [x, y] = T.toScreen(0.5, 0.5);
  assert.ok(Math.abs(x - 360) < 1e-6 && Math.abs(y - 640) < 1e-6);
  const [lx] = T.toScreen(0.4, 0.5), [rx] = T.toScreen(0.6, 0.5);
  assert.ok(lx > rx, 'mirroring puts camera-left on screen-right');
  const back = T.toCamera(...T.toScreen(0.42, 0.77));
  assert.ok(Math.abs(back[0] - 0.42) < 1e-9 && Math.abs(back[1] - 0.77) < 1e-9);
});

test('cover transform: aspect change re-anchors (resolution change)', () => {
  const a = coverTransform(640, 480, 1000, 600, false).toScreen(0.25, 0.25);
  const b = coverTransform(1920, 1080, 1000, 600, false).toScreen(0.25, 0.25);
  assert.notDeepEqual(a, b);
});

test('tracker smooths jitter and recovers after loss', () => {
  const tr = new PoseTracker({ lostAfterMs: 500 });
  const kp = (x) => Array.from({ length: 17 }, () => [x, 0.5, 0.9]);
  let t = 0, last;
  for (let i = 0; i < 30; i++) { t += 66; last = tr.update(kp(0.5 + (i % 2 ? 0.01 : -0.01)), t); }
  assert.ok(Math.abs(last.kps[0][0] - 0.5) < 0.01, 'jitter attenuated');
  t += 2000; const lost = tr.update(null, t);
  assert.equal(lost.state, 'lost'); assert.equal(lost.opacity, 0);
  const back = tr.update(kp(0.9), t + 66);
  assert.ok(Math.abs(back.kps[0][0] - 0.9) < 1e-6, 'reset on re-entry, no slide across screen');
});
