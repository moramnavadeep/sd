import { coverTransform } from './geometry.js';
import { PoseTracker } from './smoothing.js';
import { GuideSmoother, drawFrameGuide, drawSkeleton, drawRegions, drawGuides } from './overlay.js';
import { LocalVoice } from './voice.js';
import { tr } from './i18n.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), frozen = $('frozen');
const ctx = overlay.getContext('2d');
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } } };

const S = {
  ws: null, inFlight: false, frameId: 0, captureLong: 640, minKpConf: 0.3,
  latest: null, latestAt: 0, pending: new Map(), alignedBitmap: null,
  tracker: new PoseTracker(), guideSmoother: new GuideSmoother(),
  mirrored: store.get('mirror', true), aligned: store.get('aligned', false),
  showSkel: store.get('skel', true), showRegions: store.get('regions', true),
  frameH: store.get('frameH', 88), frameY: store.get('frameY', 6),
  highlight: null, highlightUntil: 0, rtt: [], status: null, lang: store.get('lang', null), cameraOk: false,
  voiceWanted: store.get('voiceOn', true),   // voice listens by default; the button turns it off
};

// ------------------------------------------------------------------ camera
async function listCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  $('cameraSel').innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${d.label || 'Camera ' + (i + 1)}</option>`).join('');
  const saved = store.get('camera', null);
  if (saved && devs.some(d => d.deviceId === saved)) $('cameraSel').value = saved;
}

// ?demo=/static/demo/clip.mp4 (or an image) replays a local recording instead of the camera —
// for offline demos and testing. The file must be served by this local server.
async function demoStream(url) {
  const c = document.createElement('canvas');
  const isVideo = /\.(mp4|webm|mov)$/i.test(url);
  const el = isVideo ? Object.assign(document.createElement('video'), { src: url, muted: true, loop: true, playsInline: true })
                     : Object.assign(new Image(), { src: url });
  await (isVideo ? el.play() : el.decode());
  c.width = isVideo ? el.videoWidth : el.naturalWidth; c.height = isVideo ? el.videoHeight : el.naturalHeight;
  const g = c.getContext('2d');
  (function tick() { g.drawImage(el, 0, 0, c.width, c.height); requestAnimationFrame(tick); })();
  return c.captureStream(30);
}

async function startCamera() {
  const [w, h] = $('resSel').value.split('x').map(Number);
  video.srcObject?.getTracks().forEach(t => t.stop());
  const demo = new URLSearchParams(location.search).get('demo');
  if (demo && demo.startsWith('/static/')) {
    video.srcObject = await demoStream(demo); await video.play();
    S.tracker = new PoseTracker({ minConf: S.minKpConf }); S.guideSmoother.reset();
    return;
  }
  const deviceId = $('cameraSel').value || undefined;
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } }, audio: false });
  } catch (e) {
    // Not fatal: voice commands and buttons keep working without the camera.
    S.cameraOk = false;
    setStatus('warn', '—');
    $('headline').textContent = tr(S.lang, 'camera_off', { err: e.name || e.message });
    throw e;
  }
  await video.play();
  S.cameraOk = true;
  S.tracker = new PoseTracker({ minConf: S.minKpConf }); S.guideSmoother.reset();
}

// ------------------------------------------------------------------ backend session
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/session?lang=${encodeURIComponent(S.lang)}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => { S.inFlight = false; if (S.busy) { S.busy = false; setTimeout(connect, 20000); return; } setStatus('bad', tr(S.lang, 'disconnected')); setTimeout(connect, 1500); };
  ws.onopen = () => { S.inFlight = false; };
  S.ws = ws;
}

const capCanvas = document.createElement('canvas');
async function sendFrame() {
  if (S.status?.lite) return;   // free demo: no vision models on the server, frames never sent
  if (!S.ws || S.ws.readyState !== 1 || S.inFlight || video.readyState < 2) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.min(1, S.captureLong / Math.max(vw, vh));
  capCanvas.width = Math.round(vw * s); capCanvas.height = Math.round(vh * s);
  capCanvas.getContext('2d').drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
  const id = ++S.frameId, ts = performance.now();
  if (S.aligned) S.pending.set(id, await createImageBitmap(capCanvas));
  const blob = await new Promise(r => capCanvas.toBlob(r, 'image/jpeg', 0.8));
  const T = coverTransform(vw, vh, overlay.clientWidth, overlay.clientHeight, S.mirrored);
  const visible = T ? T.visibleCameraRect().map(v => Math.min(1, Math.max(0, v))) : [0, 0, 1, 1];
  const header = new TextEncoder().encode(JSON.stringify({ frame_id: id, ts, w: capCanvas.width, h: capCanvas.height, visible }));
  const buf = new Uint8Array(4 + header.length + blob.size);
  new DataView(buf.buffer).setUint32(0, header.length, true);
  buf.set(header, 4); buf.set(new Uint8Array(await blob.arrayBuffer()), 4 + header.length);
  S.inFlight = true;
  S.ws.send(buf);
}

function onMessage(m) {
  if (m.type === 'hello') { $('privacy').textContent = m.privacy; showStep(m.step); if (m.speech) voice.speak(m.speech); return; }
  if (m.type === 'error') { S.inFlight = false; return; }
  if (m.type === 'busy') { S.busy = true; $('headline').textContent = m.detail; setStatus('warn', '—'); return; }
  if (m.type === 'voice') { onVoiceReply(m); return; }
  if (m.type === 'heard') { $('voiceStatus').textContent = `${tr(S.lang, 'heard')}: “${m.text}” → ${tr(S.lang, 'not_command')}`; return; }
  if (m.type !== 'result') return;
  S.inFlight = false;
  const now = performance.now();
  S.rtt.push(now - m.client_ts); if (S.rtt.length > 30) S.rtt.shift();
  // Aligned mode: show exactly the frame these results belong to; drop older pending frames.
  if (S.aligned && S.pending.has(m.frame_id)) {
    S.alignedBitmap?.close?.(); S.alignedBitmap = S.pending.get(m.frame_id);
    for (const k of [...S.pending.keys()]) if (k <= m.frame_id) { if (k !== m.frame_id) S.pending.get(k).close?.(); S.pending.delete(k); }
  }
  S.latest = m; S.latestAt = now;
  S.tracker.update(m.pose.keypoints, now);
  renderCard(m.guidance, m);
  // Speak guidance, but not the same sentence again within 60 s: repeated prompts such as
  // "step back" kept muting the microphone and drowned out voice commands.
  const sp = m.guidance.speech;
  if (sp && !(sp === S.lastSpoken && now - S.lastSpokenAt < 60000)) {
    if (voice.speak(sp)) { S.lastSpoken = sp; S.lastSpokenAt = now; }
  }
}

// ------------------------------------------------------------------ UI text
const STATUS_CLS = { ok: 'info', correction: 'bad', uncertain: 'warn', need_position: 'warn', step_complete: 'ok' };
function setStatus(cls, text) { const c = $('statusChip'); c.className = `chip ${cls}`; c.textContent = text; }

function renderCard(g, m) {
  $('stepIndex').textContent = `${g.step.index + 1}/${g.step.total}`;
  $('stepTitle').textContent = g.step.title;
  setStatus(STATUS_CLS[g.status] || 'dim', tr(S.lang, `st_${g.status}`));
  $('headline').textContent = g.headline;
  $('instruction').textContent = g.status === 'ok' ? '' : g.step.instruction;
  $('expChip').hidden = !g.experimental;
  const sc = $('stageChip');
  if (g.stage) { sc.hidden = false; sc.textContent = `stage: ${g.stage.stage.replaceAll('_', ' ')} ${(g.stage.confidence * 100).toFixed(0)}%${g.stage.trusted ? '' : ' (low confidence)'}`; }
  else sc.hidden = true;
  $('checks').innerHTML = g.checks.filter(c => c.id !== 'full_body_visible' || c.status !== 'pass')
    .map(c => `<li class="${c.status}" title="${escapeHtml(c.message)}">${c.id.replaceAll('_', ' ')}: ${c.status}${c.experimental ? ' (exp.)' : ''}</li>`).join('');
  const rtt = S.rtt.length ? (S.rtt.reduce((a, b) => a + b, 0) / S.rtt.length).toFixed(0) : '–';
  $('diag').textContent = [
    `round trip: ${rtt} ms   server: ${m.timing_ms.total} ms`,
    `pose: ${m.pose.backend} conf ${m.pose.confidence} ${m.pose.trusted ? '' : '(below threshold)'} orient ${m.pose.orientation} | ${m.pose.latency_ms} ms`,
    `segmentation: ${m.segmentation.available ? `conf ${m.segmentation.confidence} ambiguous ${m.segmentation.ambiguous_fraction}` : 'unavailable — ' + m.segmentation.reason}`,
    `pleats: ${m.pleats.available ? `${m.pleats.source} count ${m.pleats.count} conf ${m.pleats.confidence} (${m.pleats.reason})` : 'n/a — ' + m.pleats.reason}`,
    `stage: ${m.stage.available ? `${m.stage.stage} ${m.stage.confidence} via ${m.stage.source}` : 'unavailable — ' + m.stage.reason}`,
    `tracking: ${S.tracker.state}`,
  ].join('\n');
}
const escapeHtml = (s) => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ------------------------------------------------------------------ render loop (display rate)
function frameBox(W, H) {
  const h = H * S.frameH / 100, w = h * 0.45;
  return { x: (W - w) / 2, y: H * S.frameY / 100, w, h };
}

function render() {
  const dpr = window.devicePixelRatio || 1;
  const W = overlay.clientWidth, H = overlay.clientHeight;
  if (overlay.width !== Math.round(W * dpr) || overlay.height !== Math.round(H * dpr)) {
    overlay.width = Math.round(W * dpr); overlay.height = Math.round(H * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const T = coverTransform(video.videoWidth, video.videoHeight, W, H, S.mirrored);
  if (!T) return requestAnimationFrame(render);

  if (S.aligned && S.alignedBitmap) {      // draw the analysed frame itself, mirrored the same way
    ctx.save();
    if (S.mirrored) { ctx.translate(W, 0); ctx.scale(-1, 1); }
    ctx.drawImage(S.alignedBitmap, S.mirrored ? W - T.offX - T.dispW : T.offX, T.offY, T.dispW, T.dispH);
    ctx.restore();
  }
  const now = performance.now();
  const { kps, opacity, state } = S.tracker.current(now);
  const box = frameBox(W, H);
  let inBox = false;
  if (kps) {
    const need = [0, 15, 16, 5, 6];
    inBox = need.every(i => { if (kps[i][2] < S.minKpConf) return false; const [x, y] = T.toScreen(kps[i][0], kps[i][1]);
      return x >= box.x && x <= box.x + box.w && y >= box.y && y <= box.y + box.h; });
  }
  const g = S.latest?.guidance;
  if (!g || g.status === 'need_position' || !inBox) drawFrameGuide(ctx, box, inBox, state);
  if (kps && opacity > 0) {
    if (S.showRegions && S.latest?.segmentation?.regions) drawRegions(ctx, S.latest.segmentation.regions, T, opacity);
    if (S.showSkel) drawSkeleton(ctx, kps, T, S.minKpConf, opacity);
    if (g?.guides?.length) {
      const hl = now < S.highlightUntil ? S.highlight : null;
      drawGuides(ctx, S.guideSmoother.smooth(g.guides.filter(x => x.type !== 'frame'), S.latestAt), T, opacity, hl);
    }
  }
  requestAnimationFrame(render);
}

// ------------------------------------------------------------------ commands & voice
function command(intent) { if (S.ws?.readyState === 1) S.ws.send(JSON.stringify({ type: 'command', intent })); }
function showStep(st) {
  if (!st) return;
  $('stepIndex').textContent = `${st.index + 1}/${st.total}`; $('stepTitle').textContent = st.title; $('instruction').textContent = st.instruction;
}
function onVoiceReply(m) {
  if (m.heard && m.heard.text) $('voiceStatus').textContent = `${tr(S.lang, 'heard')}: “${m.heard.text}” → ${m.intent ?? tr(S.lang, 'not_command')}`;
  if (m.speech) { voice.speak(m.speech); $('headline').textContent = m.speech; }
  if (m.show_guide) { S.highlight = m.show_guide; S.highlightUntil = performance.now() + 8000; }
  showStep(m.step);
  if (m.action === 'step') { S.guideSmoother.reset(); S.lastSpoken = null; }
}
const voice = new LocalVoice({
  onTranscript: (t) => {
    if (t.source === 'browser') {          // browser recognition: the server parses the text into a command
      if (S.ws?.readyState === 1) S.ws.send(JSON.stringify({ type: 'utterance', text: t.text }));
      $('voiceStatus').textContent = `${tr(S.lang, 'heard')}: “${t.text}”…`;
      return;
    }
    let note = `${tr(S.lang, 'heard')}: “${t.text}” → ${t.intent ?? tr(S.lang, 'not_command')}`;
    if (t.intent) {
      if (S.ws?.readyState === 1) S.ws.send(JSON.stringify({ type: 'utterance', text: t.text, intent: t.intent }));
      else note += ` (${tr(S.lang, 'not_connected')})`;
    }
    $('voiceStatus').textContent = note;
  },
  onStatus: (s) => { $('voiceStatus').textContent = s; updateMicBtn(); },
  onLevel: (level, muted) => micMeter(level, muted),
});

// Live microphone meter next to the Voice button, so it is obvious whether the mic is hearing you
// and when it is paused because the app itself is talking.
function micMeter(level, muted) {
  let el = document.getElementById('micMeter');
  if (!el) {
    el = document.createElement('span');
    el.id = 'micMeter';
    el.style.cssText = 'display:inline-flex;align-items:center;gap:6px;font-size:.8rem;color:var(--dim);margin-left:4px';
    el.innerHTML = '<span style="width:70px;height:8px;border-radius:4px;background:rgba(255,255,255,.15);overflow:hidden;display:inline-block">' +
                   '<span id="micBar" style="display:block;height:100%;width:0;background:var(--ok)"></span></span><span id="micText"></span>';
    $('micBtn').after(el);
  }
  el.hidden = !voice.listening;
  const pct = Math.min(100, Math.round(Math.sqrt(level) * 220));
  $('micBar').style.width = `${muted ? 0 : pct}%`;
  $('micText').textContent = tr(S.lang, muted ? 'meter_paused' : (pct > 12 ? 'meter_hearing' : 'meter_hint'));
}

document.querySelectorAll('[data-cmd]').forEach(b => b.addEventListener('click', () => command(b.dataset.cmd)));
function updateMicBtn() {
  const b = $('micBtn');
  b.textContent = `${tr(S.lang, 'voice')}: ${tr(S.lang, voice.listening ? 'voice_on' : 'voice_off')}`;
  b.classList.toggle('on', voice.listening);
}
async function startVoice() {
  try {
    await voice.startListening();
    // Browsers keep audio suspended until the user interacts with the page; say so instead of failing silently.
    if (voice.ctx && voice.ctx.state !== 'running') $('voiceStatus').textContent = tr(S.lang, 'mic_click');
  } catch (e) {
    $('voiceStatus').textContent = tr(S.lang, 'mic_blocked', { err: e.name || e.message });
  }
  updateMicBtn();
}
// First click / key anywhere un-suspends the audio pipeline (autoplay policy).
['pointerdown', 'keydown'].forEach(ev => window.addEventListener(ev, () => {
  if (voice.ctx && voice.ctx.state !== 'running') voice.ctx.resume().then(() => {
    if (voice.listening) $('voiceStatus').textContent = 'Listening (microphone on)';
  });
}, { capture: true }));
$('micBtn').onclick = async () => {
  S.voiceWanted = !voice.listening; store.set('voiceOn', S.voiceWanted);
  if (voice.listening) { voice.stopListening(); updateMicBtn(); } else await startVoice();
};
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const map = { n: 'next', p: 'previous', r: 'repeat', c: 'confirm', ' ': 'check' };
  if (map[e.key.toLowerCase()]) { e.preventDefault(); command(map[e.key.toLowerCase()]); }
  if (e.key.toLowerCase() === 'm') $('micBtn').click();
});

// ------------------------------------------------------------------ settings
function applyLanguage() {
  document.documentElement.lang = S.lang;
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = tr(S.lang, el.dataset.i18n); });
  $('langSel').value = S.lang;
  $('langBtn').textContent = S.lang === 'te' ? 'EN' : 'తె';
  voice.lang = S.lang;
  updateMicBtn();
}
async function setLanguage(lang) {
  if (lang === S.lang) return;
  S.lang = lang; store.set('lang', lang); applyLanguage();
  voice.stopSpeaking(); S.lastSpoken = null;
  if (S.ws?.readyState === 1) S.ws.send(JSON.stringify({ type: 'set_language', lang }));
  if (voice.listening) {
    voice.stopListening(true);
    await startVoice();
  }
  $('modelStatus').textContent = $('modelStatus').textContent.replace(/text-to-speech: .*/, `text-to-speech: ${voice.ttsLabel()}`);
}

function bindSettings() {
  $('settingsBtn').onclick = () => { $('settings').hidden = !$('settings').hidden; };
  $('langSel').onchange = () => setLanguage($('langSel').value);
  $('langBtn').onclick = () => setLanguage(S.lang === 'te' ? 'en' : 'te');
  const chk = (id, key, prop, after) => { $(id).checked = S[prop]; $(id).onchange = () => { S[prop] = $(id).checked; store.set(key, S[prop]); after?.(); }; };
  chk('mirrorChk', 'mirror', 'mirrored', () => { video.classList.toggle('mirrored', S.mirrored); S.guideSmoother.reset(); });
  chk('syncChk', 'aligned', 'aligned', () => { video.style.visibility = S.aligned ? 'hidden' : 'visible'; });
  chk('skelChk', 'skel', 'showSkel');
  chk('regionChk', 'regions', 'showRegions');
  for (const [id, key] of [['frameH', 'frameH'], ['frameY', 'frameY']]) {
    $(id).value = S[key]; $(id).oninput = () => { S[key] = Number($(id).value); store.set(key, S[key]); };
  }
  $('resSel').value = store.get('res', '1280x720');
  $('resSel').onchange = () => { store.set('res', $('resSel').value); startCamera(); };
  $('cameraSel').onchange = () => { store.set('camera', $('cameraSel').value); startCamera(); };
  video.classList.toggle('mirrored', S.mirrored);
  video.style.visibility = S.aligned ? 'hidden' : 'visible';
  // Camera resolution changes mid-stream (e.g. device switches mode): reset smoothing so overlays re-anchor.
  video.addEventListener('resize', () => { S.tracker = new PoseTracker({ minConf: S.minKpConf }); S.guideSmoother.reset(); });
}

function fatal(msg) { $('fatal').hidden = false; $('fatal').textContent = msg; }

async function loadStatus() {
  const st = await (await fetch('/api/status')).json();
  S.status = st;
  S.captureLong = st.selection.pose_input_long_side;
  S.minKpConf = st.thresholds.keypoint;
  const lines = Object.entries(st.models).map(([k, v]) => `${k}: ${v.available ? 'loaded' : 'NOT available'}${v.experimental ? ' [experimental]' : ''}\n   ${v.detail}${v.version ? ' v' + v.version : ''}`);
  if (!S.lang) S.lang = st.default_language || 'en';
  voice.serverTTS = Object.fromEntries(Object.entries(st.tts_server || {}).map(([k, v]) => [k, v.startsWith('piper')]));
  if (st.lite) {   // free online demo: recognition in the browser, Telugu replies pre-recorded
    voice.mode = 'browser'; voice.ttsMode = 'prerendered'; voice.serverTTS = { te: true };
    voice.serverASR = Object.fromEntries((st.asr.server_languages || []).map(l => [l, true]));   // Telugu
  }
  applyLanguage();
  lines.push(`speech recognition: ${Object.entries(st.asr.languages || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || st.asr.detail}`);
  lines.push(`text-to-speech: ${voice.ttsLabel()}`);
  lines.push(`profile: ${st.selection.profile} (${st.hardware.gpu || 'CPU only'})`);
  if (!st.selection.meets_minimum) lines.push('WARNING: ' + st.selection.problems.join(' '));
  $('modelStatus').textContent = lines.join('\n');
  $('privacy').textContent = st.privacy;
  if (st.hosted && !st.lite) {   // be explicit: in hosted mode processing is NOT on the user's device
    const chip = $('localChip');
    chip.textContent = tr(S.lang, 'hosted_chip'); chip.className = 'chip warn'; chip.title = st.privacy;
    $('voiceStatus').textContent = st.privacy;
  }
  if (!st.selection.meets_minimum) fatal('This device does not meet the minimum requirements:\n' + st.selection.problems.join('\n'));
  return st;
}

async function main() {
  bindSettings();
  try { await loadStatus(); } catch { fatal('Cannot reach the local server. Start it with: python -m backend.server'); return; }
  // Connect the tutorial session FIRST: voice commands and buttons must work even if the camera
  // is blocked or missing (previously a camera failure stopped here and "next" went nowhere).
  connect();
  const fps = S.status.selection.target_fps;
  setInterval(sendFrame, 1000 / fps);
  requestAnimationFrame(render);
  try { await startCamera(); await listCameras(); } catch { /* message already shown; keep running */ }
  if (S.voiceWanted && voice.asrAvailable !== false) await startVoice();   // listen without needing a click
}
main();
