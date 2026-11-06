// Local voice I/O.
//  - Speech-to-text: microphone PCM is streamed ONLY to our loopback server (Vosk).
//    We deliberately do NOT use the browser's Web Speech recognition API, because in
//    Chrome/Edge it sends audio to the vendor's cloud.
//  - Text-to-speech: English uses browser speechSynthesis restricted to voices with
//    localService === true (installed OS voices; network voices are never used). Telugu has no
//    Windows voice, so it is synthesised by our local server (Piper) at /api/tts and played here.
//    If neither is available the text is shown only.

const WORKLET = `
class Pcm16 extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.ratio = sampleRate / 16000; this.acc = 0; this.sum = 0; this.n = 0; }
  process(inputs, outputs) {
    const ch = inputs[0][0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {           // decimate to 16 kHz, averaging each block (crude low-pass)
      this.sum += ch[i]; this.n += 1; this.acc += 1;
      if (this.acc >= this.ratio) { this.acc -= this.ratio; this.buf.push(this.sum / this.n); this.sum = 0; this.n = 0; }
    }
    if (outputs[0] && outputs[0][0]) outputs[0][0].fill(0);   // silent output; node stays in the render graph
    if (this.buf.length >= 1600) {                  // 100 ms chunks
      const out = new Int16Array(this.buf.length);
      for (let i = 0; i < this.buf.length; i++) out[i] = Math.max(-1, Math.min(1, this.buf[i])) * 0x7fff;
      this.port.postMessage(out.buffer, [out.buffer]); this.buf = [];
    }
    return true;
  }
}
registerProcessor('pcm16', Pcm16);`;

export class LocalVoice {
  constructor({ onTranscript, onStatus, onLevel }) {
    this.onTranscript = onTranscript; this.onStatus = onStatus; this.onLevel = onLevel;
    this.speakingUntil = 0; this.listening = false; this.voice = null; this.utterance = null;
    this.chunksSent = 0;
    this.lang = 'en'; this.serverTTS = {};   // lang -> available on the local server
    this.audio = null;
    this.pickVoice();
    if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = () => this.pickVoice();
  }
  pickVoice() {
    if (!('speechSynthesis' in window)) { this.voice = null; return; }
    const local = speechSynthesis.getVoices().filter(v => v.localService);
    this.voice = local.find(v => /en[-_](IN|GB|US)/i.test(v.lang)) || local.find(v => /^en/i.test(v.lang)) || local[0] || null;
  }
  get ttsAvailable() { return this.lang === 'en' ? !!this.voice : !!this.serverTTS[this.lang]; }
  ttsLabel() {
    if (this.lang === 'en') return this.voice ? `on-device voice: ${this.voice.name}` : 'no local English voice installed — text only';
    return this.serverTTS[this.lang] ? 'on-device Piper voice (local server)' : 'no local voice for this language — text only';
  }
  // The mic is muted while we talk (so we don't hear ourselves). This used to rely only on the
  // utterance's `end` event, which Chrome can drop (notably when the utterance object is garbage
  // collected) - that left the mic muted for good while still showing "Listening". Now the mute has
  // a hard deadline estimated from the text length, and the utterance is kept referenced.
  get speaking() { return performance.now() < this.speakingUntil; }
  speak(text, lang = this.lang) {
    if (!text) return false;
    if (lang !== 'en') return this.speakServer(text, lang);
    if (!this.voice) return false;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.voice = this.voice; u.rate = 0.95;
    const maxMs = 1500 + text.length * 90;                        // generous upper bound for this text
    this.speakingUntil = performance.now() + maxMs;
    u.onend = u.onerror = () => { if (this.utterance === u) this.speakingUntil = performance.now() + 300; };
    this.utterance = u;                                           // keep a reference (Chrome GC bug)
    speechSynthesis.speak(u);
    return true;
  }
  // Non-English speech: WAV synthesised by OUR loopback server (Piper), never a cloud voice.
  speakServer(text, lang) {
    if (!this.serverTTS[lang]) return false;
    this.stopSpeaking();
    this.speakingUntil = performance.now() + 2000 + text.length * 120;   // mute mic meanwhile (upper bound)
    const myAudio = new Audio();
    this.audio = myAudio;
    fetch('/api/tts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, lang }) })
      .then(r => { if (!r.ok) throw new Error(`tts ${r.status}`); return r.blob(); })
      .then(b => {
        if (this.audio !== myAudio) return;                               // superseded by newer speech
        myAudio.src = URL.createObjectURL(b);
        myAudio.onloadedmetadata = () => { this.speakingUntil = performance.now() + myAudio.duration * 1000 + 400; };
        myAudio.onended = myAudio.onerror = () => { if (this.audio === myAudio) this.speakingUntil = performance.now() + 300; };
        return myAudio.play();
      })
      .catch(() => { if (this.audio === myAudio) this.speakingUntil = 0; });
    return true;
  }
  stopSpeaking() {
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    if (this.audio) { this.audio.pause(); this.audio = null; }
    this.speakingUntil = 0;
  }
  async startListening() {
    if (this.listening) return;
    // Microphone first: if it is refused there is nothing to clean up (previously a dead audio socket was left open).
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
    this.onStatus?.('Starting on-device listening…');
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/audio?lang=${encodeURIComponent(this.lang)}`);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === 'asr_unavailable') { this.onStatus?.(`Voice commands unavailable: ${m.detail}`); this.stopListening(true); }
      if (m.type === 'asr_ready') this.onStatus?.(`Listening on-device (${m.languages.join(' + ')})`);
      if (m.type === 'transcript') this.onTranscript?.(m);
    };
    try {
      await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('cannot reach the local voice server')); });
    } catch (e) { stream.getTracks().forEach(t => t.stop()); throw e; }
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' })));
    if (ctx.state !== 'running') await ctx.resume();              // autoplay policy may start it suspended
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm16');
    node.port.onmessage = (e) => {
      const pcm = new Int16Array(e.data);
      let s = 0; for (let i = 0; i < pcm.length; i++) s += (pcm[i] / 32768) ** 2;
      const level = Math.sqrt(s / Math.max(1, pcm.length));
      const muted = this.speaking;                                // don't hear ourselves
      if (!muted && ws.readyState === 1) { ws.send(e.data); this.chunksSent++; }
      this.onLevel?.(level, muted);
    };
    // Keep the worklet pulled by the render graph: connect it to the output through a zero gain.
    const sink = ctx.createGain(); sink.gain.value = 0;
    src.connect(node); node.connect(sink); sink.connect(ctx.destination);
    ws.onclose = () => { if (this.listening) { this.onStatus?.('Voice connection to the local server closed — press Voice to restart'); this.stopListening(true); } };
    Object.assign(this, { ws, stream, ctx, node, listening: true });
  }
  stopListening(quiet = false) {
    this.listening = false;
    this.ws?.close(); this.stream?.getTracks().forEach(t => t.stop()); this.ctx?.close();
    this.ws = this.stream = this.ctx = null;
    this.onLevel?.(0, false);
    if (!quiet) this.onStatus?.('Voice off');
  }
}
