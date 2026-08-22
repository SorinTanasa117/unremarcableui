// Soft end-of-session chime. Uses the Web Audio API to synthesize a gentle
// two-note descending sine tone — no external assets, no network calls, no
// licensing concerns. Lazy-initializes an AudioContext on first use so we
// comply with the browser autoplay policy (a user gesture must precede
// playback).

let ctx: AudioContext | null = null;
let masterGain: GainNode | null = null;

function getCtx(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (ctx) return ctx;
  const Ctor = window.AudioContext || (window as any).webkitAudioContext;
  if (!Ctor) return null;
  ctx = new Ctor();
  masterGain = ctx.createGain();
  masterGain.gain.value = 0.18; // Soft — easy on the ears.
  masterGain.connect(ctx.destination);
  return ctx;
}

/**
 * Resume the AudioContext if it's been suspended (browser autoplay policy).
 * Safe to call on every user interaction.
 */
export function unlockAudio(): void {
  const c = getCtx();
  if (c && c.state === 'suspended') {
    void c.resume();
  }
}

/**
 * Play a soft, short two-note descending sine chime. Volume is intentionally
 * low and the tone decays quickly so it's pleasant as a session-end cue.
 */
export function playChime(): void {
  const c = getCtx();
  if (!c || !masterGain) return;
  if (c.state === 'suspended') {
    // Best-effort resume; if it fails (no user gesture yet), silently skip.
    void c.resume();
  }

  const now = c.currentTime;
  // Notes: A5 → E5. Sine waves only — no harsh harmonics.
  const notes = [
    { freq: 880, start: now + 0.00, duration: 0.45 },
    { freq: 659.25, start: now + 0.18, duration: 0.55 },
  ];

  for (const note of notes) {
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(note.freq, note.start);

    // Quick attack, gentle exponential decay — no clicks.
    gain.gain.setValueAtTime(0.0001, note.start);
    gain.gain.exponentialRampToValueAtTime(0.6, note.start + 0.04);
    gain.gain.exponentialRampToValueAtTime(0.0001, note.start + note.duration);

    osc.connect(gain);
    gain.connect(masterGain);
    osc.start(note.start);
    osc.stop(note.start + note.duration + 0.02);
  }
}

/**
 * Play a soft ascending three-note arpeggio (C5 → E5 → G5). Intentionally
 * distinct from `playChime` (which is a descending two-note end-of-session
 * tone): this one rises, signalling "your turn — the agent paused and is
 * waiting for input." Same low master volume and sine-only timbre.
 */
export function playPauseChime(): void {
  const c = getCtx();
  if (!c || !masterGain) return;
  if (c.state === 'suspended') {
    void c.resume();
  }

  const now = c.currentTime;
  // Ascending C major triad — a gentle "attention, over to you" cue.
  const notes = [
    { freq: 523.25, start: now + 0.00, duration: 0.30 },
    { freq: 659.25, start: now + 0.14, duration: 0.32 },
    { freq: 783.99, start: now + 0.28, duration: 0.50 },
  ];

  for (const note of notes) {
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(note.freq, note.start);

    gain.gain.setValueAtTime(0.0001, note.start);
    gain.gain.exponentialRampToValueAtTime(0.6, note.start + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, note.start + note.duration);

    osc.connect(gain);
    gain.connect(masterGain);
    osc.start(note.start);
    osc.stop(note.start + note.duration + 0.02);
  }
}
