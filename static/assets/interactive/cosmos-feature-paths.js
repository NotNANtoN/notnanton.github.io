const diagram = document.querySelector('figure svg');
const figure = document.querySelector('figure');
const caption = document.getElementById('caption');
const playbackButton = document.getElementById('playback');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const baseCaption = 'Choose a path above. Hover a component to inspect it.';
const cycleDuration = 8500;
const stages = [
  { at: 0, id: 'observations', label: 'Five causal RGB observations.' },
  { at: 650, id: 'vae', label: 'The video VAE encodes five frames into two latent frames.' },
  { at: 1250, id: 'context', label: () => diagram.dataset.mode === 'proposed' ? 'Patchify only the observed latents.' : 'Append noise latents, then patchify the full context.' },
  ...Array.from({ length: 20 }, (_, i) => ({ at: 1900 + i * 90, id: 'layer-' + (i + 1), layer: true, label: 'Cosmos · layer ' + (i + 1) })),
  { at: 3760, id: 'tap-path', tap: true, label: 'Read layer-20 features; skip the remaining Cosmos blocks.' },
  { at: 4350, id: 'features', label: () => diagram.dataset.mode === 'proposed' ? '2,400 unpooled features · 204 ms extraction.' : '4,800 pooled features · 1,163 ms extraction.' },
  ...Array.from({ length: 6 }, (_, i) => ({ at: 4950 + i * 170, id: 'expert-' + (i + 1), layer: true, label: 'SmolExpert · transformer action decoding.' })),
  { at: 6300, id: 'actions', label: 'A 30-step action chunk. No video generation is needed.' }
];
let elapsed = 0;
let lastTimestamp = null;
let animationFrame = null;
let stageIndex = -1;
let userPaused = reducedMotion.matches;
let inView = false;
let hoveredTip = null;
let focusedTip = null;

function currentTip() { return focusedTip || hoveredTip; }
function wantsPlayback() { return !userPaused && inView && !document.hidden && !currentTip(); }
function labelFor(stage) { return typeof stage.label === 'function' ? stage.label() : stage.label; }
function render(force = false) {
  let next = 0;
  while (next + 1 < stages.length && elapsed >= stages[next + 1].at) next++;
  if (!force && next === stageIndex) return;
  stageIndex = next;
  const stage = stages[next];
  diagram.querySelectorAll('.tracing, .active').forEach(element => element.classList.remove('tracing', 'active'));
  if (elapsed >= 3760) document.getElementById('tap-path').classList.add('active');
  document.getElementById(stage.id).classList.add(stage.layer || stage.tap ? 'active' : 'tracing');
  diagram.dataset.phase = stage.id;
  if (!currentTip()) caption.textContent = labelFor(stage);
}
function tick(timestamp) {
  animationFrame = null;
  if (!wantsPlayback()) { lastTimestamp = null; return; }
  if (lastTimestamp !== null) elapsed = (elapsed + timestamp - lastTimestamp) % cycleDuration;
  lastTimestamp = timestamp;
  render();
  animationFrame = requestAnimationFrame(tick);
}
function syncPlayback() {
  const running = wantsPlayback();
  diagram.dataset.playing = String(running);
  playbackButton.innerHTML = userPaused ? '<span aria-hidden="true">▶</span> Play' : '<span aria-hidden="true">Ⅱ</span> Pause';
  playbackButton.setAttribute('aria-label', userPaused ? 'Play looping animation' : 'Pause animation');
  document.getElementById('motion-status').textContent = userPaused ? 'Paused' : running ? 'Auto-playing' : 'Auto-play on hold';
  if (!running) {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    animationFrame = null;
    lastTimestamp = null;
  } else if (animationFrame === null) {
    lastTimestamp = null;
    animationFrame = requestAnimationFrame(tick);
  }
}
function inspect() {
  const tip = currentTip();
  if (tip) caption.textContent = tip.dataset.tip;
  else render(true);
  syncPlayback();
}
function setMode(mode) {
  diagram.dataset.mode = mode;
  elapsed = 0;
  lastTimestamp = null;
  stageIndex = -1;
  for (const button of document.querySelectorAll('button[data-mode]')) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  document.getElementById('svg-download').href = '../img/video-vam/cosmos-feature-paths-' + mode + '.svg';
  render(true);
  syncPlayback();
  document.getElementById('live').textContent = mode === 'proposed' ? 'Proposed variant selected: 2,400 input tokens; 204 milliseconds feature extraction.' : 'Reference selected: 19,200 input tokens; 1,163 milliseconds feature extraction.';
}
for (const button of document.querySelectorAll('button[data-mode]')) button.addEventListener('click', () => setMode(button.dataset.mode));
playbackButton.addEventListener('click', () => { userPaused = !userPaused; syncPlayback(); });
for (const group of diagram.querySelectorAll('[data-tip]')) {
  group.addEventListener('pointerenter', () => { hoveredTip = group; inspect(); });
  group.addEventListener('pointerleave', () => { if (hoveredTip === group) hoveredTip = null; inspect(); });
  group.addEventListener('focus', () => { focusedTip = group; inspect(); });
  group.addEventListener('blur', () => { if (focusedTip === group) focusedTip = null; inspect(); });
}
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => { inView = entries[0].isIntersecting; syncPlayback(); }, { threshold: 0.15 });
  observer.observe(figure);
} else inView = true;
document.addEventListener('visibilitychange', syncPlayback);
reducedMotion.addEventListener('change', event => { if (event.matches) { userPaused = true; syncPlayback(); } });
window.addEventListener('pagehide', () => { if (animationFrame !== null) cancelAnimationFrame(animationFrame); animationFrame = null; lastTimestamp = null; });
window.addEventListener('pageshow', syncPlayback);
if (!userPaused) render();
else { diagram.dataset.phase = 'paused'; caption.textContent = baseCaption; }
syncPlayback();

// Same-origin embeds follow their content height, including expanded notes.
const embedMain = document.querySelector('main');
let sizeFrame = null;
function fitEmbed() {
  sizeFrame = null;
  const hostFrame = window.frameElement;
  if (!hostFrame) return;
  const height = Math.ceil(embedMain.getBoundingClientRect().height + embedMain.offsetTop + 2);
  if (hostFrame.style.height !== height + 'px') hostFrame.style.height = height + 'px';
}
function scheduleFit() {
  if (sizeFrame === null) sizeFrame = requestAnimationFrame(fitEmbed);
}
const sizeObserver = new ResizeObserver(scheduleFit);
sizeObserver.observe(embedMain);
window.addEventListener('resize', scheduleFit);
window.addEventListener('pageshow', () => { sizeObserver.observe(embedMain); scheduleFit(); });
window.addEventListener('pagehide', () => {
  sizeObserver.disconnect();
  if (sizeFrame !== null) cancelAnimationFrame(sizeFrame);
  sizeFrame = null;
});
document.fonts.ready.then(scheduleFit);
scheduleFit();
