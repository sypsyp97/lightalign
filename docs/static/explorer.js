// Step-by-step view of 600 CT-RATE validation scans (circles) and their reports (triangles).
// Every point is a real feature vector projected on two fixed axes in the report feature space. The reports stay in
// place. The scans move because the map W from scan features to report features changes from step to step.
const root = document.getElementById("explorer");
const canvas = root.querySelector("canvas");
const ctx = canvas.getContext("2d");
const tip = root.querySelector(".ex-tip");
const select = root.querySelector("select");
const steps = [...root.querySelectorAll(".ex-steps button")];
const replay = root.querySelector(".ex-replay");
const fname = root.querySelector(".ex-fname");
const title = root.querySelector(".ex-step-title");
const body = root.querySelector(".ex-step-body");
const num = root.querySelector(".ex-step-num");
const reduce = matchMedia("(prefers-reduced-motion: reduce)");

const COL = { pos: "#d0562c", neg: "#9aa5b4", ink: "#23262b", muted: "#7a8089", near: "#2f5f8a" };
const R = 3.4;
const fmt = new Intl.NumberFormat("en-US");
let D, W = 0, H = 0, dpr = 1;
let state = "separate", hover = -1, pinned = -1;
let ct = [], rep = [], from = null, to = null, t0 = 0, dur = 0, raf = 0, auto = [];
const cue = root.querySelector(".ex-cue");
let cueItem = -1, touched = false;     // the example scan that the cue ring marks, until the first interaction

// Step text: ASD-STE100 style (short sentences, active voice, one topic per paragraph).
const TEXT = {
  separate: ["Two frozen encoders",
    "VoxelFM gives each CT scan a feature vector (circle). Qwen3.5-9B gives each report a feature vector (triangle). <b>The two spaces have no common axes.</b>"],
  0: ["Zero pairs",
    "A zero-pair aligner calculates the map W. On CT, <b>the map fails: the scans land at random.</b>"],
  200: ["200 pairs",
    "Ridge regression on 200 pairs calculates W. <b>The scans move toward their own reports.</b>"],
  5994: ["All 5,994 pairs",
    "With all training pairs, <b>the scans move nearer to their own reports.</b>"],
};

function layout(s) {
  if (s === "separate") {
    return { ct: D.ct0.map(([x, y]) => [x * 0.46 - 0.54, y * 0.46]), rep: D.rep.map(([x, y]) => [x * 0.46 + 0.54, y * 0.46]) };
  }
  return { ct: D.lv[s].xy, rep: D.rep };
}

const PAD = { l: 26, r: 8, t: 8, b: 28 };
let BOX = { cx: 0, cy: 0, hw: 1, hh: 1 };      // extent of every report-space layout, set after load
function toScreen([x, y]) {
  const w = W - PAD.l - PAD.r, h = H - PAD.t - PAD.b;
  if (state === "separate") return [PAD.l + w / 2 + x * w / 2, PAD.t + h / 2 - y * h / 2 * 0.9];
  const s = Math.min(w / 2 / BOX.hw, h / 2 / BOX.hh) * 0.96;     // equal scale: plot distances stay true
  return [PAD.l + w / 2 + (x - BOX.cx) * s, PAD.t + h / 2 - (y - BOX.cy) * s];
}

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

function go(s, ms = 1300) {
  const target = layout(s);
  state = s;
  steps.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.step === s)));
  hover = pinned = -1; tip.hidden = true;
  showText();
  cue.hidden = true;
  if (reduce.matches || ms === 0 || !ct.length) { ct = target.ct; rep = target.rep; from = to = null; draw(); placeCue(); }
  else { from = { ct, rep }; to = target; t0 = performance.now(); dur = ms; loop(); }
}

function loop() {
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame((now) => {
    if (!to) return;
    const k = ease(Math.min(1, (now - t0) / dur));
    const mix = (a, b) => a.map((p, i) => [p[0] + (b[i][0] - p[0]) * k, p[1] + (b[i][1] - p[1]) * k]);
    ct = mix(from.ct, to.ct); rep = mix(from.rep, to.rep);
    draw();
    if (k < 1) loop(); else { ct = to.ct; rep = to.rep; from = to = null; draw(); placeCue(); }
  });
}

const finding = () => Number(select.value);
const has = (i, f) => (D.cases[i].lab >> f) & 1;

// share of the scans with finding f whose 3 nearest reports include at least 2 that state f
function hit(s, f) {
  let pos = 0, ok = 0;
  D.lv[s].top3.forEach((near, i) => {
    if (!has(i, f)) return;
    pos++;
    if (near.filter((j) => has(j, f)).length >= 2) ok++;
  });
  return pos ? ok / pos : 0;
}

function showText() {
  const [t, b] = TEXT[state];
  title.textContent = t; body.innerHTML = b;     // fixed strings from TEXT, with <b> for the key clause
  if (state === "separate") { num.innerHTML = ""; return; }
  const L = D.lv[state], f = finding();
  num.innerHTML =
    `<dt>To own report</dt><dd>${L.dOwn.toFixed(2)}</dd>` +
    `<dt>To a random report</dt><dd>${L.dRand.toFixed(2)}</dd>` +
    `<dt class="pos">Orange scans near orange reports</dt><dd class="pos">${Math.round(hit(state, f) * 100)}&#8239;%</dd>`;
}

function tri(x, y, r) {
  ctx.moveTo(x, y - r * 1.25); ctx.lineTo(x + r * 1.1, y + r * 0.75); ctx.lineTo(x - r * 1.1, y + r * 0.75); ctx.closePath();
}

// name an axis end by the finding that correlates most with it (|r| > 0.15, matching sign); none: "fewer findings"
function axisName(end, sign) {
  const hit = end.find(([, r]) => r * sign > 0.15);
  return hit ? `more ${hit[0].toLowerCase()}` : "fewer findings";
}

function drawAxes() {
  ctx.font = "500 11.5px Inter, system-ui, sans-serif"; ctx.fillStyle = COL.muted;
  if (state === "separate") {
    ctx.textAlign = "center";
    const y = H - 10;
    ctx.fillText("Scan features (VoxelFM)", PAD.l + (W - PAD.l - PAD.r) * 0.23, y);
    ctx.fillText("Report features (Qwen3.5-9B)", PAD.l + (W - PAD.l - PAD.r) * 0.77, y);
    return;
  }
  const [ax, ay] = D.axes;
  ctx.strokeStyle = "#d5d9df"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.l, H - PAD.b + 8); ctx.lineTo(W - PAD.r, H - PAD.b + 8); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD.l - 8, PAD.t); ctx.lineTo(PAD.l - 8, H - PAD.b + 8); ctx.stroke();
  ctx.textAlign = "left"; ctx.fillText(`← ${axisName(ax.low, -1)}`, PAD.l, H - 10);
  ctx.textAlign = "right"; ctx.fillText(`${axisName(ax.high, 1)} →`, W - PAD.r, H - 10);
  ctx.save(); ctx.translate(12, H - PAD.b); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "left"; ctx.fillText(`← ${axisName(ay.low, -1)}`, 0, 0);
  ctx.textAlign = "right"; ctx.fillText(`${axisName(ay.high, 1)} →`, H - PAD.b - PAD.t, 0);
  ctx.restore();
}

function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!D) return;
  drawAxes();
  const f = finding();
  const focus = pinned >= 0 ? pinned : hover;
  const dim = focus >= 0 ? 0.25 : 1;
  for (const pass of [0, 1]) {            // points without the finding first, so the orange points stay on top
    for (let i = 0; i < rep.length; i++) {
      if (has(i, f) !== pass) continue;
      const [x, y] = toScreen(rep[i]);
      ctx.globalAlpha = dim * (pass ? 0.95 : 0.5);
      ctx.strokeStyle = pass ? COL.pos : COL.neg; ctx.lineWidth = 1.3;
      ctx.beginPath(); tri(x, y, R); ctx.stroke();
    }
    for (let i = 0; i < ct.length; i++) {
      if (has(i, f) !== pass) continue;
      const [x, y] = toScreen(ct[i]);
      ctx.globalAlpha = dim * (pass ? 0.9 : 0.5);
      ctx.fillStyle = pass ? COL.pos : COL.neg;
      ctx.beginPath(); ctx.arc(x, y, R, 0, 7); ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  if (focus >= 0) drawFocus(focus, f);
}

function drawFocus(item, f) {
  const n = D.cases.length, i = item % n;
  const c = toScreen(ct[i]), own = toScreen(rep[i]);
  if (state !== "separate") {           // the scan and its own report: same patient
    ctx.strokeStyle = COL.ink; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(...c); ctx.lineTo(...own); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = COL.ink; ctx.lineWidth = 1.6;
  ctx.fillStyle = has(i, f) ? COL.pos : COL.neg;
  ctx.beginPath(); ctx.arc(c[0], c[1], R + 2.5, 0, 7); ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.beginPath(); tri(own[0], own[1], R + 2.5); ctx.fill(); ctx.stroke();
}

function pick(mx, my) {
  let best = -1, bd = 14 * 14;
  const n = D.cases.length;
  for (let i = 0; i < n; i++) {
    for (const [arr, off] of [[ct, 0], [rep, n]]) {
      const [x, y] = toScreen(arr[i]);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd) { bd = d; best = i + off; }
    }
  }
  return best;
}

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function showTip(item, mx, my, pin = false) {
  // short card: the findings that the report states
  const n = D.cases.length, i = item % n, isRep = item >= n;
  const stated = D.findings.filter((_, f) => has(i, f));
  let html = `<p class="ex-imp"><b>${isRep ? "Report" : "Scan"}.</b> `;
  html += stated.length ? `The report states: ${esc(stated.join(", ").toLowerCase())}.` : "The report states none of the 18 findings.";
  html += "</p>";
  tip.innerHTML = html;
  tip.classList.toggle("pinned", pin);
  tip.hidden = false;
  const stage = canvas.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = mx + 16, y = my - th / 2;
  if (x + tw > stage.width - 4) x = mx - tw - 16;
  if (x < 4) x = Math.max(4, Math.min(stage.width - tw - 4, mx - tw / 2));
  y = Math.max(4, Math.min(y, stage.height - th - 4));
  tip.style.translate = `${x}px ${y}px`;
}

function placeCue() {
  if (touched || cueItem < 0 || state === "separate" || to) { cue.hidden = true; return; }
  const [x, y] = toScreen(ct[cueItem]);
  cue.style.translate = `${x}px ${y}px`;
  cue.classList.toggle("left", x > W - 180);      // keep the label inside the plot
  cue.hidden = false;
}
function dropCue() { touched = true; cue.hidden = true; }

function resize() {
  const r = canvas.getBoundingClientRect();
  dpr = Math.min(devicePixelRatio || 1, 2);
  W = r.width; H = r.height;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  draw();
  if (D) placeCue();
}

const local = (e) => { const b = canvas.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
canvas.addEventListener("pointermove", (e) => {
  if (!D || to || pinned >= 0) return;
  if (!touched && pick(...local(e)) >= 0) dropCue();
  const [mx, my] = local(e), h = pick(mx, my);
  if (h !== hover) { hover = h; draw(); if (h >= 0) showTip(h, mx, my); else tip.hidden = true; }
  canvas.style.cursor = h >= 0 ? "pointer" : "default";
});
canvas.addEventListener("pointerleave", () => { if (pinned < 0) { hover = -1; tip.hidden = true; draw(); } });
canvas.addEventListener("click", (e) => {
  if (!D || to) return;
  dropCue();
  const [mx, my] = local(e), h = pick(mx, my);
  pinned = h; hover = -1; draw();
  if (h >= 0) showTip(h, mx, my, true); else tip.hidden = true;
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && pinned >= 0) { pinned = -1; tip.hidden = true; draw(); } });

function stopAuto() { auto.forEach(clearTimeout); auto = []; }
function play() {
  stopAuto();
  if (reduce.matches) { go("200", 0); return; }
  go("separate", 0);
  auto.push(setTimeout(() => go("0"), 2500), setTimeout(() => go("200", 1600), 6500));
}
steps.forEach((b) => b.addEventListener("click", () => { stopAuto(); go(b.dataset.step); }));
replay.addEventListener("click", play);
const nameFinding = () => { fname.textContent = D.findings[finding()].toLowerCase(); };
select.addEventListener("change", () => { nameFinding(); showText(); draw(); });
new ResizeObserver(resize).observe(canvas);

D = await (await fetch("data/explorer.json")).json();   // layouts arrive scaled to [-1, 1]; scans share the report transform
{
  const all = [D.rep, ...Object.values(D.lv).map((l) => l.xy)].flat();
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  BOX = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, hw: (x1 - x0) / 2, hh: (y1 - y0) / 2 };
}
D.findings.forEach((name, f) => select.add(new Option(name, f, false, f === D.defaultFinding)));
nameFinding();
{
  const r = D.lv["200"].rank;
  D.cases.forEach((_, i) => { if (has(i, D.defaultFinding) && (cueItem < 0 || r[i] < r[cueItem])) cueItem = i; });
}
resize();
showText();
new IntersectionObserver((es, io) => { if (es[0].isIntersecting) { io.disconnect(); play(); } }, { threshold: 0.4 }).observe(canvas);
