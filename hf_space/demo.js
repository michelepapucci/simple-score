// In-browser SIMPLE demo: tokenizer (tokenizers.js) + ONNX Runtime Web (WebAssembly).
// Nothing heavy is fetched until the visitor presses "Load the demo"; the model and tokenizer
// are then kept in the browser's Cache Storage for later visits.

const ORT_VERSION = "1.31.1";
const TOKENIZERS_VERSION = "0.2.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const TOKENIZERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@${TOKENIZERS_VERSION}/dist/tokenizers.min.mjs`;
const MANIFEST_URL = "model/manifest.json";
const CACHE_PREFIX = "simple-metric-demo-";
const MAX_TOKENS = 512;

const $ = (id) => document.getElementById(id);
let manifest = null;
let tokenizer = null;
let session = null;
let ort = null;

const EXAMPLES = [
  ["Good simplification",
   "The physician recommended that the patient refrain from strenuous physical activity for several weeks.",
   "The doctor told the patient not to exercise hard for a few weeks.", ""],
  ["Sentence splitting",
   "Due to the inclement weather conditions, the municipal authorities decided to postpone the outdoor festival indefinitely.",
   "The weather was bad. So the city put off the outdoor festival. No new date was set.", ""],
  ["Copy of the source",
   "The physician recommended that the patient refrain from strenuous physical activity for several weeks.",
   "The physician recommended that the patient refrain from strenuous physical activity for several weeks.", ""],
  ["Complexification",
   "The doctor told the patient not to exercise hard for a few weeks.",
   "The physician recommended that the patient refrain from strenuous physical activity for several weeks.", ""],
  ["Compare two candidates",
   "Due to the inclement weather conditions, the municipal authorities decided to postpone the outdoor festival indefinitely.",
   "Because of the bad weather, the city put off the outdoor festival.",
   "The authorities decided."],
];

// ---------------------------------------------------------------- helpers

function formatMB(bytes) {
  return `${Math.round(bytes / 1e6)} MB`;
}

function cacheName() {
  return CACHE_PREFIX + (manifest ? manifest.version : "unknown");
}

async function openCache() {
  try {
    return await caches.open(cacheName());
  } catch {
    return null; // private windows or blocked storage: work without a cache
  }
}

async function isCached() {
  try {
    const cache = await openCache();
    return Boolean(cache && (await cache.match(manifest.model.path)));
  } catch {
    return false;
  }
}

// Fetch a file, reporting bytes received; serve it from / store it in Cache Storage.
async function cachedFetch(path, onBytes) {
  const cache = await openCache();
  if (cache) {
    try {
      const hit = await cache.match(path);
      if (hit) {
        const buf = await hit.arrayBuffer();
        onBytes(buf.byteLength);
        return buf;
      }
    } catch { /* fall through to the network */ }
  }
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not download ${path} (HTTP ${response.status}).`);
  let buf;
  if (response.body && response.body.getReader) {
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onBytes(value.length);
    }
    const all = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) { all.set(c, offset); offset += c.length; }
    buf = all.buffer;
  } else {
    buf = await response.arrayBuffer();
    onBytes(buf.byteLength);
  }
  if (cache) {
    try { await cache.put(path, new Response(buf)); } catch { /* storage full: skip caching */ }
  }
  return buf;
}

// Standard normal CDF (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7).
function phi(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    * Math.exp(-x * x / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}

function verdict(mu) {
  if (mu >= 0.5) return "Successful simplification";
  if (mu >= 0.15) return "Partial simplification";
  if (mu > -0.15) return "Close to a copy of the source";
  return "Unsuccessful: complexification, or degenerate text";
}

function confidence(sigma) {
  if (sigma <= 0.1) return "the model is confident";
  if (sigma <= 0.2) return "less certain than usual";
  return "uncertain: worth a human look";
}

// ---------------------------------------------------------------- loading

async function loadDemo() {
  $("load-error").classList.add("hidden");
  $("intro").classList.add("hidden");
  $("loading").classList.remove("hidden");
  const total = manifest.model.bytes + manifest.tokenizer.bytes + manifest.tokenizer_config.bytes;
  let received = 0;
  const onBytes = (n) => {
    received += n;
    const pct = Math.min(100, (100 * received) / total);
    $("bar").style.width = `${pct.toFixed(1)}%`;
    $("bar").parentElement.setAttribute("aria-valuenow", pct.toFixed(0));
    $("status").textContent = `Loading the model: ${formatMB(received)} of ${formatMB(total)}`;
  };
  try {
    const [tokLib, ortLib, tokJson, tokConfig, modelBuf] = await Promise.all([
      import(TOKENIZERS_URL),
      import(`${ORT_BASE}ort.wasm.min.mjs`),
      cachedFetch(manifest.tokenizer.path, onBytes),
      cachedFetch(manifest.tokenizer_config.path, onBytes),
      cachedFetch(manifest.model.path, onBytes),
    ]);
    $("status").textContent = "Starting the model…";
    const decoder = new TextDecoder();
    tokenizer = new tokLib.Tokenizer(JSON.parse(decoder.decode(tokJson)), JSON.parse(decoder.decode(tokConfig)));
    ort = ortLib;
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    session = await ort.InferenceSession.create(new Uint8Array(modelBuf), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    $("loading").classList.add("hidden");
    $("demo").classList.remove("hidden");
    $("source").focus();
  } catch (err) {
    $("loading").classList.add("hidden");
    $("intro").classList.remove("hidden");
    $("load-error").textContent = `The demo could not start: ${err.message || err}`;
    $("load-error").classList.remove("hidden");
    $("load").textContent = "Try again";
    $("load").addEventListener("click", loadDemo, { once: true });
  }
}

// ---------------------------------------------------------------- scoring

async function scorePair(source, candidate) {
  const ids = tokenizer.encode(source, { text_pair: candidate }).ids;
  if (ids.length > MAX_TOKENS) {
    throw new Error(`The texts are too long for the demo (${ids.length} tokens; the limit is ${MAX_TOKENS}).`);
  }
  const shape = [1, ids.length];
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, (v) => BigInt(v)), shape),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
  };
  const out = await session.run(feeds);
  const mu = Math.max(-1, Math.min(1, Number(out.mu.data[0])));
  return { mu, sigma: Number(out.sigma.data[0]) };
}

function place(which, res) {
  const pos = (v) => `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%`;
  const marker = $(`marker-${which}`);
  const band = $(`band-${which}`);
  if (!res) {
    marker.style.display = "none";
    band.style.display = "none";
    return;
  }
  marker.style.display = "block";
  marker.style.left = pos(res.mu);
  band.style.display = "block";
  band.style.left = pos(res.mu - res.sigma);
  band.style.width = `calc(${pos(res.mu + res.sigma)} - ${pos(res.mu - res.sigma)})`;
}

function card(which, label, res) {
  const el = $(`card-${which}`);
  el.classList.remove("hidden");
  el.innerHTML = "";
  const title = document.createElement("div");
  title.className = "small muted";
  title.textContent = label;
  const big = document.createElement("div");
  big.className = "big";
  big.textContent = `μ = ${res.mu >= 0 ? "+" : "−"}${Math.abs(res.mu).toFixed(3)}`;
  const v = document.createElement("div");
  v.className = "verdict";
  v.textContent = verdict(res.mu);
  const s = document.createElement("div");
  s.className = "small muted";
  s.textContent = `σ = ${res.sigma.toFixed(3)}: ${confidence(res.sigma)}`;
  el.append(title, big, v, s);
}

async function onScore() {
  hideError();
  const source = $("source").value.trim();
  const a = $("cand-a").value.trim();
  const b = $("cand-b").value.trim();
  if (!source || !a) {
    showError("Please enter a source sentence and a simplification.");
    return;
  }
  $("score").disabled = true;
  $("score").textContent = "Scoring…";
  try {
    const resA = await scorePair(source, a);
    const resB = b ? await scorePair(source, b) : null;
    $("result").style.display = "block";
    $("marker-b").querySelector("span").textContent = "B";
    $("marker-a").querySelector("span").textContent = resB ? "A" : "μ";
    place("a", resA);
    place("b", resB);
    card("a", resB ? "Simplification A" : "Simplification", resA);
    if (resB) {
      card("b", "Simplification B", resB);
      const p = phi((resA.mu - resB.mu) / Math.sqrt(resA.sigma ** 2 + resB.sigma ** 2));
      $("pref").classList.remove("hidden");
      const pct = p > 0.999 ? "> 99.9%" : p < 0.001 ? "< 0.1%" : `${(100 * p).toFixed(1)}%`;
      $("pref").textContent = `Probability that A is the better simplification: ${pct} ` +
        "(Thurstone's model on the two scores and their uncertainties).";
    } else {
      $("card-b").classList.add("hidden");
      $("pref").classList.add("hidden");
    }
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    $("score").disabled = false;
    $("score").textContent = "Score";
  }
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").classList.remove("hidden");
}

function hideError() {
  $("error").classList.add("hidden");
}

// ---------------------------------------------------------------- page setup

function setupExamples() {
  for (const [label, source, a, b] of EXAMPLES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => {
      $("source").value = source;
      $("cand-a").value = a;
      $("cand-b").value = b;
      onScore();
    });
    $("examples").append(btn);
  }
}

async function init() {
  setupExamples();
  $("score").addEventListener("click", onScore);
  $("forget").addEventListener("click", async () => {
    try {
      for (const key of await caches.keys()) if (key.startsWith(CACHE_PREFIX)) await caches.delete(key);
      $("forget").textContent = "Removed. The model will be downloaded again next time.";
    } catch {
      $("forget").textContent = "Could not access the browser storage.";
    }
  });
  try {
    manifest = await (await fetch(MANIFEST_URL)).json();
  } catch {
    showError("Could not read the demo's model manifest.");
    $("load").disabled = true;
    return;
  }
  const total = manifest.model.bytes + manifest.tokenizer.bytes + manifest.runtime_bytes;
  $("size").textContent = `about ${formatMB(total)}`;
  if (await isCached()) {
    $("load").textContent = "Start the demo";
    $("cached-note").classList.remove("hidden");
  } else {
    $("load").textContent = `Load the demo (${formatMB(total)})`;
  }
  $("load").addEventListener("click", loadDemo, { once: true });
}

init();
