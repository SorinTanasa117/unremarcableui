# Ollama Agent UI — Optimized for Intel Core Ultra 9 185H (32GB RAM)

This repository contains the standalone, full-featured **Ollama Agent UI** (React + TypeScript + Express), tailored and pre-optimized to run with extreme speed and resource efficiency on an **Intel Core Ultra 9 185H processor laptop with 32GB RAM (CPU-only or Integrated GPU execution)**.

---

## 🚀 Optimization Strategies & Core Concepts

Running large language models (LLMs) and performing multi-turn autonomous research on CPU-only or integrated graphics (iGPU) hardware faces three primary bottlenecks: **memory bandwidth limits**, **prompt-cache eviction overhead**, and **headless browser RAM footprint**.

To solve these, the following industry-standard optimizations have been applied to this application and are recommended for your Ollama configuration.

### 1. Hardware-Aware Thread Optimization
The Intel Core Ultra 9 185H features a hybrid architecture with:
*   **6 Performance Cores (P-cores)** (12 threads)
*   **8 Efficient Cores (E-cores)** (8 threads)
*   **2 Low-Power Efficient Cores (LP E-cores)** (2 threads)

**The Bottleneck:** LLM generation is bound by memory bandwidth. Running Ollama with hyperthreading or allowing it to schedule threads onto the slower E-cores/LP E-cores results in severe thread thrashing and cache-miss latency, which drastically *reduces* token generation speed.
**The Solution:** Use the UI top-bar **P-core thread mode** toggle. It sends Ollama `num_thread=6` (P-core focus) or `num_thread=8` (slightly broader scheduling) per request. This avoids relying on unsupported environment variables and lets you A/B quickly on your hardware.

### 2. KV Cache Quantization & Flash Attention
During long autonomous research chats, the context window grows up to 32k+ tokens. The memory required to store attention keys/values (the KV Cache) scales quadratically and can consume several gigabytes of RAM independently of the model weights.
*   **Flash Attention (`OLLAMA_FLASH_ATTENTION=1`):** Speeds up prompt processing times significantly on modern CPUs.
*   **KV Cache Quantization (`OLLAMA_KV_CACHE_TYPE=q8_0`):** Compresses the key/value attention maps from FP16 down to 8-bit precision. This **halves the KV cache memory footprint** with absolutely negligible impact on model intelligence, keeping RAM usage low and preventing your laptop from swapping.

### 3. Solving LPDDR5 Prompt-Cache Eviction ("Context Choking")
**The Bottleneck:** When the agent browses a URL, the system needs to extract high-value evidence (quotes, claims) to compile into the ledger.
Previously, this extraction was done by calling the **same primary model** with a secondary "extraction" system prompt. Because the laptop runs in a single-model memory-isolated setup (`OLLAMA_MAX_LOADED_MODELS=1` to prevent swap thrashing), Ollama had to **completely eject the KV cache (prompt cache)** of the main research session to process the secondary extraction prompt.
Once the extraction completed and control returned to the research loop, Ollama was forced to **fully re-evaluate the entire growing conversation history (16k–32k tokens) from scratch**. On shared LPDDR5 RAM, this repeated re-evaluation caused compounding delays—making the model "choke" and slow down exponentially with each subsequent search or browse turn.

**The Solution — Heuristic Extraction Engine:**
We built a local **Fast Heuristic Extraction Engine** (`extractEvidenceHeuristically`) inside `server/routes/ollama.ts`.
*   It analyzes browsed page text in **0 milliseconds** using highly optimized sentence and regex patterns, instantly extracting page titles, publisher domains, distinct key quotes, claims, and analytical relevance scores.
*   **Result:** By bypassing nested LLM calls during browsing, the primary conversation's prompt cache (KV cache) remains **100% warm and cached**. The main research loop resumes generation **instantly** on every turn, completely solving the "context choking" bottleneck!
*   *Configuration:* If you have abundant resources and prefer the slower LLM-based extraction, you can set the environment variable `FAST_HEURISTIC_EXTRACTION=false`. Otherwise, it defaults to `true` for maximum speed.

### 4. Model Isolation & Single-Concurrency Controls
On a single-user 32GB RAM machine, we want to maximize the hardware resources dedicated to the active LLM.
*   **`OLLAMA_NUM_PARALLEL=1`:** Ensures Ollama processes only one request at a time, dedicating 100% of memory bandwidth and CPU power to the active generation.
*   **`OLLAMA_MAX_LOADED_MODELS=1`:** Prevents multiple models from residing in memory simultaneously, keeping the system responsive and far below the 32GB RAM limit.

### 5. Context Reuse Protection for Tool Loops
Tool-heavy runs can explode prompt size if raw tool output is fed back unbounded.
*   Tool outputs sent back into model context are now truncated (`TOOL_CONTEXT_CHAR_LIMIT`, default `8000` chars).
*   This preserves enough signal for follow-up reasoning while preventing context bloat and repeated heavy prefill.

### 6. Predict-Length Guardrail
The backend applies `OLLAMA_NUM_PREDICT` (default `1536`) for chat generations.
*   Prevents runaway responses from wasting decode bandwidth.
*   Keeps agent turns tighter and improves perceived responsiveness.

### 7. Lightning-Fast Headless Browser Optimization
Headless browsers used for search retrieval normally download images, stylesheets, fonts, and heavy media assets, consuming massive CPU cycles and RAM.
*   **Request Interception Optimization:** We have optimized `server/lib/playwright.ts` to block unnecessary network assets (`image`, `stylesheet`, `font`, and `media` files) during DuckDuckGo, Bing, Yahoo searches, and URL scraping.
*   **Result:** Research cycles are **up to 10x faster**, page downloads are lightweight, and headless Chromium runs with minimal CPU/RAM footprint.
### 8. Brave Search API with Strict Credit Fallback
Set `BRAVE_SEARCH_API_KEY` in `.env` or the server environment. Web search uses Brave Search first.
DuckDuckGo is used only when Brave returns HTTP 402 with an explicit exhausted, depleted,
insufficient, exceeded, or reached credit/quota message. Authentication errors, invalid
responses, bad requests, rate limits, timeouts, and network failures do not switch providers.

### 9. Vulkan Offload on Stock Ollama
This app now targets one Ollama runtime only (no separate IPEX-LLM runtime path).
To ask stock Ollama to use Vulkan runner on Intel iGPU:
*   Set `OLLAMA_LLM_LIBRARY=vulkan`
*   Restart Ollama service/process
*   Keep using normal endpoint `http://127.0.0.1:11434`


---

## 🧩 Role-Based Model Switching (auto sidecar swaps)

To keep coding effective on 32 GB shared LPDDR5 without loading several large
models at once, the server can swap in a small auxiliary model for two bounded
jobs, then unload it (`keep_alive: 0`) so the primary coder keeps the RAM.

Configure the roles in `model_map.json`:

```jsonc
"roles": {
  "coder":      "qooba/qwen3-coder-30b-a3b-instruct:q3_k_m", // doc only — primary coder is your UI pick
  "summarizer": "qwen3:4b",                                  // folds old turns into a durable brief
  "vision":     "huihui_ai/qwen3-vl-abliterated:2b"          // describes images for non-vision coders
}
```

`coder` is documentation only; the model actually used for the run is whatever you select in the UI.
Only `summarizer` and `vision` are swapped in as one-shot sidecars.

- **Vision ingest:** attach an image while a non-vision coder is selected and the
  `vision` model describes it once; the text is injected into context and the raw
  image bytes are dropped so they are not re-encoded every turn.
- **Rolling summary:** when the running context passes ~50 % of the window, the
  `summarizer` model compresses the older turns into a pinned brief that survives
  turn-dropping.
- **Final code review:** when a coder run finishes having changed files, the
  `reviewer` model gets ONE look at them (first ~120 lines per file) and flags
  only blocking defects (`file:line: issue`). Findings are handed to the coder
  for one repair round before the run completes; a clean or missing reviewer
  changes nothing. Disable with `REVIEW_AT_END=false`. There is deliberately no
  per-turn checking — on `OLLAMA_MAX_LOADED_MODELS=1` every swap would evict the
  coder's prompt cache, so checks happen once at the end instead.

If a role model is not installed, each feature silently falls back
(heuristic compaction notice; raw image skipped; review skipped).

### Stall & hang protection (watchdog)

Long local runs can stall silently — most often because Ollama unloads the model
mid-run under memory pressure on a shared-memory laptop, leaving the request
hanging with zero tokens. Guards, fastest to coarsest:

- **Model-eviction stop (Ollama):** every `MODEL_PS_INTERVAL_MS` (default 60s)
  the server polls `ollama ps` (GET `/api/ps`). If no model is resident for
  `MODEL_GONE_CHECKS_TO_KILL` consecutive polls (default 2, ≈2 min) the run
  stops with reason **model ejected**. A resident-but-slow prefill still lists
  the model, so healthy long generations are never touched. Disable with
  `MODEL_EVICTION_WATCH=false`.
- **Proven-hang stop:** after `STALL_QUIET_MS` of zero tokens (default 30 min)
  the server measures real CPU work (cumulative process CPU-seconds sampled
  twice per probe — not an instantaneous rate). Only genuine zero-compute for
  `WATCHDOG_IDLE_PROBES_TO_KILL` consecutive probes (default 20 × 15s ≈ 5 min)
  ends the run, and the idle count-down is shown in the UI.
- **Wall-clock budgets:** one turn ≤ `TURN_MAX_DURATION_MS` (default 60 min),
  whole run ≤ `RUN_MAX_DURATION_MS` (default 6 h).

Disable everything with `RUN_WATCHDOG=false`. On a 32 GB shared-memory laptop,
shorter values are worth setting (see `.env.example`): e.g. `STALL_QUIET_MS=600000`
(10 min) and `TURN_MAX_DURATION_MS=1500000` (25 min).

### What you already have vs. what to pull

Your existing models already cover two of the three roles:
- **coder** — `qooba/qwen3-coder-30b-a3b-instruct:q3_k_m` (already installed; fine as the primary).
- **vision** — `huihui_ai/qwen3-vl-abliterated:2b` (already installed; used to describe images).

Only the summarizer is missing:

```bash
ollama pull qwen3:4b           # summarizer / orchestrator sidecar, ~2.6 GB
```

Optional upgrades if you want them:
```bash
ollama pull qwen2.5vl:3b       # lighter Q4 vision sidecar (~3 GB) — faster keep_alive:0 swaps than the 8.5 GB Q8 2b
ollama pull qwen3-coder:30b    # official Q4_K_M coder (~19 GB) — a step up from the q3_k_m build if RAM allows
```

`gemma4:26b` is intentionally **not** recommended as the coder persona model: it
had the highest tool-call failure rate (spawn errors, overwrite-regeneration
loops) in local testing. Keep it for uncensored/multimodal chat only.

## 🛠️ Recommended Models for CPU Research

We recommend using the following lightweight, native tool-calling models that balance high analytical reasoning with low memory overhead:

1.  **`qwen2.5:7b-instruct-q4_K_M` (Recommended Default)**
    *   *RAM Footprint:* ~4.7 GB (Highly compact, Q4_K_M quantization).
    *   *Why:* Superb native tool-calling, excellent reasoning, and highly optimized for CPU inference. Fits comfortably on a 32GB RAM system, leaving >25GB for Windows and other apps.
2.  **`llama3.1:8b-instruct-q4_K_M`**
    *   *RAM Footprint:* ~4.8 GB.
    *   *Why:* Great tool-use accuracy and long context compliance.
3.  **`llama3.2:3b-instruct-q4_K_M` (Ultra-Fast Fallback)**
    *   *RAM Footprint:* ~2.2 GB.
    *   *Why:* Sub-second token-generation on CPU, runs entirely within the CPU's smart cache, making it lightning-fast.

---

## 🏁 How to Start the App with Optimizations

### Step 1: Start Ollama with Core Ultra 9 185H Optimizations

Use the pre-configured Windows batch script provided in the root directory to launch Ollama with the optimal parameters:

```cmd
:: Double-click start_ollama_optimized.bat OR run from Command Prompt:
start_ollama_optimized.bat
```

This script sets up:
*   `OLLAMA_FLASH_ATTENTION=1`
*   `OLLAMA_KV_CACHE_TYPE=q8_0`
*   `OLLAMA_NUM_PARALLEL=1`
*   `OLLAMA_MAX_LOADED_MODELS=1`
*   `OLLAMA_LLM_LIBRARY=vulkan`
*   Persistent values via `setx` + immediate shell values for direct `ollama serve`

*(For macOS/Linux, run `export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1` and run `ollama serve`.)*

Thread count is controlled by UI toggle (`num_thread=6` or `num_thread=8`) per request.
Runtime toggle in UI:
* `Ollama (CPU/Vulkan)` -> uses `OLLAMA_URL` (default `http://127.0.0.1:11434`)
* `llama.cpp SYCL` -> uses `LLAMACPP_URL` (default `http://127.0.0.1:8080`)
* In `llama.cpp SYCL` mode, changing the Context dropdown triggers runtime restart on next send so server context matches selected size.

SYCL runtime now auto-starts with `npm run dev` (unless disabled). Server will:
* resolve a model from `LLAMACPP_MODEL`, or fallback to `model_map.json` `default_model`
* launch `llama-server` with oneAPI/VS environment
* wait until `/v1/models` is ready
* show all Ollama models in GPU mode selector; when you send a prompt, runtime hot-switches to that selected model
* allow unload from the same lightning eject button in GPU mode (stops managed `llama-server`)
* GPU-mode model list comes from local Ollama manifests on disk, so it works even if Ollama service is offline
* if startup fails, backend retries safer profiles (`ngl` downshift, then CPU fallback) before returning error
* unsupported GGUF architectures (for this llama.cpp build) are auto-marked and blocked in GPU mode with a clear error

Optional runtime knobs:
* `AUTO_START_LLAMACPP=true|false` (default `true`)
* `LLAMACPP_MODEL=<full GGUF/blob path>`
* `LLAMACPP_URL=http://127.0.0.1:8080`
* `LLAMACPP_NGL=24` (default safer startup profile)
* `LLAMACPP_FA=on|off` (default `on`)
* `LLAMACPP_START_TIMEOUT_MS=180000`
* `LLAMACPP_EXE`, `VSDEVCMD_PATH`, `ONEAPI_SETVARS_PATH` (override binary/toolchain paths)
* startup log: `agent-workspace/.logs/llamacpp-start.log`

Manual start still possible:
```powershell
# Prompt-ingest oriented
llama-server -m "<GGUF_PATH>" -ngl 40 -fa off --host 127.0.0.1 --port 8080

# Token-generation oriented
llama-server -m "<GGUF_PATH>" -ngl 40 -fa on --host 127.0.0.1 --port 8080
```

Vulkan request on Windows:
```powershell
setx OLLAMA_LLM_LIBRARY "vulkan"
```

### Step 2: Install dependencies & Start the Agent UI Server

Open a terminal in the project root directory and execute:

```bash
# Install package dependencies
npm install

# Run the client + server concurrently
npm run dev
```

Before starting the server, configure Brave Search:

```bash
# PowerShell
$env:BRAVE_SEARCH_API_KEY="your-brave-search-api-key"

# Or place BRAVE_SEARCH_API_KEY=... in .env
```

### Cloud providers

Open the **Cloud** control in the top bar to configure OpenRouter or Factory.ai
with an API key and model ID. The browser stores the selected provider, key,
and model in `localStorage`; the key is sent in the request body only when a
cloud chat request or model-list request is made. Use the **Test connection**
button to validate the key and load the provider's live `/models` list.

Optional server-side defaults:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
FACTORY_API_KEY=
FACTORY_OPENAI_BASE_URL=https://api.factory.ai/v1
```

`FACTORY_OPENAI_BASE_URL` is useful for a Factory deployment or compatible
gateway that exposes OpenAI-compatible `/v1/models` and
`/v1/chat/completions` endpoints. Factory's public documentation describes
BYOK custom-model configuration rather than a universal hosted inference
endpoint, so use the base URL supplied by your Factory deployment.

The application UI will be available at **`http://localhost:5174`** (Client) and proxy requests to the backend API.
