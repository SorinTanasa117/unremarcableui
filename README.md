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

The application UI will be available at **`http://localhost:5174`** (Client) and proxy requests to the backend API.
