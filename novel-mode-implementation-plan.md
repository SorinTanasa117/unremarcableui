# Novel Mode — Implementation Plan

**Target:** Ollama Agent UI (React + TS + Express)
**Goal:** Add a "Novel Mode" that takes a story idea, proposes a chapter outline, and drafts chapters/segments one at a time using compacted prior-chapter summaries as context instead of the full manuscript — avoiding the KV-cache eviction problem already documented in the README for research mode.

---

## 1. Why this needs its own context strategy

The existing research mode solved "context choking" by never feeding the full growing transcript back into the model — it summarizes with a fast heuristic engine and keeps the primary KV cache warm.

Novel Mode has the same underlying problem but a different shape:
- A novel will blow past any context window in raw form long before the story is done.
- Unlike research (many browse calls per session), chapter summarization happens once per chapter — infrequent enough that a KV-cache eviction from a secondary summarization call is an acceptable cost, unlike per-turn browsing.
- Prose continuity (voice, tense, character voice) needs *some* raw text carried forward, not just summary — pure summary-to-summary chaining causes voice drift over a novel-length work.

**Design principle:** each chapter draft gets (a) a structured summary of everything before it, and (b) the raw tail of the immediately preceding chapter, not the whole book.

---

## 2. Storage layout

```
.novels/
  <novel-id>/
    manifest.json              # title, premise, style, model config, chapter list, status
    bible.json                 # characters, locations, established facts, tone/style notes
    chapters/
      ch01.txt                 # raw drafted prose
      ch02.txt
    summaries/
      ch01.json                # structured compacted summary
      ch02.json
    outline.json               # proposed + confirmed chapter list with beats
```

### `manifest.json`
```json
{
  "id": "uuid",
  "title": "string",
  "premise": "string",
  "created_at": "iso8601",
  "updated_at": "iso8601",
  "status": "outlining | drafting | complete",
  "model_id": "mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso:latest",
  "persona": "novelist",
  "target_chapter_count": 24,
  "current_chapter": 7,
  "num_predict_novel": 5000
}
```

### `outline.json`
Array of chapter objects: `{ number, title, beat_summary, pov_character, status: "planned|drafted|revised" }`. Regenerable — user can ask "propose next 3 chapters" and the model appends/edits this file rather than the manifest.

### `summaries/chXX.json` (the compacted context unit)
```json
{
  "chapter": 6,
  "summary": "2-4 paragraph plot/character-state summary",
  "characters_active": ["name: state/location/emotional beat"],
  "open_threads": ["unresolved plot elements introduced or advanced"],
  "tone_notes": "voice/pacing notes if they shifted",
  "last_600_words_raw": "verbatim tail of the chapter, for prose continuity"
}
```

---

## 3. Context assembly per chapter draft

For drafting chapter N, build the prompt from:

1. **System prompt** — novel-specific persona (see §6), not the general "Creative" persona in `personas.md`, which is tuned for a different genre entirely.
2. **`bible.json`** — characters/world facts, kept small and hand-curated/model-updated.
3. **Outline entry for chapter N** — the beat this chapter needs to hit.
4. **Rolling summary** — concatenated `summary` + `open_threads` fields from the last 2–3 chapters only (not all prior chapters — re-summarize periodically, see §5).
5. **Raw tail** — `last_600_words_raw` from chapter N-1, verbatim, so the model picks up mid-voice rather than re-establishing tone from a description of it.
6. **Drafting instruction** — target length, POV, any user notes for this chapter.

This keeps the per-request context small and roughly constant in size regardless of how long the novel gets — the same fix as the research-mode heuristic extractor, applied to a different bottleneck.

---

## 4. Handling output-length limits (chapter segmentation)

`OLLAMA_NUM_PREDICT` caps single-turn output (currently 1536 in this repo). A chapter will usually exceed that.

- Add a **novel-mode-specific predict ceiling**, `num_predict_novel = 5000`, set per-request rather than changing the global default, since research/chat turns shouldn't get slower by default.
- 5000 tokens is deliberately below what a full chapter may need, and below the point where local 8–24B models tend to lose coherence in a single unbroken generation. Long chapters are handled by a **continuation loop** rather than by raising this ceiling further:
  1. Generate up to `num_predict_novel` (5000 tokens).
  2. Heuristically check for a natural stopping point (paragraph/scene break near the end, or explicit end-of-chapter marker the model is instructed to emit).
  3. If cut off mid-scene, checkpoint the segment to disk, then auto-issue a continuation turn: system prompt + last ~500 words of *this* draft (raw, not summarized) + "continue exactly where this left off."
  4. Concatenate segments into the chapter file; repeat until the model emits the end marker or a max-segment count is hit (safety valve, e.g. 4 segments ≈ 20,000 tokens).
- Ask the model to emit a literal marker like `[END_OF_CHAPTER]` on completion — cheap and reliable signal, avoids guessing from prose.
- Checkpointing each 5000-token segment to disk as it completes matters more than the exact ceiling: on CPU/iGPU generation, a single segment at 5000 tokens is a few minutes, not the 20–60+ minutes a 16k single-shot call would take — so a failed or wandering segment costs little, and the UI can show live per-segment progress instead of one long silent wait.

---

## 5. Summarization step (after each chapter)

Two options, pick based on how much you value speed vs. fidelity — could also be configurable like `FAST_HEURISTIC_EXTRACTION`:

- **Model-based (default):** one extra call to the same model with a dedicated summarization system prompt, output constrained to the `summaries/chXX.json` schema. Accept the KV-cache eviction cost here — it happens once per chapter, not once per turn.
- **Heuristic fallback:** if you want a zero-cost option akin to `extractEvidenceHeuristically`, a pure summarization heuristic is much lower-fidelity for narrative prose than it is for extracting quotes/claims from articles — recommend against making this the default for novel mode, but fine as an optional low-resource mode.

Additionally: every ~5 chapters, run a **re-summarization pass** that compacts the last 5 chapter summaries into one "arc summary," so the rolling context window described in §3 doesn't slowly grow as the book gets long. Store as `summaries/arc-01.json` etc., and have context assembly prefer the latest arc summary + the last 1–2 individual chapter summaries over an ever-growing chain.

---

## 6. New persona

Add a dedicated `Novelist` persona to `personas.md` rather than repurposing the existing `Creative` persona — that one is written for a specific genre and isn't a general-purpose prose voice. The novel persona should cover: consistent long-form narrative voice, adherence to the outline beat and character bible, natural chapter-ending behavior, and the `[END_OF_CHAPTER]` marker convention from §4.

---

## 7. Backend work (server side)

New route module, e.g. `server/routes/novels.ts`:
- `POST /api/novels` — create novel from premise, kick off outline proposal
- `GET /api/novels/:id` — manifest + outline + chapter list
- `POST /api/novels/:id/outline` — propose/regenerate N chapters
- `POST /api/novels/:id/chapters/:n/draft` — run the draft+continuation loop for one chapter, streaming to the client like existing chat streaming
- `POST /api/novels/:id/chapters/:n/summarize` — trigger summary generation (usually auto-called after a successful draft)
- File I/O for `.novels/` mirrors the existing `research-ledger.json` pattern already in the researcher persona's tool design — reuse whatever file-write utilities already exist for that rather than writing new ones.

## 8. Frontend work (client side)

- Mode toggle alongside existing Coder/Researcher/Creative/System persona switcher.
- Sidebar: chapter list with status (planned/drafted/revised), click to view/edit.
- "Propose next chapters" action → shows outline diff for user approval before committing to `outline.json`.
- Chapter view: streamed draft text, visible segment boundaries during continuation loop (so the user sees it's still the same chapter, not silently starting over), manual "regenerate this chapter" and "continue this chapter" actions.
- Model selector: default to a **non-thinking** model for novel mode (e.g. the Llama 3.1 8B Virtuoso entry in `model_map.json`) — thinking-mode models burn part of the 5000-token `num_predict_novel` budget on reasoning tokens before prose starts, reducing effective prose-per-segment.
- Route novel mode to a model whose `max_context` comfortably covers prompt (persona + bible + rolling summary + raw tail, typically 2–4k tokens) plus the 5000-token generation — this is well within reach for most entries in `model_map.json` (several report `max_context: 262144`), but should still be an explicit check rather than assumed, since `phi4` (16384) and similar smaller-context models are a poor fit for novel mode regardless of prose quality.

---

## 9. Suggested build order

1. Storage layer + manifest/outline schema, no generation yet — just CRUD and file structure.
2. Outline proposal endpoint (single model call, structured output).
3. Single-chapter draft endpoint, no continuation loop, no summarization — prove the context-assembly shape works end to end.
4. Add continuation loop for long chapters.
5. Add summarization step + rolling-context assembly.
6. Add arc-summary compaction pass for long novels.
7. Frontend UI last, once the API shape is stable.

---

## 10. Open questions worth resolving before Droid Factory starts

- Max chapter length target — with `num_predict_novel` fixed at 5000 tokens per segment, this now just drives *how many* continuation segments a chapter is allowed before the safety valve kicks in.
- Whether outline changes mid-novel should retroactively flag already-drafted chapters as needing revision.
- Whether the arc-summary compaction (§5) should be user-visible/editable, since it's effectively becoming the model's long-term memory of the book.
