/**
 * Novel Mode API routes.
 * Handles novel CRUD, outline proposals, chapter drafting with continuation loop,
 * and summarization.
 */

import { Router, Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import * as storage from '../lib/novel/storage.js';
import type {
  NovelManifest,
  ChapterOutline,
  ChapterSummary,
  ArcSummary,
  NovelBible,
} from '../lib/novel/types.js';
import { inferenceDispatcher } from '../lib/inferenceDispatcher.js';

const router = Router();

function routeParam(value: string | string[] | undefined): string {
  return Array.isArray(value) ? value[0] ?? '' : value ?? '';
}

const OLLAMA_URL = process.env.OLLAMA_URL ?? 'http://127.0.0.1:11434';
const OLLAMA_KEEP_ALIVE = process.env.OLLAMA_KEEP_ALIVE ?? '15m';
const NUM_PREDICT_NOVEL = Number.parseInt(process.env.NUM_PREDICT_NOVEL ?? '5000', 10);
const OUTLINE_NUM_PREDICT = Number.parseInt(process.env.OUTLINE_NUM_PREDICT ?? '12000', 10);
const NOVEL_NUM_CTX = Number.parseInt(process.env.NOVEL_NUM_CTX ?? '16384', 10);
const MAX_CONTINUATION_SEGMENTS = 4;
const END_OF_CHAPTER_MARKER = '[END_OF_CHAPTER]';

// Default model for novel mode (non-thinking, good prose quality)
const DEFAULT_NOVEL_MODEL = 'mirage335/Llama-3_1-8B-Instruct-abliterated-virtuoso:latest';

// Abort controllers for streaming operations
const novelAbortControllers = new Map<string, AbortController>();

// ─────────────────────────────────────────────────────────────────────────────
// Persona prompt for novel mode
// ─────────────────────────────────────────────────────────────────────────────

const NOVELIST_SYSTEM_PROMPT = `You are a dedicated long-form fiction writer specializing in novel-length narratives. Your purpose is to draft compelling chapters that maintain voice consistency, character continuity, and plot coherence across an extended work.

## Core Principles
- **Voice Consistency**: Maintain the established narrative voice, tense, and style throughout. When given the raw tail of a previous chapter, seamlessly continue from that prose without reintroducing yourself or resetting tone.
- **Character Bible Adherence**: Follow the character bible exactly. Do not contradict established traits, relationships, or backstories.
- **Outline Fidelity**: Each chapter must hit the beats specified in its outline entry.
- **Natural Chapter Flow**: Write chapters with proper pacing. Avoid abrupt endings mid-scene.

## Output Conventions
- When you complete a chapter, emit the literal marker [END_OF_CHAPTER] on its own line.
- If continuing a cut-off chapter, pick up exactly where the prose ended—no recap.
- Keep chapters between 2,000–5,000 words unless specified otherwise.

Write as if you know the story intimately. Show, don't tell. Dialogue should reveal character. Sensory details ground scenes. Use section breaks (* * *) for scene transitions.`;

const OUTLINE_SYSTEM_PROMPT = `You are a skilled story architect. Given a premise and length targets, you propose chapter outlines that build compelling narrative arcs.

Output ONLY a valid JSON array of chapter objects with this exact schema (no markdown, no explanation):
[
  {
    "number": 1,
    "title": "Chapter Title",
    "description": "Exactly 2-3 sentences describing what takes place in this chapter, including its main conflict, emotional movement, and plot advancement.",
    "pov_character": "Name of POV character for this chapter",
    "characters_involved": ["Character Name"],
    "target_pages": 10,
    "target_words": 2500
  }
]

Create chapters that:
- Build tension progressively with proper pacing for the target length
- Distribute word counts appropriately (some chapters longer for major events, shorter for transitions)
- Allocate the requested total pages across chapters; total target_pages must equal the requested page count
- Include every character who materially appears or affects events in characters_involved
- Alternate POV if multiple characters are central
- Include turning points, complications, and revelations at appropriate intervals
- End strong—cliffhangers or emotional punches
- Serve the overall story arc while allowing character development

The total of all target_pages must equal the requested page count. The total of all target_words should approximately equal the requested total. Every description must contain 2-3 complete sentences.`;

const FINAL_OUTLINE_SYSTEM_PROMPT = `Return ONLY one valid JSON array. No markdown, no explanation, no reasoning, no preamble.
Create exactly the requested number of chapter objects. Each object must contain:
number, title, description (exactly 2-3 sentences), pov_character, characters_involved (array), target_pages, target_words, status.
Use concise descriptions. Make target_pages sum exactly to requested total pages.`;

const SUMMARIZATION_SYSTEM_PROMPT = `You are a precise narrative analyst. Given a chapter of fiction, extract a structured summary for use as context in drafting future chapters.

Output JSON with this exact schema:
{
  "chapter": <number>,
  "summary": "2-4 paragraph plot and character-state summary",
  "characters_active": ["character_name: state/location/emotional beat"],
  "open_threads": ["unresolved plot elements introduced or advanced in this chapter"],
  "tone_notes": "Brief notes on voice/pacing if they shifted significantly"
}

Focus on:
- What happened (plot events)
- Where characters ended up (physical and emotional state)
- What's unresolved or foreshadowed
- Any notable stylistic elements that should carry forward`;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Call Ollama
// ─────────────────────────────────────────────────────────────────────────────

async function callOllama(
  model: string,
  messages: Array<{ role: string; content: string }>,
  options: { stream?: boolean; format?: 'json'; numPredict?: number; signal?: AbortSignal; numCtx?: number } = {}
): Promise<{ content: string; error?: string }> {
  const body = {
    model,
    messages,
    stream: options.stream ?? false,
    options: {
      num_ctx: options.numCtx ?? NOVEL_NUM_CTX,
      num_predict: options.numPredict ?? NUM_PREDICT_NOVEL,
    },
    keep_alive: OLLAMA_KEEP_ALIVE,
    ...(options.format ? { format: options.format } : {}),
  };

  try {
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: options.signal,
      dispatcher: inferenceDispatcher,
    });

    if (!response.ok) {
      const detail = await response.text();
      return { content: '', error: `Ollama error ${response.status}: ${detail}` };
    }

    const data = await response.json() as { message?: { content?: string } };
    return { content: data.message?.content ?? '' };
  } catch (err: any) {
    if (err.name === 'AbortError') {
      return { content: '', error: 'Aborted' };
    }
    return { content: '', error: err.message };
  }
}

async function* streamOllama(
  model: string,
  messages: Array<{ role: string; content: string }>,
  signal?: AbortSignal,
  think = false,
  numPredict = NUM_PREDICT_NOVEL,
  format?: 'json',
  numCtx?: number,
): AsyncGenerator<{ content?: string; thinking?: string; done?: boolean; error?: string }> {
  const body = {
    model,
    messages,
    stream: true,
    think,
    options: {
      num_ctx: numCtx ?? NOVEL_NUM_CTX,
      num_predict: numPredict,
    },
    keep_alive: OLLAMA_KEEP_ALIVE,
    ...(format ? { format } : {}),
  };

  try {
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
      dispatcher: inferenceDispatcher,
    });

    if (!response.ok) {
      const detail = await response.text();
      yield { error: `Ollama error ${response.status}: ${detail}` };
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      yield { error: 'No response body' };
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const parsed = JSON.parse(trimmed);
          if (parsed.message?.thinking) {
            yield { thinking: parsed.message.thinking };
          }
          if (parsed.message?.content) {
            yield { content: parsed.message.content };
          }
          if (parsed.done) {
            yield { done: true };
          }
        } catch {}
      }
    }

    // Some Ollama-compatible servers close stream without trailing newline.
    // Parse final buffered JSON instead of silently dropping it.
    buffer += decoder.decode();
    const finalLine = buffer.trim();
    if (finalLine) {
      try {
        const parsed = JSON.parse(finalLine);
        if (parsed.message?.thinking) yield { thinking: parsed.message.thinking };
        if (parsed.message?.content) yield { content: parsed.message.content };
        if (parsed.done) yield { done: true };
      } catch {}
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      yield { error: 'Aborted' };
    } else {
      yield { error: err.message };
    }
  }
}

function parseProposedOutline(
  content: string,
  startFrom: number,
  preserveNumbers = false,
): ChapterOutline[] | null {
  try {
    let jsonContent = content.trim();
    const jsonMatch = jsonContent.match(/\[[\s\S]*\]/);
    if (jsonMatch) jsonContent = jsonMatch[0];

    const parsed = JSON.parse(jsonContent) as Array<Partial<ChapterOutline>>;
    if (!Array.isArray(parsed) || parsed.length === 0) return null;

    return parsed.map((chapter, index) => ({
      number: preserveNumbers && Number.isFinite(Number(chapter.number))
        ? Number(chapter.number)
        : startFrom + index,
      title: chapter.title || `Chapter ${startFrom + index}`,
      description: chapter.description || chapter.beat_summary || chapter.summary || '',
      beat_summary: chapter.description || chapter.beat_summary || chapter.summary || '',
      pov_character: chapter.pov_character || chapter.pov || 'Unknown',
      characters_involved: Array.isArray(chapter.characters_involved ?? chapter.characters)
        ? (chapter.characters_involved ?? chapter.characters ?? []).map(String).filter(Boolean)
        : [],
      target_pages: Number.isFinite(Number(chapter.target_pages))
        ? Number(chapter.target_pages)
        : undefined,
      target_words: Number.isFinite(Number(chapter.target_words))
        ? Number(chapter.target_words)
        : undefined,
      status: 'planned' as const,
    }));
  } catch {
    return null;
  }
}

function rebalancePageTargets(outline: ChapterOutline[], totalPages: number): ChapterOutline[] {
  if (!outline.length || !Number.isFinite(totalPages) || totalPages < outline.length) return outline;

  const requested = outline.map((chapter) => (
    Number.isFinite(Number(chapter.target_pages)) ? Math.max(1, Math.round(Number(chapter.target_pages))) : 1
  ));
  const requestedTotal = requested.reduce((sum, pages) => sum + pages, 0);
  const scale = requestedTotal > 0 ? totalPages / requestedTotal : 1;
  const pages = requested.map((value) => Math.max(1, Math.round(value * scale)));
  let difference = totalPages - pages.reduce((sum, value) => sum + value, 0);
  while (difference > 0) {
    pages[difference % pages.length] += 1;
    difference -= 1;
  }
  while (difference < 0) {
    const index = pages.findIndex((value) => value > 1);
    if (index < 0) break;
    pages[index] -= 1;
    difference += 1;
  }

  return outline.map((chapter, index) => ({ ...chapter, target_pages: pages[index] }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Routes
// ─────────────────────────────────────────────────────────────────────────────

// GET /api/novels — List all novels
router.get('/', async (_req: Request, res: Response) => {
  try {
    const novels = await storage.listNovels();
    res.json({ novels });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/novels/characters/library — List all characters across all past novels
router.get('/characters/library', async (_req: Request, res: Response) => {
  try {
    const library = await storage.getCharacterLibrary();
    res.json({ characters: library });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/novels — Create new novel from premise
// ID can be provided (session-based) or auto-generated
router.post('/', async (req: Request, res: Response) => {
  const { id, title, premise, model_id, target_chapters, target_pages, words_per_page, bible } = req.body as {
    id?: string;
    title: string;
    premise: string;
    model_id?: string;
    target_chapters?: number;
    target_pages?: number;
    words_per_page?: number;
    bible?: Partial<NovelBible>;
  };

  if (!title?.trim() || !premise?.trim()) {
    res.status(400).json({ error: 'Title and premise are required' });
    return;
  }

  const pages = target_pages ?? 200;
  const wpp = words_per_page ?? 250;
  const chapters = target_chapters ?? 20;
  const totalWords = pages * wpp;
  const wordsPerChapter = Math.round(totalWords / chapters);

  const manifest: NovelManifest = {
    id: id ?? uuidv4(),
    title: title.trim(),
    premise: premise.trim(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    status: 'outlining',
    model_id: model_id ?? DEFAULT_NOVEL_MODEL,
    persona: 'novelist',
    target_chapter_count: chapters,
    current_chapter: 0,
    num_predict_novel: NUM_PREDICT_NOVEL,
    target_pages: pages,
    words_per_page: wpp,
    target_words: totalWords,
    words_per_chapter: wordsPerChapter,
  };

  const initialBible: NovelBible = {
    pov: bible?.pov?.trim() || '',
    characters: Array.isArray(bible?.characters) ? bible.characters : [],
    locations: Array.isArray(bible?.locations) ? bible.locations : [],
    facts: Array.isArray(bible?.facts) ? bible.facts : [],
    style_notes: bible?.style_notes?.trim() || '',
  };

  try {
    await storage.createNovel(manifest, initialBible);
    res.json({ novel: manifest });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/novels/:id/move — Move session-scoped novel to dedicated chat ID
router.post('/:id/move', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const { targetId } = req.body as { targetId?: string };
  if (!targetId?.trim()) {
    res.status(400).json({ error: 'Target chat ID is required' });
    return;
  }

  const moved = await storage.moveNovel(id, targetId.trim());
  if (!moved) {
    res.status(409).json({ error: 'Could not move novel to dedicated chat' });
    return;
  }
  res.json({ success: true, id: targetId.trim() });
});

// GET /api/novels/:id — Get novel with all details
router.get('/:id', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  try {
    const novel = await storage.getNovel(id);
    if (!novel) {
      res.status(404).json({ error: 'Novel not found' });
      return;
    }
    res.json(novel);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/novels/:id — Delete novel
router.delete('/:id', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  try {
    const deleted = await storage.deleteNovel(id);
    res.json({ success: deleted });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// PATCH /api/novels/:id — Update novel manifest
router.patch('/:id', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const updates = req.body as Partial<NovelManifest>;
  try {
    const updated = await storage.updateManifest(id, updates);
    if (!updated) {
      res.status(404).json({ error: 'Novel not found' });
      return;
    }
    res.json({ manifest: updated });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Bible operations
// ─────────────────────────────────────────────────────────────────────────────

// GET /api/novels/:id/bible
router.get('/:id/bible', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  try {
    const bible = await storage.getBible(id);
    res.json({ bible });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// PUT /api/novels/:id/bible
router.put('/:id/bible', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const bible = req.body as NovelBible;
  try {
    await storage.saveBible(id, bible);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/novels/:id/steering — Add or update author steering directives / course corrections
router.post('/:id/steering', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const { directive, notes } = req.body as { directive?: string; notes?: string[] };
  try {
    const bible = await storage.getBible(id);
    if (Array.isArray(notes)) {
      bible.steering_notes = notes.map(n => String(n).trim()).filter(Boolean);
    } else if (directive?.trim()) {
      bible.steering_notes = bible.steering_notes ?? [];
      bible.steering_notes.push(directive.trim());
    }
    await storage.saveBible(id, bible);
    res.json({ success: true, steering_notes: bible.steering_notes ?? [] });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Outline operations
// ─────────────────────────────────────────────────────────────────────────────

// GET /api/novels/:id/outline
router.get('/:id/outline', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  try {
    const outline = await storage.getOutline(id);
    res.json({ outline });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// PUT /api/novels/:id/outline — Save full outline
router.put('/:id/outline', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const { outline, revision } = req.body as { outline: ChapterOutline[]; revision?: boolean };
  try {
    if (revision) {
      const novel = await storage.getNovel(id);
      if (novel?.manifest.outline_revision_used) {
        res.status(409).json({ error: 'Final outline revision has already been used' });
        return;
      }
    }
    await storage.saveOutline(id, outline);
    await storage.clearDraftOutline(id);
    if (revision) {
      await storage.updateManifest(id, { outline_revision_used: true });
    }
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/novels/:id/outline/draft — Discard pending outline proposal
router.delete('/:id/outline/draft', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  try {
    await storage.clearDraftOutline(id);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/novels/:id/outline/propose — AI proposes chapters (SSE streaming)
router.post('/:id/outline/propose', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const { append, think, model_id, edited_outline, numCtx } = req.body as {
    append?: boolean;
    think?: boolean;
    model_id?: string;
    edited_outline?: ChapterOutline[] | null;
    numCtx?: number;
  };

  try {
    const novel = await storage.getNovel(id);
    if (!novel) {
      res.status(404).json({ error: 'Novel not found' });
      return;
    }

    const existingOutline = novel.outline;
    const modelId = model_id?.trim() || novel.manifest.model_id;
    if (modelId !== novel.manifest.model_id) {
      await storage.updateManifest(id, { model_id: modelId });
    }
    const revisionOutline = Array.isArray(edited_outline) && edited_outline.length > 0
      ? edited_outline
      : null;
    const revisionCount = revisionOutline?.length ?? 0;
    if (revisionOutline && novel.manifest.outline_revision_used) {
      res.status(409).json({ error: 'Final outline revision has already been used' });
      return;
    }
    const startFrom = append ? existingOutline.length + 1 : 1;
    const revision = Boolean(revisionOutline);
    const numChapters = revision
      ? revisionCount
      : append
        ? Math.max(1, novel.manifest.target_chapter_count - existingOutline.length)
        : novel.manifest.target_chapter_count;
    const targetWords = novel.manifest.target_words ?? 50000;
    const wordsPerChapter = novel.manifest.words_per_chapter ?? Math.round(targetWords / numChapters);
    const targetPages = novel.manifest.target_pages
      ?? Math.max(1, Math.round(targetWords / (novel.manifest.words_per_page || 250)));
    const existingPages = existingOutline.reduce(
      (sum, chapter) => sum + (chapter.target_pages ?? 0),
      0,
    );
    const pagesForProposal = revision
      ? targetPages
      : append
        ? Math.max(numChapters, targetPages - existingPages)
        : targetPages;
    const pagesPerChapter = Math.max(1, Math.round(pagesForProposal / numChapters));

    let contextPrompt = `Premise: ${novel.manifest.premise}\n\n`;
    const bible = novel.bible;
    if (bible.pov) {
      contextPrompt += `POV Style: ${bible.pov}\n`;
    }
    if (bible.characters.length > 0) {
      contextPrompt += `Main Characters:\n`;
      for (const char of bible.characters) {
        contextPrompt += `- ${char.name}: ${char.description}${char.traits.length ? ` (Traits: ${char.traits.join(', ')})` : ''}\n`;
      }
    }
    if (bible.locations.length > 0) {
      contextPrompt += `Key Locations:\n`;
      for (const loc of bible.locations) {
        contextPrompt += `- ${loc.name}: ${loc.description}\n`;
      }
    }
    if (bible.facts.length > 0) {
      contextPrompt += `World Facts: ${bible.facts.join('; ')}\n`;
    }
    if (bible.style_notes) {
      contextPrompt += `Style & Tone: ${bible.style_notes}\n`;
    }
    contextPrompt += '\n';
    contextPrompt += `Novel length target: ${targetPages} total pages, approximately ${targetWords.toLocaleString()} total words across ${novel.manifest.target_chapter_count} chapters.\n`;
    contextPrompt += `Average per chapter: ~${pagesPerChapter} pages and ~${wordsPerChapter.toLocaleString()} words (vary based on chapter importance).\n\n`;

    if (revisionOutline) {
      contextPrompt += `User-edited outline requiring its one final AI revision. Keep exactly ${revisionCount} chapters, preserve chapter numbers and titles unless correction is necessary, rewrite every description to exactly 2-3 sentences, identify characters involved, and adjust page allocations so they total exactly ${pagesForProposal} pages.\n`;
      for (const ch of revisionOutline ?? []) {
        contextPrompt += `- Chapter ${ch.number}: ${ch.title}; current pages=${ch.target_pages ?? 'unset'}; current description=${ch.description || ch.beat_summary}; characters=${(ch.characters_involved ?? []).join(', ') || 'unset'}\n`;
      }
    } else if (append && existingOutline.length > 0) {
      contextPrompt += `Existing chapters:\n`;
      for (const ch of existingOutline.slice(-5)) {
        contextPrompt += `- Chapter ${ch.number}: ${ch.title} — ${ch.description || ch.beat_summary}; characters=${(ch.characters_involved ?? []).join(', ')}; pages=${ch.target_pages ?? 'unset'}\n`;
      }
      contextPrompt += `\nContinue from chapter ${startFrom}.\n`;
    }
    contextPrompt += revision
      ? `\nReturn revised chapters as a JSON array. Output ONLY valid JSON array, no markdown.`
      : `\nPropose ${numChapters} chapters starting from chapter ${startFrom}. Allocate exactly ${pagesForProposal} pages across these chapters. Include description, characters_involved, target_pages, and target_words for each chapter. Output ONLY valid JSON array, no markdown.`;

    // Setup SSE for streaming progress
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const send = (event: string, data: unknown) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    console.log(`[Novel outline] Streaming from model ${modelId}...`);
    send('status', { message: 'Generating outline (attempt 1/3)...' });

    const collectOutline = async (
      thinkingEnabled: boolean,
      systemPrompt = OUTLINE_SYSTEM_PROMPT,
    ): Promise<{ content: string; error?: string }> => {
      let content = '';
      for await (const chunk of streamOllama(
        modelId,
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: contextPrompt },
        ],
        undefined,
        thinkingEnabled,
        OUTLINE_NUM_PREDICT,
        'json',
        numCtx,
      )) {
        if (chunk.error) return { content, error: chunk.error };
        if (chunk.thinking) send('thinking', { content: chunk.thinking });
        if (chunk.content) {
          content += chunk.content;
          send('token', { content: chunk.content });
        }
      }
      return { content };
    };

    let attempt = 1;
    let streamResult = await collectOutline(Boolean(think));
    let proposed = parseProposedOutline(streamResult.content, revision ? 1 : startFrom, revision);

    // Give model at most three attempts. First pass may think; later passes
    // prioritize valid compact JSON over reasoning.
    while ((!proposed || streamResult.error) && attempt < 3 && streamResult.error !== 'Aborted') {
      attempt += 1;
      const finalAttempt = attempt === 3;
      console.warn(`[Novel outline] Attempt ${attempt}/3 produced no usable JSON; retrying.`);
      send('status', {
        message: finalAttempt
          ? 'Final outline attempt: forcing concise JSON output...'
          : `No usable outline yet. Retrying (${attempt}/3)...`,
      });
      send('reset', {});
      streamResult = await collectOutline(false, finalAttempt ? FINAL_OUTLINE_SYSTEM_PROMPT : OUTLINE_SYSTEM_PROMPT);
      proposed = parseProposedOutline(streamResult.content, revision ? 1 : startFrom, revision);
    }

    console.log(`[Novel outline] Got ${streamResult.content.length} chars`);

    if (streamResult.error) {
      send('error', { message: streamResult.error });
      res.end();
      return;
    }

    if (!proposed) {
      const message = streamResult.content.trim()
        ? 'Model returned invalid outline JSON.'
        : 'Model returned empty response after retry.';
      console.error(`[Novel outline] ${message}`);
      send('error', { message });
      res.end();
      return;
    }

    if (revision && proposed.length !== revisionCount) {
      send('error', {
        message: `Revision returned ${proposed.length} chapters; expected ${revisionCount}.`,
      });
      res.end();
      return;
    }

    proposed = rebalancePageTargets(proposed, pagesForProposal);
    await storage.saveDraftOutline(id, proposed, revision);

    console.log(`[Novel outline] Proposed ${proposed.length} chapters`);
    send('done', { proposed, append: Boolean(append) });
    res.end();
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Chapter drafting with continuation loop
// ─────────────────────────────────────────────────────────────────────────────

// POST /api/novels/:id/chapters/:n/draft — Stream draft for chapter N
router.post('/:id/chapters/:n/draft', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const n = routeParam(req.params.n);
  const chapterNum = parseInt(n, 10);
  const { userNotes } = req.body as { userNotes?: string };

  if (isNaN(chapterNum) || chapterNum < 1) {
    res.status(400).json({ error: 'Invalid chapter number' });
    return;
  }

  try {
    const novel = await storage.getNovel(id);
    if (!novel) {
      res.status(404).json({ error: 'Novel not found' });
      return;
    }

    const outlineEntry = novel.outline.find(o => o.number === chapterNum);
    if (!outlineEntry) {
      res.status(400).json({ error: `Chapter ${chapterNum} not in outline` });
      return;
    }

    // Setup SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const send = (event: string, data: unknown) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const ac = new AbortController();
    novelAbortControllers.set(`${id}-${chapterNum}`, ac);

    // Build context
    const bible = novel.bible;
    const rollingContext = await storage.getRollingContext(id, chapterNum);

    // Assemble the prompt
    let contextContent = '';

    // Partition Bible Characters by chapter roster to prevent early appearance of future characters
    const rosterNames = [
      outlineEntry?.pov_character,
      ...(outlineEntry?.characters_involved ?? []),
      ...(outlineEntry?.characters ?? []),
    ].filter(Boolean).map(c => c!.toLowerCase().trim());

    const presentChars = bible.characters.filter(c => {
      const charNameLower = c.name.toLowerCase();
      return rosterNames.some(r => charNameLower.includes(r) || r.includes(charNameLower));
    });

    const futureChars = bible.characters.filter(c => {
      const charNameLower = c.name.toLowerCase();
      return !rosterNames.some(r => charNameLower.includes(r) || r.includes(charNameLower));
    });

    // Bible
    if (bible.pov || bible.characters.length > 0 || bible.locations.length > 0 || bible.facts.length > 0 || bible.style_notes) {
      contextContent += '## Character & World Bible\n';
      if (bible.pov) contextContent += `POV Style: ${bible.pov}\n`;
      if (presentChars.length > 0) {
        contextContent += '### Approved Characters For This Chapter\n';
        for (const char of presentChars) {
          contextContent += `- **${char.name}**: ${char.description}`;
          if (char.traits.length) contextContent += ` Traits: ${char.traits.join(', ')}.`;
          if (char.relationships.length) contextContent += ` Relationships: ${char.relationships.join(', ')}.`;
          contextContent += '\n';
        }
      }
      if (bible.locations.length > 0) {
        contextContent += '### Key Locations\n';
        for (const loc of bible.locations) {
          contextContent += `- **${loc.name}**: ${loc.description}\n`;
        }
      }
      if (bible.facts.length > 0) {
        contextContent += '### Established World Facts\n';
        for (const fact of bible.facts) {
          contextContent += `- ${fact}\n`;
        }
      }
      if (bible.style_notes) {
        contextContent += `### Style Notes\n${bible.style_notes}\n`;
      }
      contextContent += '\n';
    }

    // Author Steering & Directives
    if (bible.steering_notes && bible.steering_notes.length > 0) {
      contextContent += '## AUTHOR MANDATORY DIRECTIVES & COURSE CORRECTIONS\n';
      for (const note of bible.steering_notes) {
        contextContent += `- ${note}\n`;
      }
      contextContent += '\n';
    }

    // Arc summary
    if (rollingContext.arcSummaries.length > 0) {
      contextContent += '## Story So Far (Arc Summary)\n';
      for (const arc of rollingContext.arcSummaries) {
        contextContent += `Chapters ${arc.chapters_covered.join('-')}: ${arc.summary}\n`;
        if (arc.major_threads.length) {
          contextContent += `Open threads: ${arc.major_threads.join('; ')}\n`;
        }
      }
      contextContent += '\n';
    }

    // Recent chapter summaries
    if (rollingContext.recentSummaries.length > 0) {
      contextContent += '## Recent Chapters\n';
      for (const sum of rollingContext.recentSummaries) {
        contextContent += `### Chapter ${sum.chapter}\n${sum.summary}\n`;
        if (sum.open_threads.length) {
          contextContent += `Open threads: ${sum.open_threads.join('; ')}\n`;
        }
      }
      contextContent += '\n';
    }

    // Chapter outline
    contextContent += `## STRICT CHAPTER OUTLINE & SCOPE\n`;
    contextContent += `**Chapter ${outlineEntry.number}: ${outlineEntry.title}**\n`;
    contextContent += `POV Character: ${outlineEntry.pov_character}\n`;
    contextContent += `Description / Beats: ${outlineEntry.description || outlineEntry.beat_summary}\n`;
    contextContent += `Approved Roster: ${presentChars.map(c => c.name).join(', ') || (outlineEntry.characters_involved ?? []).join(', ') || 'POV character only'}\n`;
    if (outlineEntry.target_pages || outlineEntry.target_words) {
      contextContent += `Target length: ${outlineEntry.target_pages ?? '?'} pages, approximately ${outlineEntry.target_words ?? '?'} words.\n`;
    }
    if (userNotes) {
      contextContent += `User notes: ${userNotes}\n`;
    }
    if (futureChars.length > 0) {
      contextContent += `\n⚠️ ABSOLUTE CHARACTER RESTRICTION:\n`;
      contextContent += `DO NOT include, introduce, mention, or feature the following characters in Chapter ${chapterNum} (they belong to LATER chapters and are NOT in this scene): ${futureChars.map(c => c.name).join(', ')}.\n`;
    }
    contextContent += `\nSTRICT OUTLINE ADHERENCE MANDATE:\n`;
    contextContent += `1. Follow ONLY the beats described in this chapter's outline.\n`;
    contextContent += `2. Do NOT introduce future characters or jump ahead to future outline events.\n`;
    contextContent += `3. Obey all Author Mandatory Directives above strictly.\n\n`;

    // Raw tail for voice continuity
    if (rollingContext.rawTail) {
      contextContent += `## Previous Chapter Ending (for voice continuity)\n`;
      contextContent += `...${rollingContext.rawTail}\n\n`;
    }

    contextContent += `Now write Chapter ${chapterNum}. When complete, end with [END_OF_CHAPTER] on its own line.`;

    let fullDraft = '';
    let segmentIndex = 0;
    let completed = false;

    // Continuation loop
    while (!completed && segmentIndex < MAX_CONTINUATION_SEGMENTS && !ac.signal.aborted) {
      send('segment_start', { segment: segmentIndex });

      const messages: Array<{ role: string; content: string }> = [
        { role: 'system', content: NOVELIST_SYSTEM_PROMPT },
      ];

      if (segmentIndex === 0) {
        messages.push({ role: 'user', content: contextContent });
      } else {
        // Continuation prompt
        const tail = fullDraft.slice(-2000);
        messages.push({
          role: 'user',
          content: `Continue the chapter exactly where this text ends. Do not recap or restart. When complete, end with [END_OF_CHAPTER].\n\n...${tail}`,
        });
      }

      let segmentContent = '';

      for await (const chunk of streamOllama(novel.manifest.model_id, messages, ac.signal)) {
        if (chunk.error) {
          if (chunk.error !== 'Aborted') {
            send('error', { message: chunk.error });
          }
          break;
        }
        if (chunk.content) {
          segmentContent += chunk.content;
          send('token', { content: chunk.content });
        }
        if (chunk.done) {
          break;
        }
      }

      // Check for completion marker
      if (segmentContent.includes(END_OF_CHAPTER_MARKER)) {
        const markerIdx = segmentContent.indexOf(END_OF_CHAPTER_MARKER);
        segmentContent = segmentContent.slice(0, markerIdx).trim();
        completed = true;
      }

      fullDraft += segmentContent;
      segmentIndex++;

      send('segment_end', { segment: segmentIndex - 1, completed });

      // Save checkpoint
      await storage.saveChapterText(id, chapterNum, fullDraft);
    }

    // Update outline status
    await storage.updateChapterOutline(id, chapterNum, { status: 'drafted' });
    await storage.updateManifest(id, { current_chapter: chapterNum, status: 'drafting' });

    send('done', { segments: segmentIndex, completed, wordCount: fullDraft.split(/\s+/).length });

    novelAbortControllers.delete(`${id}-${chapterNum}`);
    res.end();
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/novels/:id/chapters/:n/stop — Stop ongoing draft
router.post('/:id/chapters/:n/stop', (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const n = routeParam(req.params.n);
  const key = `${id}-${n}`;
  const ac = novelAbortControllers.get(key);
  if (ac) {
    ac.abort();
    novelAbortControllers.delete(key);
  }
  res.json({ stopped: true });
});

// GET /api/novels/:id/chapters/:n — Get chapter text
router.get('/:id/chapters/:n', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const n = routeParam(req.params.n);
  const chapterNum = parseInt(n, 10);
  try {
    const text = await storage.getChapterText(id, chapterNum);
    if (text === null) {
      res.status(404).json({ error: 'Chapter not found' });
      return;
    }
    res.json({ chapter: chapterNum, text, wordCount: text.split(/\s+/).length });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// PUT /api/novels/:id/chapters/:n — Save/update chapter text manually
router.put('/:id/chapters/:n', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const n = routeParam(req.params.n);
  const chapterNum = parseInt(n, 10);
  const { text } = req.body as { text: string };
  try {
    await storage.saveChapterText(id, chapterNum, text);
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Summarization
// ─────────────────────────────────────────────────────────────────────────────

// POST /api/novels/:id/chapters/:n/summarize — Generate summary for chapter
router.post('/:id/chapters/:n/summarize', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const n = routeParam(req.params.n);
  const chapterNum = parseInt(n, 10);

  try {
    const novel = await storage.getNovel(id);
    if (!novel) {
      res.status(404).json({ error: 'Novel not found' });
      return;
    }

    const chapterText = await storage.getChapterText(id, chapterNum);
    if (!chapterText) {
      res.status(404).json({ error: 'Chapter text not found' });
      return;
    }

    const result = await callOllama(
      novel.manifest.model_id,
      [
        { role: 'system', content: SUMMARIZATION_SYSTEM_PROMPT },
        { role: 'user', content: `Chapter ${chapterNum}:\n\n${chapterText}` },
      ],
      { format: 'json', numPredict: 2000 }
    );

    if (result.error) {
      res.status(500).json({ error: result.error });
      return;
    }

    let summary: ChapterSummary;
    try {
      summary = JSON.parse(result.content);
      summary.chapter = chapterNum;

      // Extract last 600 words
      const words = chapterText.split(/\s+/);
      summary.last_600_words_raw = words.slice(-600).join(' ');
    } catch {
      res.status(500).json({ error: 'Failed to parse summary JSON from model' });
      return;
    }

    await storage.saveChapterSummary(id, chapterNum, summary);
    res.json({ summary });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/novels/:id/chapters/:n/summary — Get chapter summary
router.get('/:id/chapters/:n/summary', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const n = routeParam(req.params.n);
  const chapterNum = parseInt(n, 10);
  try {
    const summary = await storage.getChapterSummary(id, chapterNum);
    if (!summary) {
      res.status(404).json({ error: 'Summary not found' });
      return;
    }
    res.json({ summary });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Arc summary compaction (every ~5 chapters)
// ─────────────────────────────────────────────────────────────────────────────

// POST /api/novels/:id/arcs/:arcNum/compact — Generate arc summary
router.post('/:id/arcs/:arcNum/compact', async (req: Request, res: Response) => {
  const id = routeParam(req.params.id);
  const arcNum = routeParam(req.params.arcNum);
  const arcNumber = parseInt(arcNum, 10);

  try {
    const novel = await storage.getNovel(id);
    if (!novel) {
      res.status(404).json({ error: 'Novel not found' });
      return;
    }

    // Arc N covers chapters (N-1)*5+1 to N*5
    const startChapter = (arcNumber - 1) * 5 + 1;
    const endChapter = arcNumber * 5;

    const summaries: ChapterSummary[] = [];
    for (let i = startChapter; i <= endChapter; i++) {
      const sum = await storage.getChapterSummary(id, i);
      if (sum) summaries.push(sum);
    }

    if (summaries.length === 0) {
      res.status(400).json({ error: 'No chapter summaries found for this arc' });
      return;
    }

    const summaryText = summaries.map(s =>
      `Chapter ${s.chapter}: ${s.summary}\nOpen threads: ${s.open_threads.join('; ')}`
    ).join('\n\n');

    const result = await callOllama(
      novel.manifest.model_id,
      [
        { role: 'system', content: `Compact these chapter summaries into a single arc summary. Output JSON with: arc_number, chapters_covered (array), summary (2-3 paragraphs), characters_state (array), major_threads (array of unresolved elements).` },
        { role: 'user', content: `Arc ${arcNumber} (chapters ${startChapter}-${endChapter}):\n\n${summaryText}` },
      ],
      { format: 'json', numPredict: 2000 }
    );

    if (result.error) {
      res.status(500).json({ error: result.error });
      return;
    }

    let arcSummary: ArcSummary;
    try {
      arcSummary = JSON.parse(result.content);
      arcSummary.arc_number = arcNumber;
      arcSummary.chapters_covered = summaries.map(s => s.chapter);
    } catch {
      res.status(500).json({ error: 'Failed to parse arc summary JSON' });
      return;
    }

    await storage.saveArcSummary(id, arcNumber, arcSummary);
    res.json({ arcSummary });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
