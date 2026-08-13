/**
 * Novel Mode storage layer.
 * Handles all file I/O for .novels/ directory structure.
 */

import fs from 'fs/promises';
import path from 'path';
import { WORKSPACE_DIR } from '../tools.js';
import type {
  NovelManifest,
  ChapterOutline,
  ChapterSummary,
  ArcSummary,
  NovelBible,
  NovelWithDetails,
} from './types.js';

const NOVELS_DIR = path.resolve(WORKSPACE_DIR, '.novels');

// Ensure base novels directory exists
export async function ensureNovelsDir(): Promise<void> {
  await fs.mkdir(NOVELS_DIR, { recursive: true });
}

function novelPath(novelId: string): string {
  return path.join(NOVELS_DIR, novelId);
}

function manifestPath(novelId: string): string {
  return path.join(novelPath(novelId), 'manifest.json');
}

function biblePath(novelId: string): string {
  return path.join(novelPath(novelId), 'bible.json');
}

function outlinePath(novelId: string): string {
  return path.join(novelPath(novelId), 'outline.json');
}

function draftOutlinePath(novelId: string): string {
  return path.join(novelPath(novelId), 'outline.draft.json');
}

function chapterPath(novelId: string, chapterNum: number): string {
  const padded = String(chapterNum).padStart(2, '0');
  return path.join(novelPath(novelId), 'chapters', `ch${padded}.txt`);
}

function summaryPath(novelId: string, chapterNum: number): string {
  const padded = String(chapterNum).padStart(2, '0');
  return path.join(novelPath(novelId), 'summaries', `ch${padded}.json`);
}

function arcSummaryPath(novelId: string, arcNum: number): string {
  const padded = String(arcNum).padStart(2, '0');
  return path.join(novelPath(novelId), 'summaries', `arc-${padded}.json`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Novel CRUD
// ─────────────────────────────────────────────────────────────────────────────

export async function createNovel(manifest: NovelManifest, initialBible?: NovelBible): Promise<void> {
  const basePath = novelPath(manifest.id);
  await fs.mkdir(basePath, { recursive: true });
  await fs.mkdir(path.join(basePath, 'chapters'), { recursive: true });
  await fs.mkdir(path.join(basePath, 'summaries'), { recursive: true });

  await fs.writeFile(manifestPath(manifest.id), JSON.stringify(manifest, null, 2), 'utf-8');

  // Initialize bible and outline
  const bibleToSave: NovelBible = initialBible ?? {
    pov: '',
    characters: [],
    locations: [],
    facts: [],
    style_notes: '',
  };
  await fs.writeFile(biblePath(manifest.id), JSON.stringify(bibleToSave, null, 2), 'utf-8');
  await fs.writeFile(outlinePath(manifest.id), '[]', 'utf-8');
  await fs.rm(draftOutlinePath(manifest.id), { force: true });
}

export async function listNovels(): Promise<NovelManifest[]> {
  await ensureNovelsDir();
  const entries = await fs.readdir(NOVELS_DIR, { withFileTypes: true });
  const novels: NovelManifest[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    try {
      const content = await fs.readFile(path.join(NOVELS_DIR, entry.name, 'manifest.json'), 'utf-8');
      novels.push(JSON.parse(content));
    } catch {
      // Skip invalid novel directories
    }
  }

  return novels.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
}

export async function getNovel(novelId: string): Promise<NovelWithDetails | null> {
  try {
    const manifestContent = await fs.readFile(manifestPath(novelId), 'utf-8');
    const manifest: NovelManifest = JSON.parse(manifestContent);

    let outline: ChapterOutline[] = [];
    try {
      const outlineContent = await fs.readFile(outlinePath(novelId), 'utf-8');
      outline = JSON.parse(outlineContent);
    } catch {}

    let draftOutline: ChapterOutline[] = [];
    try {
      const draftContent = await fs.readFile(draftOutlinePath(novelId), 'utf-8');
      draftOutline = JSON.parse(draftContent);
    } catch {}

    let bible: NovelBible = { characters: [], locations: [], facts: [], style_notes: '' };
    try {
      const bibleContent = await fs.readFile(biblePath(novelId), 'utf-8');
      bible = JSON.parse(bibleContent);
    } catch {}

    // Gather chapter metadata
    const chaptersDir = path.join(novelPath(novelId), 'chapters');
    const chapters: { number: number; wordCount: number; status: string }[] = [];
    try {
      const files = await fs.readdir(chaptersDir);
      for (const file of files) {
        const match = file.match(/^ch(\d+)\.txt$/);
        if (!match) continue;
        const num = parseInt(match[1], 10);
        const content = await fs.readFile(path.join(chaptersDir, file), 'utf-8');
        const wordCount = content.split(/\s+/).filter(Boolean).length;
        const outlineEntry = outline.find(o => o.number === num);
        chapters.push({
          number: num,
          wordCount,
          status: outlineEntry?.status ?? 'drafted',
        });
      }
    } catch {}

    return {
      manifest,
      outline,
      draft_outline: draftOutline,
      bible,
      chapters: chapters.sort((a, b) => a.number - b.number),
    };
  } catch {
    return null;
  }
}

export async function updateManifest(novelId: string, updates: Partial<NovelManifest>): Promise<NovelManifest | null> {
  try {
    const content = await fs.readFile(manifestPath(novelId), 'utf-8');
    const manifest: NovelManifest = JSON.parse(content);
    const updated = { ...manifest, ...updates, updated_at: new Date().toISOString() };
    await fs.writeFile(manifestPath(novelId), JSON.stringify(updated, null, 2), 'utf-8');
    return updated;
  } catch {
    return null;
  }
}

export async function deleteNovel(novelId: string): Promise<boolean> {
  try {
    await fs.rm(novelPath(novelId), { recursive: true, force: true });
    return true;
  } catch {
    return false;
  }
}

export async function moveNovel(novelId: string, targetId: string): Promise<boolean> {
  if (!/^[A-Za-z0-9_-]+$/.test(novelId) || !/^[A-Za-z0-9_-]+$/.test(targetId)) {
    return false;
  }
  if (novelId === targetId) return true;

  try {
    const sourcePath = novelPath(novelId);
    const targetPath = novelPath(targetId);
    await fs.access(sourcePath);
    await fs.access(targetPath);
    return false;
  } catch {
    // Target normally does not exist. Continue with the atomic rename.
  }

  try {
    await fs.rename(novelPath(novelId), novelPath(targetId));
    const content = await fs.readFile(manifestPath(targetId), 'utf-8');
    const manifest = JSON.parse(content) as NovelManifest;
    manifest.id = targetId;
    manifest.dedicated_chat = true;
    manifest.updated_at = new Date().toISOString();
    await fs.writeFile(manifestPath(targetId), JSON.stringify(manifest, null, 2), 'utf-8');
    return true;
  } catch {
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Outline operations
// ─────────────────────────────────────────────────────────────────────────────

export async function getOutline(novelId: string): Promise<ChapterOutline[]> {
  try {
    const content = await fs.readFile(outlinePath(novelId), 'utf-8');
    return JSON.parse(content);
  } catch {
    return [];
  }
}

export async function saveOutline(novelId: string, outline: ChapterOutline[]): Promise<void> {
  await fs.writeFile(outlinePath(novelId), JSON.stringify(outline, null, 2), 'utf-8');
  await updateManifest(novelId, { target_chapter_count: outline.length });
}

export async function saveDraftOutline(
  novelId: string,
  outline: ChapterOutline[],
  isRevision = false,
): Promise<void> {
  await fs.writeFile(draftOutlinePath(novelId), JSON.stringify(outline, null, 2), 'utf-8');
  await updateManifest(novelId, { draft_outline_is_revision: isRevision });
}

export async function clearDraftOutline(novelId: string): Promise<void> {
  await fs.rm(draftOutlinePath(novelId), { force: true });
  await updateManifest(novelId, { draft_outline_is_revision: undefined });
}

export async function updateChapterOutline(
  novelId: string,
  chapterNum: number,
  updates: Partial<ChapterOutline>
): Promise<void> {
  const outline = await getOutline(novelId);
  const idx = outline.findIndex(o => o.number === chapterNum);
  if (idx >= 0) {
    outline[idx] = { ...outline[idx], ...updates };
    await saveOutline(novelId, outline);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bible operations
// ─────────────────────────────────────────────────────────────────────────────

export async function getBible(novelId: string): Promise<NovelBible> {
  try {
    const content = await fs.readFile(biblePath(novelId), 'utf-8');
    return JSON.parse(content);
  } catch {
    return { characters: [], locations: [], facts: [], style_notes: '' };
  }
}

export async function saveBible(novelId: string, bible: NovelBible): Promise<void> {
  await fs.writeFile(biblePath(novelId), JSON.stringify(bible, null, 2), 'utf-8');
}

export async function getCharacterLibrary(): Promise<{ name: string; description: string; relationships: string[]; source_novel: string }[]> {
  const manifests = await listNovels();
  const libraryMap = new Map<string, { name: string; description: string; relationships: string[]; source_novel: string }>();

  for (const manifest of manifests) {
    try {
      const bible = await getBible(manifest.id);
      for (const char of bible.characters) {
        if (!char.name?.trim()) continue;
        const key = char.name.trim().toLowerCase();
        if (!libraryMap.has(key)) {
          libraryMap.set(key, {
            name: char.name.trim(),
            description: char.description || '',
            relationships: Array.isArray(char.relationships) ? char.relationships : [],
            source_novel: manifest.title,
          });
        }
      }
    } catch {}
  }

  return Array.from(libraryMap.values());
}

// ─────────────────────────────────────────────────────────────────────────────
// Chapter text operations
// ─────────────────────────────────────────────────────────────────────────────

export async function getChapterText(novelId: string, chapterNum: number): Promise<string | null> {
  try {
    return await fs.readFile(chapterPath(novelId, chapterNum), 'utf-8');
  } catch {
    return null;
  }
}

export async function saveChapterText(novelId: string, chapterNum: number, text: string): Promise<void> {
  const filePath = chapterPath(novelId, chapterNum);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, text, 'utf-8');
}

export async function appendChapterText(novelId: string, chapterNum: number, text: string): Promise<void> {
  const filePath = chapterPath(novelId, chapterNum);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  try {
    await fs.appendFile(filePath, text, 'utf-8');
  } catch {
    await fs.writeFile(filePath, text, 'utf-8');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary operations
// ─────────────────────────────────────────────────────────────────────────────

export async function getChapterSummary(novelId: string, chapterNum: number): Promise<ChapterSummary | null> {
  try {
    const content = await fs.readFile(summaryPath(novelId, chapterNum), 'utf-8');
    return JSON.parse(content);
  } catch {
    return null;
  }
}

export async function saveChapterSummary(novelId: string, chapterNum: number, summary: ChapterSummary): Promise<void> {
  const filePath = summaryPath(novelId, chapterNum);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(summary, null, 2), 'utf-8');
}

export async function getArcSummary(novelId: string, arcNum: number): Promise<ArcSummary | null> {
  try {
    const content = await fs.readFile(arcSummaryPath(novelId, arcNum), 'utf-8');
    return JSON.parse(content);
  } catch {
    return null;
  }
}

export async function saveArcSummary(novelId: string, arcNum: number, summary: ArcSummary): Promise<void> {
  const filePath = arcSummaryPath(novelId, arcNum);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(summary, null, 2), 'utf-8');
}

/**
 * Get rolling context for drafting chapter N.
 * Returns summaries from last 2-3 chapters plus latest arc summary if available.
 */
export async function getRollingContext(
  novelId: string,
  targetChapter: number
): Promise<{
  arcSummaries: ArcSummary[];
  recentSummaries: ChapterSummary[];
  rawTail: string;
}> {
  const arcSummaries: ArcSummary[] = [];
  const recentSummaries: ChapterSummary[] = [];
  let rawTail = '';

  // Get latest arc summary (every 5 chapters)
  const latestArcNum = Math.floor((targetChapter - 1) / 5);
  if (latestArcNum > 0) {
    const arc = await getArcSummary(novelId, latestArcNum);
    if (arc) arcSummaries.push(arc);
  }

  // Get last 2-3 chapter summaries (not covered by arc)
  const arcCoverageEnd = latestArcNum * 5;
  for (let i = Math.max(1, targetChapter - 3); i < targetChapter; i++) {
    if (i <= arcCoverageEnd && latestArcNum > 0) continue; // Skip if covered by arc
    const summary = await getChapterSummary(novelId, i);
    if (summary) recentSummaries.push(summary);
  }

  // Get raw tail from previous chapter
  if (targetChapter > 1) {
    const prevSummary = await getChapterSummary(novelId, targetChapter - 1);
    if (prevSummary?.last_600_words_raw) {
      rawTail = prevSummary.last_600_words_raw;
    } else {
      // Fallback: read last 600 words from chapter file
      const prevText = await getChapterText(novelId, targetChapter - 1);
      if (prevText) {
        const words = prevText.split(/\s+/);
        rawTail = words.slice(-600).join(' ');
      }
    }
  }

  return { arcSummaries, recentSummaries, rawTail };
}
