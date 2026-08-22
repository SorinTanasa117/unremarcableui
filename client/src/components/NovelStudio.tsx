import React, { useEffect, useState, useCallback } from 'react';
import type { InferenceBackend } from '../hooks/useOllamaStream';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface NovelManifest {
  id: string;
  title: string;
  premise: string;
  created_at: string;
  updated_at: string;
  status: 'outlining' | 'drafting' | 'complete';
  model_id: string;
  target_chapter_count: number;
  current_chapter: number;
  dedicated_chat?: boolean;
  outline_revision_used?: boolean;
  draft_outline_is_revision?: boolean;
}

interface ChapterOutline {
  number: number;
  title: string;
  beat_summary: string;
  description?: string;
  pov_character: string;
  characters_involved?: string[];
  target_pages?: number;
  target_words?: number;
  status: 'planned' | 'drafted' | 'revised';
}

interface BibleCharacter {
  name: string;
  description: string;
  traits: string[];
  relationships: string[];
}

interface BibleLocation {
  name: string;
  description: string;
}

interface NovelBible {
  pov?: string;
  characters: BibleCharacter[];
  locations: BibleLocation[];
  facts: string[];
  style_notes: string;
  steering_notes?: string[];
}

interface ChapterMeta {
  number: number;
  wordCount: number;
  status: string;
}

interface NovelWithDetails {
  manifest: NovelManifest;
  outline: ChapterOutline[];
  draft_outline: ChapterOutline[];
  bible: NovelBible;
  chapters: ChapterMeta[];
}

// ─────────────────────────────────────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────────────────────────────────────

async function fetchNovel(sessionId: string): Promise<NovelWithDetails | null> {
  const res = await fetch(`/api/novels/${sessionId}`);
  if (!res.ok) return null;
  return res.json();
}

async function saveBible(id: string, bible: NovelBible): Promise<void> {
  const res = await fetch(`/api/novels/${id}/bible`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bible),
  });
  if (!res.ok) throw new Error('Failed to save character bible.');
}

interface PastCharacter {
  name: string;
  description: string;
  relationships: string[];
  source_novel: string;
}

async function fetchCharacterLibrary(): Promise<PastCharacter[]> {
  const res = await fetch('/api/novels/characters/library');
  if (!res.ok) return [];
  const data = await res.json();
  return data.characters ?? [];
}

interface CreateNovelParams {
  id: string;
  title: string;
  premise: string;
  model_id: string;
  target_pages: number;
  words_per_page: number;
  target_chapters: number;
  bible?: {
    pov?: string;
    style_notes?: string;
    characters?: BibleCharacter[];
    locations?: BibleLocation[];
    facts?: string[];
  };
}

async function createNovel(params: CreateNovelParams): Promise<NovelManifest> {
  const res = await fetch('/api/novels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.novel;
}

async function moveNovel(id: string, targetId: string): Promise<void> {
  const res = await fetch(`/api/novels/${encodeURIComponent(id)}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ targetId }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || 'Failed to move novel to dedicated chat');
}

async function proposeOutlineStream(
  id: string,
  count: number,
  append: boolean,
  think: boolean,
  modelId: string,
  editedOutline: ChapterOutline[] | null,
  numCtx: number | undefined,
  onToken: (content: string) => void,
  onThinking: (content: string) => void,
  onStatus: (message: string) => void,
  onReset: () => void,
): Promise<{ proposed: ChapterOutline[]; append: boolean } | { error: string }> {
  const res = await fetch(`/api/novels/${id}/outline/propose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      count,
      append,
      think,
      model_id: modelId,
      edited_outline: editedOutline,
      numCtx,
    }),
  });

  if (!res.ok || !res.body) {
    const text = await res.text();
    return { error: text || 'Request failed' };
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: { proposed: ChapterOutline[]; append: boolean } | null = null;
  let errorMsg: string | null = null;
  let eventName = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventName = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          if (eventName === 'thinking') onThinking(data.content || '');
          else if (eventName === 'token') onToken(data.content || '');
          else if (eventName === 'status') onStatus(data.message || '');
          else if (eventName === 'reset') onReset();
          else if (eventName === 'done' && 'proposed' in data) {
            result = { proposed: data.proposed, append: data.append };
          } else if (eventName === 'error') {
            errorMsg = data.message || 'Outline generation failed';
          }
        } catch {}
        eventName = '';
      }
    }
  }

  if (errorMsg) return { error: errorMsg };
  if (result) return result;
  return { error: 'No response' };
}

async function saveOutline(id: string, outline: ChapterOutline[], revision = false): Promise<void> {
  const res = await fetch(`/api/novels/${id}/outline`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ outline, revision }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || 'Failed to save outline');
}

async function clearDraftOutline(id: string): Promise<void> {
  const res = await fetch(`/api/novels/${encodeURIComponent(id)}/outline/draft`, {
    method: 'DELETE',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || 'Failed to discard outline proposal');
}

async function fetchChapterText(id: string, num: number): Promise<string | null> {
  const res = await fetch(`/api/novels/${id}/chapters/${num}`);
  if (!res.ok) return null;
  const data = await res.json();
  return data.text ?? null;
}

async function saveChapterTextApi(id: string, num: number, text: string): Promise<boolean> {
  const res = await fetch(`/api/novels/${id}/chapters/${num}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return res.ok;
}



// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

interface Props {
  sessionId: string;
  modelId: string;
  thinkingMode: boolean;
  contextSize?: number;
  isRunning?: boolean;
  stop?: (sessionId: string) => void;
  killModel?: (model: string, inferenceBackend: InferenceBackend) => Promise<void>;
  inferenceBackend?: InferenceBackend;
  onDraftChapter: (novelId: string, chapterNum: number, chapterTitle: string) => void;
  onCreateNovelSession: (title: string) => Promise<string>; // Creates new session, returns session ID
  onActivateNovelSession: (sessionId: string) => void;
  onOutlineReady: (hasOutline: boolean, firstChapter?: ChapterOutline) => void;
}

export function NovelStudio({
  sessionId,
  modelId,
  thinkingMode,
  contextSize,
  isRunning,
  stop,
  killModel,
  inferenceBackend,
  onDraftChapter,
  onCreateNovelSession,
  onActivateNovelSession,
  onOutlineReady,
}: Props) {
  const [novel, setNovel] = useState<NovelWithDetails | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ text: string; type: 'info' | 'error' | 'success' } | null>(null);
  const [outlineStream, setOutlineStream] = useState('');
  const [thinkingStream, setThinkingStream] = useState('');
  const [proposedOutline, setProposedOutline] = useState<ChapterOutline[] | null>(null);
  const [outlineDraft, setOutlineDraft] = useState<ChapterOutline[] | null>(null);
  const [outlineBaseline, setOutlineBaseline] = useState<ChapterOutline[] | null>(null);
  const [proposalIsRevision, setProposalIsRevision] = useState(false);
  const [outlineRevisionUsed, setOutlineRevisionUsed] = useState(false);
  const [editingOutline, setEditingOutline] = useState(false);
  
  // Track current novel session ID (may differ from chat sessionId)
  const [novelSessionId, setNovelSessionId] = useState<string | null>(null);

  // Setup form (shown when no novel exists)
  const [title, setTitle] = useState('');
  const [premise, setPremise] = useState('');
  const [targetPages, setTargetPages] = useState(200); // Standard novel ~200-300 pages
  const [wordsPerPage, setWordsPerPage] = useState(250); // Standard: 250 words/page
  const [targetChapters, setTargetChapters] = useState(20);

  // Setup form Bible fields
  const [povMode, setPovMode] = useState<'Auto - variable' | 'Auto - single character' | 'External narrator' | 'User defined'>('Auto - variable');
  const [customPov, setCustomPov] = useState('');
  const [formCharacters, setFormCharacters] = useState<{ name: string; description: string; relationships: string }[]>([]);
  const [pastCharacters, setPastCharacters] = useState<PastCharacter[]>([]);
  const [styleNotes, setStyleNotes] = useState('');
  const [formFacts, setFormFacts] = useState<string[]>([]);
  const [formLocations, setFormLocations] = useState<{ name: string; description: string }[]>([]);

  // Bible editing
  const [editingBible, setEditingBible] = useState(false);
  const [bibleDraft, setBibleDraft] = useState<NovelBible | null>(null);

  // Chapter text viewer/editor
  const [viewingChapter, setViewingChapter] = useState<number | null>(null);
  const [chapterText, setChapterText] = useState('');
  const [chapterTextLoading, setChapterTextLoading] = useState(false);
  const [chapterTextDirty, setChapterTextDirty] = useState(false);

  // Load novel for this session
  const loadNovel = useCallback(async () => {
    try {
      const data = await fetchNovel(sessionId);
      setNovel(data);
      setNovelSessionId(data?.manifest.dedicated_chat ? sessionId : null);
      setOutlineRevisionUsed(Boolean(data?.manifest.outline_revision_used));
      if (data) {
        const pendingOutline = data.draft_outline?.length ? data.draft_outline : null;
        setProposedOutline(pendingOutline);
        setProposalIsRevision(Boolean(pendingOutline && data.manifest.draft_outline_is_revision));
        setOutlineDraft(pendingOutline ?? data.outline);
        setOutlineBaseline(pendingOutline ?? data.outline);
        if (pendingOutline) {
          setStatusMessage({ text: 'Restored pending outline proposal. Accept or discard it.', type: 'info' });
        }
      }
    } catch {}
  }, [sessionId]);

  useEffect(() => {
    loadNovel();
  }, [loadNovel]);

  // Load character library from past novels for the setup form
  useEffect(() => {
    if (novel) return; // only needed when no novel exists yet
    fetchCharacterLibrary().then(setPastCharacters).catch(() => {});
  }, [novel]);

  useEffect(() => {
    setOutlineDraft(null);
    setOutlineBaseline(null);
    setProposedOutline(null);
    setProposalIsRevision(false);
    setOutlineRevisionUsed(false);
    setEditingOutline(false);
  }, [sessionId]);

  // ── Create novel ──
  const handleCreate = async () => {
    if (!title.trim() || !premise.trim()) return;
    setIsLoading(true);
    setStatusMessage({ text: 'Creating novel and initializing story bible...', type: 'info' });
    try {
      const selectedPov = povMode === 'User defined' ? customPov.trim() : povMode;
      const formattedCharacters: BibleCharacter[] = formCharacters
        .filter(c => c.name.trim())
        .map(c => ({
          name: c.name.trim(),
          description: c.description.trim(),
          traits: [],
          relationships: c.relationships.split(',').map(r => r.trim()).filter(Boolean),
        }));
      const formattedLocations: BibleLocation[] = formLocations
        .filter(l => l.name.trim())
        .map(l => ({
          name: l.name.trim(),
          description: l.description.trim(),
        }));
      const formattedFacts = formFacts.map(f => f.trim()).filter(Boolean);

      await createNovel({
        id: sessionId,
        title: title.trim(),
        premise: premise.trim(),
        model_id: modelId,
        target_pages: targetPages,
        words_per_page: wordsPerPage,
        target_chapters: targetChapters,
        bible: {
          pov: selectedPov,
          style_notes: styleNotes.trim(),
          characters: formattedCharacters,
          locations: formattedLocations,
          facts: formattedFacts,
        },
      });
      setStatusMessage(null);
      await loadNovel();
    } catch (err: any) {
      setStatusMessage({ text: err.message || 'Failed to create', type: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  // ── Propose outline ──
  const handleProposeOutline = async () => {
    if (!novel) return;

    const currentDraft = outlineDraft ?? novel.outline;
    const baseline = outlineBaseline ?? novel.outline;
    const hasEdits = JSON.stringify(currentDraft) !== JSON.stringify(baseline);
    const isRevision = hasEdits && (novel.outline.length > 0 || proposedOutline !== null);
    if (isRevision && outlineRevisionUsed) {
      setStatusMessage({ text: 'Final AI outline revision already used. Save edits directly.', type: 'error' });
      return;
    }
    
    // Create dedicated session for this novel if not exists
    let novelId = novel.manifest.id;
    if (!novelSessionId) {
      const newId = await onCreateNovelSession(novel.manifest.title);
      await moveNovel(novelId, newId);
      setNovelSessionId(newId);
      novelId = newId;
      setNovel((current) => current
        ? { ...current, manifest: { ...current.manifest, id: newId } }
        : current);
      onActivateNovelSession(newId);
    }
    
    const append = !isRevision && novel.outline.length > 0;
    setIsLoading(true);
    setOutlineStream('');
    setThinkingStream('');
    setProposedOutline(null);
    setProposalIsRevision(isRevision);
    setStatusMessage({ text: 'Generating outline...', type: 'info' });

    try {
      const result = await proposeOutlineStream(
        novelId,
        5,
        append,
        thinkingMode,
        modelId,
        isRevision ? currentDraft : null,
        contextSize,
        (content) => setOutlineStream((prev) => prev + content),
        (thinking) => setThinkingStream((prev) => prev + thinking),
        (message) => setStatusMessage({ text: message, type: 'info' }),
        () => setOutlineStream(''),
      );

      if ('error' in result) {
        setStatusMessage({ text: result.error, type: 'error' });
      } else if (result.proposed.length === 0) {
        setStatusMessage({ text: 'Empty outline returned', type: 'error' });
      } else {
        setProposedOutline(result.proposed);
        setOutlineDraft(
          isRevision || novel.outline.length === 0
            ? result.proposed
            : [...novel.outline, ...result.proposed],
        );
        setOutlineBaseline(
          isRevision || novel.outline.length === 0
            ? result.proposed
            : [...novel.outline, ...result.proposed],
        );
        setStatusMessage({ text: `${result.proposed.length} chapters proposed`, type: 'success' });
      }
    } catch (err: any) {
      setStatusMessage({ text: err.message, type: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  // ── Accept outline ──
  const handleAcceptOutline = async () => {
    if (!novel || !proposedOutline) return;
    setIsLoading(true);
    try {
      const newOutline = proposalIsRevision
        ? proposedOutline
        : [...novel.outline, ...proposedOutline];
      await saveOutline(novel.manifest.id, newOutline, proposalIsRevision);
      await loadNovel();
      setOutlineDraft(newOutline);
      setOutlineBaseline(newOutline);
      if (proposalIsRevision) setOutlineRevisionUsed(true);
      setProposedOutline(null);
      setProposalIsRevision(false);
      setEditingOutline(false);
      setStatusMessage(null);
      // Notify parent that outline is ready
      const firstUnwritten = newOutline.find(ch => ch.status === 'planned');
      onOutlineReady(true, firstUnwritten);
    } catch (err: any) {
      setStatusMessage({ text: err.message, type: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveOutlineEdits = async () => {
    if (!novel || !outlineDraft) return;
    setIsLoading(true);
    try {
      await saveOutline(novel.manifest.id, outlineDraft);
      await loadNovel();
      setOutlineDraft(outlineDraft);
      setOutlineBaseline(outlineDraft);
      setEditingOutline(false);
      setStatusMessage({ text: 'Outline edits saved.', type: 'success' });
    } catch (err: any) {
      setStatusMessage({ text: err.message, type: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const updateOutlineChapter = (number: number, updates: Partial<ChapterOutline>) => {
    setOutlineDraft((current) => {
      const source = current ?? novel?.outline ?? [];
      return source.map((chapter) => chapter.number === number ? { ...chapter, ...updates } : chapter);
    });
  };

  const discardOutlineProposal = async () => {
    if (!novel) return;
    try {
      await clearDraftOutline(novel.manifest.id);
      setProposedOutline(null);
      setProposalIsRevision(false);
      setOutlineBaseline(novel.outline);
      setOutlineDraft(novel.outline);
      setStatusMessage(null);
    } catch (err: any) {
      setStatusMessage({ text: err.message, type: 'error' });
    }
  };

  const displayedOutline = outlineDraft ?? novel?.outline ?? [];
  const hasOutlineEdits = novel
    ? JSON.stringify(displayedOutline) !== JSON.stringify(outlineBaseline ?? novel.outline)
    : false;
  const canReviseOutline = Boolean(
    novel?.outline.length && hasOutlineEdits && !outlineRevisionUsed
  );
  
  // Notify parent when novel loads with existing outline
  useEffect(() => {
    if (novel && novel.outline.length > 0) {
      const firstUnwritten = novel.outline.find(ch => ch.status === 'planned');
      onOutlineReady(true, firstUnwritten);
    } else {
      onOutlineReady(false);
    }
  }, [novel, onOutlineReady]);

  // ── Draft chapter ──
  const [currentDraftChapter, setCurrentDraftChapter] = useState<ChapterOutline | null>(null);

  const handleDraftChapter = (ch: ChapterOutline) => {
    if (!novel) return;
    setCurrentDraftChapter(ch);
    onDraftChapter(novel.manifest.id, ch.number, ch.title);
  };

  // ── View / edit chapter text ──
  const handleViewChapter = async (ch: ChapterOutline) => {
    if (!novel) return;
    setViewingChapter(ch.number);
    setChapterText('');
    setChapterTextDirty(false);
    setChapterTextLoading(true);
    const text = await fetchChapterText(novel.manifest.id, ch.number);
    setChapterText(text ?? '');
    setChapterTextLoading(false);
  };

  const handleSaveChapterText = async () => {
    if (!novel || viewingChapter === null) return;
    setChapterTextLoading(true);
    const ok = await saveChapterTextApi(novel.manifest.id, viewingChapter, chapterText);
    setChapterTextLoading(false);
    if (ok) {
      setChapterTextDirty(false);
      setStatusMessage({ text: `Chapter ${viewingChapter} text saved to file.`, type: 'success' });
      await loadNovel();
    } else {
      setStatusMessage({ text: 'Failed to save chapter text.', type: 'error' });
    }
  };

  // Steering directives state
  const [newDirective, setNewDirective] = useState('');

  const handleAddDirective = async () => {
    if (!novel || !newDirective.trim()) return;
    const text = newDirective.trim();
    setNewDirective('');
    try {
      // Save the steering directive first so it's in bible.json
      const res = await fetch(`/api/novels/${novel.manifest.id}/steering`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directive: text }),
      });
      if (res.ok) {
        await loadNovel();
        // If AI is currently generating, fully interrupt: stop generation, unload model,
        // then restart the same chapter draft so the model reloads with the new
        // steering directive injected into the context.
        if (isRunning && stop && currentDraftChapter) {
          setStatusMessage({ text: `Steering directive saved. Unloading model and restarting Chapter ${currentDraftChapter.number} with your correction...`, type: 'info' });
          // 1. Abort the current SSE stream / generation
          stop(sessionId);
          // 2. Unload the model from Ollama memory so it starts fresh
          if (killModel && inferenceBackend) {
            try { await killModel(modelId, inferenceBackend); } catch {}
          }
          // 3. Re-trigger the chapter draft — model reloads automatically with
          //    updated steering directives now in bible.json
          setTimeout(() => {
            onDraftChapter(novel.manifest.id, currentDraftChapter.number, currentDraftChapter.title);
          }, 500);
        } else {
          setStatusMessage({ text: 'Author steering directive saved. AI will strictly obey this in upcoming chapters.', type: 'info' });
        }
      }
    } catch {}
  };

  const handleRemoveDirective = async (index: number) => {
    if (!novel) return;
    const currentNotes = novel.bible.steering_notes ?? [];
    const updated = currentNotes.filter((_: string, i: number) => i !== index);
    try {
      const res = await fetch(`/api/novels/${novel.manifest.id}/steering`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: updated }),
      });
      if (res.ok) {
        await loadNovel();
      }
    } catch {}
  };

  // ── Bible ──
  const startEditBible = () => {
    if (!novel) return;
    setBibleDraft({ ...novel.bible });
    setEditingBible(true);
  };

  const handleSaveBible = async () => {
    if (!novel || !bibleDraft) return;
    setIsLoading(true);
    try {
      await saveBible(novel.manifest.id, bibleDraft);
      await loadNovel();
      setEditingBible(false);
      setBibleDraft(null);
    } catch {}
    setIsLoading(false);
  };

  // ── Download ──
  const handleDownload = async () => {
    if (!novel) return;
    setStatusMessage({ text: 'Preparing download...', type: 'info' });

    try {
      let content = `${novel.manifest.title}\n${'='.repeat(novel.manifest.title.length)}\n\n`;
      content += `${novel.manifest.premise}\n\n---\n\n`;

      for (const ch of novel.outline) {
        const text = await fetchChapterText(novel.manifest.id, ch.number);
        if (text) {
          content += `Chapter ${ch.number}: ${ch.title}\n\n${text}\n\n---\n\n`;
        }
      }

      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${novel.manifest.title.replace(/[^a-z0-9]/gi, '_')}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setStatusMessage({ text: 'Downloaded!', type: 'success' });
    } catch (err: any) {
      setStatusMessage({ text: `Download failed: ${err.message}`, type: 'error' });
    }
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Styles
  // ─────────────────────────────────────────────────────────────────────────────

  const containerStyle: React.CSSProperties = {
    flex: 1,
    minHeight: 0,
    overflowY: 'auto',
    background: 'var(--bg-base)',
    padding: '12px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    fontSize: 12,
  };

  const cardStyle: React.CSSProperties = {
    background: 'var(--bg-surface)',
    border: '1px solid var(--border)',
    borderRadius: 6,
    padding: 10,
  };

  const btnStyle: React.CSSProperties = {
    padding: '6px 12px',
    background: 'var(--accent)',
    color: 'white',
    border: 'none',
    borderRadius: 5,
    cursor: 'pointer',
    fontSize: 11,
    fontWeight: 600,
  };

  const btnSecondaryStyle: React.CSSProperties = {
    ...btnStyle,
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
  };

  const btnSmallStyle: React.CSSProperties = {
    padding: '4px 8px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 10,
    color: 'var(--text-secondary)',
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Render: Setup form (no novel yet)
  // ─────────────────────────────────────────────────────────────────────────────

  if (!novel) {
    return (
      <div style={containerStyle}>
        <div style={{ color: 'var(--accent-light)', fontSize: 11, fontWeight: 700, letterSpacing: '0.05em' }}>
          📖 NOVEL MODE
        </div>

        <div style={cardStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Start Your Novel</div>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Novel title"
            disabled={isLoading}
            style={{ width: '100%', marginBottom: 6, padding: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 12 }}
          />
          <textarea
            value={premise}
            onChange={(e) => setPremise(e.target.value)}
            placeholder="Premise: genre, tone, setting, main characters, plot overview..."
            rows={4}
            disabled={isLoading}
            style={{ width: '100%', marginBottom: 8, padding: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical', lineHeight: 1.5 }}
          />
          
          {/* Length settings */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Pages</div>
              <input
                type="number"
                value={targetPages}
                onChange={(e) => setTargetPages(Math.max(10, parseInt(e.target.value) || 100))}
                disabled={isLoading}
                min={10}
                max={1000}
                style={{ width: '100%', padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
              />
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Words/page</div>
              <input
                type="number"
                value={wordsPerPage}
                onChange={(e) => setWordsPerPage(Math.max(100, parseInt(e.target.value) || 250))}
                disabled={isLoading}
                min={100}
                max={500}
                style={{ width: '100%', padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
              />
            </div>
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Chapters</div>
              <input
                type="number"
                value={targetChapters}
                onChange={(e) => setTargetChapters(Math.max(1, parseInt(e.target.value) || 20))}
                disabled={isLoading}
                min={1}
                max={100}
                style={{ width: '100%', padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
              />
            </div>
          </div>

          {/* Calculated totals */}
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 12, padding: '6px 8px', background: 'var(--bg-base)', borderRadius: 4 }}>
            Target: ~{(targetPages * wordsPerPage).toLocaleString()} words ({Math.round(targetPages * wordsPerPage / targetChapters).toLocaleString()} words/chapter, ~{Math.round(targetPages * wordsPerPage * 1.3 / 1000)}k tokens)
          </div>

          {/* Story Bible Setup */}
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10, marginBottom: 12 }}>
            <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 8, color: 'var(--accent-light)' }}>
              📚 Story Bible Setup
            </div>

            {/* POV Selector */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Point of View (POV)</div>
              <select
                value={povMode}
                onChange={(e) => setPovMode(e.target.value as any)}
                disabled={isLoading}
                style={{ width: '100%', padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
              >
                <option value="Auto - variable">Auto - variable (POV shifts per chapter as needed)</option>
                <option value="Auto - single character">Auto - single character (Focused third-person limited)</option>
                <option value="External narrator">External narrator (Third-person omniscient / distant)</option>
                <option value="User defined">User defined (Supply custom POV style below)</option>
              </select>
              {povMode === 'User defined' && (
                <input
                  type="text"
                  value={customPov}
                  onChange={(e) => setCustomPov(e.target.value)}
                  placeholder="e.g. First-person limited (Sarah Connor), past tense"
                  disabled={isLoading}
                  style={{ width: '100%', marginTop: 6, padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
                />
              )}
            </div>

            {/* Character Creator */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)' }}>Main Characters (Starting State)</span>
                <button
                  type="button"
                  style={btnSmallStyle}
                  disabled={isLoading}
                  onClick={() => setFormCharacters(prev => [...prev, { name: '', description: '', relationships: '' }])}
                >
                  + Add Character
                </button>
              </div>

              {/* Character Library Dropdown */}
              {pastCharacters.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Import from Past Characters</div>
                  <select
                    value=""
                    disabled={isLoading}
                    onChange={(e) => {
                      if (!e.target.value) return;
                      const found = pastCharacters.find(c => c.name === e.target.value);
                      if (found) {
                        setFormCharacters(prev => [...prev, {
                          name: found.name,
                          description: found.description,
                          relationships: found.relationships.join(', '),
                        }]);
                      }
                    }}
                    style={{ width: '100%', padding: 5, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
                  >
                    <option value="">— Select a past character to import —</option>
                    {pastCharacters.map(c => (
                      <option key={c.name} value={c.name} title={c.description}>
                        {c.name} (from: {c.source_novel})
                      </option>
                    ))}
                  </select>
                  {/* Hover preview list */}
                  <div style={{ maxHeight: 100, overflowY: 'auto', marginTop: 4, border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg-base)' }}>
                    {pastCharacters.map(c => (
                      <div
                        key={c.name}
                        style={{ padding: '4px 8px', cursor: 'pointer', borderBottom: '1px solid var(--border)', fontSize: 10 }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'var(--bg-hover)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent';
                        }}
                        onClick={() => {
                          setFormCharacters(prev => [...prev, {
                            name: c.name,
                            description: c.description,
                            relationships: c.relationships.join(', '),
                          }]);
                        }}
                        title={c.description}
                      >
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{c.name}</span>
                        <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>from: {c.source_novel}</span>
                        <div style={{ color: 'var(--text-muted)', fontSize: 9, marginTop: 2, lineHeight: 1.3 }}>
                          {c.description.slice(0, 120)}{c.description.length > 120 ? '...' : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {formCharacters.length === 0 ? (
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontStyle: 'italic', marginBottom: 4 }}>
                  No main characters added yet. Click + Add Character or import from past novels above.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {formCharacters.map((char, index) => (
                    <div key={index} style={{ background: 'var(--bg-base)', border: '1px solid var(--border)', borderRadius: 5, padding: 8 }}>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
                        <input
                          type="text"
                          value={char.name}
                          onChange={(e) => {
                            const updated = [...formCharacters];
                            updated[index].name = e.target.value;
                            setFormCharacters(updated);
                          }}
                          placeholder="Character Name"
                          disabled={isLoading}
                          style={{ flex: 1, padding: 5, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, fontWeight: 600 }}
                        />
                        <button
                          type="button"
                          disabled={isLoading}
                          onClick={() => setFormCharacters(prev => prev.filter((_, i) => i !== index))}
                          title="Remove character"
                          style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 14, padding: '0 4px' }}
                        >
                          ×
                        </button>
                      </div>
                      <textarea
                        value={char.description}
                        onChange={(e) => {
                          const updated = [...formCharacters];
                          updated[index].description = e.target.value;
                          setFormCharacters(updated);
                        }}
                        placeholder="Starting description (appearance, background, initial role or state before story actions change them)..."
                        rows={2}
                        disabled={isLoading}
                        style={{ width: '100%', marginBottom: 6, padding: 5, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical' }}
                      />
                      <input
                        type="text"
                        value={char.relationships}
                        onChange={(e) => {
                          const updated = [...formCharacters];
                          updated[index].relationships = e.target.value;
                          setFormCharacters(updated);
                        }}
                        placeholder="Relationships (e.g. Sister of Marcus, rival of Dr. Chen)"
                        disabled={isLoading}
                        style={{ width: '100%', padding: 5, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 10 }}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Writing Style Notes */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Writing Style & Tone Notes</div>
              <textarea
                value={styleNotes}
                onChange={(e) => setStyleNotes(e.target.value)}
                placeholder="e.g. Fast-paced thriller, vivid sensory details, sharp dialogue, tense atmospheric descriptions..."
                rows={2}
                disabled={isLoading}
                style={{ width: '100%', padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical' }}
              />
            </div>

            {/* Key World Facts */}
            <div style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Established World Facts / Rules</span>
                <button
                  type="button"
                  style={btnSmallStyle}
                  disabled={isLoading}
                  onClick={() => setFormFacts(prev => [...prev, ''])}
                >
                  + Add Fact
                </button>
              </div>
              {formFacts.map((fact, index) => (
                <div key={index} style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
                  <input
                    type="text"
                    value={fact}
                    onChange={(e) => {
                      const updated = [...formFacts];
                      updated[index] = e.target.value;
                      setFormFacts(updated);
                    }}
                    placeholder="e.g. Magic requires physical stamina; hyperdrive jumps take 2 hours to compute"
                    disabled={isLoading}
                    style={{ flex: 1, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
                  />
                  <button
                    type="button"
                    disabled={isLoading}
                    onClick={() => setFormFacts(prev => prev.filter((_, i) => i !== index))}
                    style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 14 }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>

            {/* Key Locations */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Key Locations</span>
                <button
                  type="button"
                  style={btnSmallStyle}
                  disabled={isLoading}
                  onClick={() => setFormLocations(prev => [...prev, { name: '', description: '' }])}
                >
                  + Add Location
                </button>
              </div>
              {formLocations.map((loc, index) => (
                <div key={index} style={{ background: 'var(--bg-base)', border: '1px solid var(--border)', borderRadius: 4, padding: 6, marginBottom: 4 }}>
                  <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
                    <input
                      type="text"
                      value={loc.name}
                      onChange={(e) => {
                        const updated = [...formLocations];
                        updated[index].name = e.target.value;
                        setFormLocations(updated);
                      }}
                      placeholder="Location Name"
                      disabled={isLoading}
                      style={{ flex: 1, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, fontWeight: 600 }}
                    />
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => setFormLocations(prev => prev.filter((_, i) => i !== index))}
                      style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 14 }}
                    >
                      ×
                    </button>
                  </div>
                  <input
                    type="text"
                    value={loc.description}
                    onChange={(e) => {
                      const updated = [...formLocations];
                      updated[index].description = e.target.value;
                      setFormLocations(updated);
                    }}
                    placeholder="Short description of location..."
                    disabled={isLoading}
                    style={{ width: '100%', padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 10 }}
                  />
                </div>
              ))}
            </div>
          </div>
          
          <button style={btnStyle} onClick={handleCreate} disabled={isLoading || !title.trim() || !premise.trim()}>
            {isLoading ? 'Creating...' : 'Create Novel'}
          </button>
        </div>

        {statusMessage && (
          <div style={{
            padding: '8px 10px',
            borderRadius: 5,
            fontSize: 11,
            background: statusMessage.type === 'error' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(59, 130, 246, 0.15)',
            color: statusMessage.type === 'error' ? '#ef4444' : '#3b82f6',
          }}>
            {statusMessage.text}
          </div>
        )}
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Render: Novel exists
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div>
        <div style={{ color: 'var(--accent-light)', fontSize: 11, fontWeight: 700, letterSpacing: '0.05em' }}>
          📖 NOVEL MODE
        </div>
        <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>
          {novel.manifest.title}
        </div>
      </div>

      {/* Premise — full, copy-pasteable */}
      <div style={{ ...cardStyle, padding: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Premise</span>
          <button
            type="button"
            onClick={() => {
              navigator.clipboard?.writeText(novel.manifest.premise).catch(() => {});
            }}
            title="Copy premise to clipboard"
            style={{ ...btnSmallStyle, fontSize: 9 }}
          >
            ⧉ Copy
          </button>
        </div>
        <textarea
          value={novel.manifest.premise}
          readOnly
          rows={4}
          style={{ width: '100%', padding: 6, background: 'var(--bg-base)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical', lineHeight: 1.5, userSelect: 'text' }}
        />
      </div>

      {/* Status */}
      {statusMessage && (
        <div style={{
          padding: '8px 10px',
          borderRadius: 5,
          fontSize: 11,
          background: statusMessage.type === 'error' ? 'rgba(239, 68, 68, 0.15)' :
                      statusMessage.type === 'success' ? 'rgba(34, 197, 94, 0.15)' :
                      'rgba(59, 130, 246, 0.15)',
          color: statusMessage.type === 'error' ? '#ef4444' :
                 statusMessage.type === 'success' ? '#22c55e' : '#3b82f6',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>{statusMessage.text}</span>
          <button onClick={() => setStatusMessage(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>✕</button>
        </div>
      )}

      {/* Thinking stream, styled like main chat */}
      {thinkingStream && (
        <div style={{
          padding: '8px 10px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(124,106,247,0.08)',
          border: '1px solid rgba(124,106,247,0.2)',
          color: 'var(--text-secondary)',
          fontSize: 12,
          lineHeight: 1.5,
        }}>
          <div style={{
            marginBottom: 5,
            color: 'var(--accent-light)',
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}>
            Thinking process
          </div>
          <pre style={{
            margin: 0,
            whiteSpace: 'pre-wrap',
            overflowWrap: 'anywhere',
            font: 'inherit',
          }}>
            {thinkingStream}
          </pre>
        </div>
      )}

      {/* Streaming outline progress */}
      {outlineStream && (
        <div style={{
          padding: 10,
          borderRadius: 6,
          fontSize: 10,
          fontFamily: 'monospace',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          maxHeight: 120,
          overflowY: 'auto',
          whiteSpace: 'pre-wrap',
          color: 'var(--text-secondary)',
        }}>
          {outlineStream.slice(-600)}▌
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <button
          style={btnStyle}
          onClick={handleProposeOutline}
          disabled={isLoading || (hasOutlineEdits && outlineRevisionUsed)}
        >
          {isLoading
            ? '⏳...'
            : canReviseOutline
              ? 'Revise outline with AI (last pass)'
              : novel.outline.length === 0 ? 'Generate Outline' : '+ More Chapters'}
        </button>
        {displayedOutline.length > 0 && (
          <button
            style={btnSecondaryStyle}
            onClick={() => setEditingOutline((current) => !current)}
            disabled={isLoading}
          >
            {editingOutline ? 'Stop Editing' : 'Edit Outline'}
          </button>
        )}
        {editingOutline && hasOutlineEdits && (
          <button style={btnSecondaryStyle} onClick={handleSaveOutlineEdits} disabled={isLoading}>
            Save edits
          </button>
        )}
        <button style={btnSecondaryStyle} onClick={startEditBible} disabled={isLoading}>
          Bible
        </button>
        <button style={btnSecondaryStyle} onClick={handleDownload} disabled={isLoading || novel.chapters.length === 0}>
          ⬇ .txt
        </button>
      </div>

      {/* Proposed outline */}
      {proposedOutline && (
        <div style={{ ...cardStyle, borderColor: 'var(--accent)' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--accent-light)', fontSize: 11 }}>
            {proposalIsRevision ? 'Revised Outline Proposal' : 'Proposed Chapters'}
          </div>
          {proposedOutline.map((ch) => (
            <div key={ch.number} style={{ marginBottom: 6, padding: 6, background: 'var(--bg-card)', borderRadius: 4 }}>
              <div style={{ fontWeight: 600, fontSize: 11 }}>Ch {ch.number}: {ch.title}</div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                {ch.description || ch.beat_summary}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10, marginTop: 3 }}>
                Characters: {(ch.characters_involved ?? []).join(', ') || 'None specified'} · {ch.target_pages ?? '?'} pages
              </div>
            </div>
          ))}
          <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
            <button style={btnStyle} onClick={handleAcceptOutline} disabled={isLoading}>Accept</button>
            <button style={btnSecondaryStyle} onClick={discardOutlineProposal}>Discard</button>
          </div>
        </div>
      )}

      {/* Author Steering & Course Correction directives */}
      <div style={{ ...cardStyle, background: 'rgba(59, 130, 246, 0.05)', borderColor: 'rgba(59, 130, 246, 0.25)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontWeight: 600, fontSize: 11, color: '#3b82f6', display: 'flex', alignItems: 'center', gap: 4 }}>
            ⚡ Author Steering & Course Correction
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Persists in AI memory</span>
        </div>

        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6 }}>
          Add mandatory author overrides or course corrections if the AI strays from the outline (e.g. "Do NOT include Marcus in Chapter 1" or "Sarah stays at the diner").
        </div>

        {(novel.bible.steering_notes ?? []).length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
            {(novel.bible.steering_notes ?? []).map((note: string, index: number) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-card)', border: '1px solid var(--border)', padding: '4px 8px', borderRadius: 4, fontSize: 11 }}>
                <span style={{ color: 'var(--text-primary)' }}>• {note}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveDirective(index)}
                  style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 14, padding: '0 4px' }}
                  title="Delete steering directive"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 6 }}>
          <input
            type="text"
            value={newDirective}
            onChange={(e) => setNewDirective(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleAddDirective(); }}
            placeholder="e.g. Do NOT introduce Marcus until Chapter 4..."
            style={{ flex: 1, padding: 5, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
          />
          <button style={btnSmallStyle} onClick={handleAddDirective} disabled={!newDirective.trim()}>
            + Steer AI
          </button>
        </div>
      </div>

      {/* Chapter list */}
      <div style={{ fontWeight: 600, fontSize: 11, color: 'var(--text-secondary)' }}>
        Chapters ({displayedOutline.length})
      </div>
      {displayedOutline.length === 0 ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'center', padding: 12 }}>
          Generate an outline to start drafting.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {displayedOutline.map((ch) => {
            const meta = novel.chapters.find((c) => c.number === ch.number);
            const isDrafted = ch.status === 'drafted' || ch.status === 'revised';
            return (
              <div
                key={ch.number}
                style={{
                  ...cardStyle,
                  padding: 8,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  opacity: isDrafted ? 0.7 : 1,
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  {editingOutline ? (
                    <input
                      value={ch.title}
                      onChange={(event) => updateOutlineChapter(ch.number, { title: event.target.value })}
                      style={{ width: '100%', padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-primary)', fontSize: 11, fontWeight: 600 }}
                    />
                  ) : (
                    <div style={{ fontWeight: 600, fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ color: isDrafted ? '#22c55e' : 'var(--text-muted)' }}>
                      {isDrafted ? '✓' : '○'}
                    </span>
                    Ch {ch.number}: {ch.title}
                    </div>
                  )}
                  {editingOutline && (
                    <>
                      <textarea
                        value={ch.description || ch.beat_summary}
                        onChange={(event) => updateOutlineChapter(ch.number, {
                          description: event.target.value,
                          beat_summary: event.target.value,
                        })}
                        rows={3}
                        placeholder="2-3 sentence chapter description"
                        style={{ width: '100%', marginTop: 4, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-primary)', fontSize: 10, resize: 'vertical' }}
                      />
                      <input
                        value={(ch.characters_involved ?? []).join(', ')}
                        onChange={(event) => updateOutlineChapter(ch.number, {
                          characters_involved: event.target.value.split(',').map((name) => name.trim()).filter(Boolean),
                        })}
                        placeholder="Characters involved, comma separated"
                        style={{ width: '100%', marginTop: 4, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-primary)', fontSize: 10 }}
                      />
                      <input
                        value={ch.pov_character}
                        onChange={(event) => updateOutlineChapter(ch.number, { pov_character: event.target.value })}
                        placeholder="POV character"
                        style={{ width: '100%', marginTop: 4, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-primary)', fontSize: 10 }}
                      />
                      <div style={{ display: 'flex', gap: 4 }}>
                        <input
                          type="number"
                          min={1}
                          value={ch.target_pages ?? ''}
                          onChange={(event) => updateOutlineChapter(ch.number, { target_pages: Number(event.target.value) || undefined })}
                          placeholder="Target pages"
                          style={{ width: 100, marginTop: 4, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-primary)', fontSize: 10 }}
                        />
                        <input
                          type="number"
                          min={1}
                          value={ch.target_words ?? ''}
                          onChange={(event) => updateOutlineChapter(ch.number, { target_words: Number(event.target.value) || undefined })}
                          placeholder="Target words"
                          style={{ width: 110, marginTop: 4, padding: 4, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-primary)', fontSize: 10 }}
                        />
                      </div>
                    </>
                  )}
                  {meta && (
                    <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>
                      {meta.wordCount.toLocaleString()} words
                    </div>
                  )}
                  {!editingOutline && (
                    <div style={{ color: 'var(--text-muted)', fontSize: 10, lineHeight: 1.35 }}>
                      {ch.description || ch.beat_summary}
                      <br />
                      Characters: {(ch.characters_involved ?? []).join(', ') || 'None specified'} · {ch.target_pages ?? '?'} pages · {ch.target_words ?? '?'} words
                    </div>
                  )}
                </div>
                <button style={btnSmallStyle} onClick={() => handleDraftChapter(ch)} title="Draft in chat">
                  {isDrafted ? '↻' : '▶'}
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* Bible editor modal */}
      {editingBible && bibleDraft && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ ...cardStyle, width: '90%', maxWidth: 520, maxHeight: '80vh', overflow: 'auto' }}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Story Bible & Author Directives</div>

            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4 }}>POV Style</div>
              <input
                type="text"
                value={bibleDraft.pov ?? ''}
                onChange={(e) => setBibleDraft({ ...bibleDraft, pov: e.target.value })}
                style={{ width: '100%', padding: 6, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11 }}
                placeholder="e.g. Auto - variable, or First person limited (Sarah Connor)"
              />
            </div>

            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4, color: '#3b82f6' }}>⚡ Author Mandatory Directives & Course Corrections (one per line)</div>
              <textarea
                value={(bibleDraft.steering_notes ?? []).join('\n')}
                onChange={(e) => setBibleDraft({ ...bibleDraft, steering_notes: e.target.value.split('\n').filter(Boolean) })}
                rows={4}
                style={{ width: '100%', padding: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical' }}
                placeholder="e.g. Do NOT include Marcus in Chapter 1&#10;Sarah stays at the diner&#10;Write in dark moody thriller style"
              />
            </div>

            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Style & Tone Notes</div>
              <textarea
                value={bibleDraft.style_notes}
                onChange={(e) => setBibleDraft({ ...bibleDraft, style_notes: e.target.value })}
                rows={2}
                style={{ width: '100%', padding: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical' }}
                placeholder="Voice, tone, prose guidelines..."
              />
            </div>

            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Facts (one per line)</div>
              <textarea
                value={bibleDraft.facts.join('\n')}
                onChange={(e) => setBibleDraft({ ...bibleDraft, facts: e.target.value.split('\n').filter(Boolean) })}
                rows={3}
                style={{ width: '100%', padding: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 11, resize: 'vertical' }}
                placeholder="World rules, timeline, character facts..."
              />
            </div>

            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button style={btnStyle} onClick={handleSaveBible} disabled={isLoading}>Save Bible</button>
              <button style={btnSecondaryStyle} onClick={() => { setEditingBible(false); setBibleDraft(null); }}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* Chapter text viewer/editor modal */}
      {viewingChapter !== null && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
          <div style={{ ...cardStyle, width: '90%', maxWidth: 700, maxHeight: '85vh', overflow: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <div style={{ fontSize: 14, fontWeight: 700 }}>Chapter {viewingChapter} Text</div>
              <div style={{ fontSize: 10, color: chapterTextDirty ? '#f59e0b' : 'var(--text-muted)' }}>
                {chapterTextDirty ? '● Unsaved changes' : chapterTextLoading ? 'Loading...' : `${chapterText.split(/\s+/).filter(Boolean).length.toLocaleString()} words`}
              </div>
            </div>
            <textarea
              value={chapterText}
              onChange={(e) => { setChapterText(e.target.value); setChapterTextDirty(true); }}
              rows={20}
              disabled={chapterTextLoading}
              placeholder={chapterTextLoading ? 'Loading chapter text...' : 'No text yet. Draft this chapter or paste/edit text here, then Save.'}
              style={{ width: '100%', padding: 8, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)', fontSize: 12, resize: 'vertical', fontFamily: 'Georgia, serif', lineHeight: 1.6 }}
            />
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button style={btnStyle} onClick={handleSaveChapterText} disabled={chapterTextLoading || !chapterTextDirty}>
                Save to File
              </button>
              <button style={btnSecondaryStyle} onClick={() => { setViewingChapter(null); setChapterText(''); setChapterTextDirty(false); }}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
