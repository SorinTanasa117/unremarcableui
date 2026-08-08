import React, { useEffect, useRef, useState } from 'react';
import type { AgentStatus, ChatMessage } from '../hooks/useOllamaStream';

interface StoryDraft {
  title: string;
  premise: string;
  memory: string;
  manuscript: string;
  nextBeat: string;
  paragraphs: number;
  uncompactedText: string;
  roundsSinceCompaction: number;
  storyFilePath: string;
}

interface Props {
  sessionId: string;
  messages: ChatMessage[];
  status: AgentStatus;
  modelAvailable: boolean;
  onGenerate: (prompt: string) => void;
}

const EMPTY_DRAFT: StoryDraft = {
  title: 'Untitled story',
  premise: '',
  memory: '',
  manuscript: '',
  nextBeat: '',
  paragraphs: 2,
  uncompactedText: '',
  roundsSinceCompaction: 0,
  storyFilePath: '',
};

// Local models spend most of their time ingesting context, so Story Mode keeps
// a deliberately small continuity packet for each isolated continuation.
const RECENT_PROSE_CHARS = 12_000;
const LEDGER_CHARS = 6_000;
const COMPACTION_AFTER_ROUNDS = 4;

function storageKey(sessionId: string) {
  return `ollama_story_draft_${sessionId}`;
}

function loadDraft(sessionId: string): StoryDraft {
  try {
    const saved = localStorage.getItem(storageKey(sessionId));
    return saved ? { ...EMPTY_DRAFT, ...JSON.parse(saved) } : EMPTY_DRAFT;
  } catch {
    return EMPTY_DRAFT;
  }
}

function buildPrompt(draft: StoryDraft): string {
  const recentManuscript = draft.uncompactedText.slice(-RECENT_PROSE_CHARS) || draft.manuscript.slice(-RECENT_PROSE_CHARS);
  const sections = [
    draft.title.trim() && draft.title.trim() !== EMPTY_DRAFT.title && `Story title: ${draft.title.trim()}`,
    draft.premise.trim() && `Premise and style: ${draft.premise.trim()}`,
    draft.memory.trim() && `Story ledger (established facts only):\n${draft.memory.trim().slice(-LEDGER_CHARS)}`,
    draft.nextBeat.trim() && `What should happen next: ${draft.nextBeat.trim()}`,
    recentManuscript.trim() && `Recent manuscript:\n${recentManuscript.trim()}`,
  ].filter(Boolean);
  return `[Story Studio — generate prose only]
Write exactly ${draft.paragraphs} polished story paragraph${draft.paragraphs === 1 ? '' : 's'} that continue the manuscript. Do not add a title, outline, explanation, Markdown, or commentary. Preserve established facts, voice, tense, and point of view.

${sections.join('\n\n') || 'Begin the opening scene naturally.'}`;
}

function buildCompactionPrompt(draft: StoryDraft): string {
  return `[Story Studio — compact the story ledger]
Create a concise factual story ledger from the existing ledger and the new prose below. Keep only durable continuity information: named characters and their current state, relationships, setting/world rules, established events in order, unresolved threads, important objects, and the exact current scene. Do not write fiction, commentary, or Markdown headings. Use compact bullet points and stay under 300 words.

Existing ledger:
${draft.memory.trim().slice(-LEDGER_CHARS) || '(none)'}

New prose to compact:
${draft.uncompactedText.slice(-RECENT_PROSE_CHARS)}`;
}

function storyFilePath(draft: StoryDraft, sessionId: string): string {
  if (draft.storyFilePath) return draft.storyFilePath;
  const slug = (draft.title || 'untitled-story')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60) || 'untitled-story';
  return `stories/${slug}-${sessionId}.md`;
}

async function saveStoryFile(path: string, draft: StoryDraft): Promise<void> {
  const content = `# ${draft.title}\n\n## Story ledger\n\n${draft.memory || '(No ledger has been generated yet.)'}\n\n## Manuscript\n\n${draft.manuscript}`;
  const response = await fetch('/api/files/write', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, content }),
  });
  if (!response.ok) throw new Error('Unable to save the story file');
}

export function StoryStudio({ sessionId, messages, status, modelAvailable, onGenerate }: Props) {
  const [draft, setDraft] = useState<StoryDraft>(() => loadDraft(sessionId));
  const [hydratedSessionId, setHydratedSessionId] = useState(sessionId);
  const [paneHeights, setPaneHeights] = useState({ manuscript: 300, memory: 300 });
  const draftRef = useRef(draft);
  const pendingRequest = useRef<'story' | 'compact' | null>(null);
  const assistantIdsAtRequest = useRef(new Set<string>());
  const isRunning = status === 'thinking' || status === 'tool';

  useEffect(() => {
    const loaded = loadDraft(sessionId);
    setDraft(loaded);
    pendingRequest.current = null;
    setHydratedSessionId(sessionId);
  }, [sessionId]);

  useEffect(() => {
    if (hydratedSessionId !== sessionId) return;
    localStorage.setItem(storageKey(sessionId), JSON.stringify(draft));
  }, [draft, hydratedSessionId, sessionId]);

  useEffect(() => {
    draftRef.current = draft;
  }, [draft]);

  useEffect(() => {
    if (!pendingRequest.current || isRunning) return;
    const response = [...messages].reverse().find((message) =>
      message.role === 'assistant' &&
      !message.isActivity &&
      !assistantIdsAtRequest.current.has(message.id) &&
      message.content.trim()
    );
    if (!response) return;

    if (pendingRequest.current === 'story') {
      pendingRequest.current = null;
      const current = draftRef.current;
      const next = {
        ...current,
        manuscript: [current.manuscript.trim(), response.content.trim()].filter(Boolean).join('\n\n'),
        uncompactedText: [current.uncompactedText.trim(), response.content.trim()].filter(Boolean).join('\n\n'),
        roundsSinceCompaction: current.roundsSinceCompaction + 1,
        nextBeat: '',
      };
      draftRef.current = next;
      setDraft(next);

      if (next.roundsSinceCompaction >= COMPACTION_AFTER_ROUNDS || next.uncompactedText.length >= RECENT_PROSE_CHARS) {
        pendingRequest.current = 'compact';
        assistantIdsAtRequest.current = new Set(
          messages.filter((message) => message.role === 'assistant').map((message) => message.id),
        );
        onGenerate(buildCompactionPrompt(next));
      }
      return;
    }

    pendingRequest.current = null;
    const current = draftRef.current;
    const path = storyFilePath(current, sessionId);
    const next = {
      ...current,
      memory: response.content.trim(),
      uncompactedText: '',
      roundsSinceCompaction: 0,
      storyFilePath: path,
    };
    draftRef.current = next;
    setDraft(next);
    void saveStoryFile(path, next).catch(() => {});
  }, [isRunning, messages]);

  const update = <K extends keyof StoryDraft>(key: K, value: StoryDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };

  // Native textarea resizing is immediate; record the final size so a later
  // React render does not snap the pane back to its initial height.
  const savePaneHeight = (pane: 'manuscript' | 'memory', element: HTMLTextAreaElement) => {
    setPaneHeights((current) => ({ ...current, [pane]: element.offsetHeight }));
  };

  const generate = () => {
    if (isRunning || !modelAvailable) return;
    pendingRequest.current = 'story';
    assistantIdsAtRequest.current = new Set(
      messages.filter((message) => message.role === 'assistant').map((message) => message.id),
    );
    onGenerate(buildPrompt(draft));
  };

  return (
    <section style={{
      flex: 1, minHeight: 0, overflowY: 'auto', background: 'var(--bg-base)',
      padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14,
    }}>
      <div className="flex items-center justify-between">
        <div>
          <div style={{ color: 'var(--accent-light)', fontSize: 12, fontWeight: 700, letterSpacing: '0.05em' }}>✦ STORY MODE</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
            {draft.storyFilePath ? `Story saved to ${draft.storyFilePath}` : 'Focused drafting with compact continuity memory'}
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between" style={{ padding: '9px 12px', background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 7 }}>
        <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>Continuation length</span>
        <select
          value={draft.paragraphs}
          onChange={(event) => update('paragraphs', Number(event.target.value))}
          disabled={isRunning}
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', borderRadius: 5, padding: '5px 8px', fontSize: 11 }}
          title="Paragraphs to generate"
        >
          {Array.from({ length: 20 }, (_, index) => index + 1).map((count) => (
            <option key={count} value={count}>
              {count} paragraph{count === 1 ? '' : 's'}
            </option>
          ))}
        </select>
      </div>
      <div style={{ color: 'var(--text-secondary)', fontSize: 12, fontWeight: 600 }}>Manuscript</div>
      <textarea
        value={draft.manuscript}
        onChange={(event) => update('manuscript', event.target.value)}
        disabled={isRunning}
        aria-label="Story manuscript"
        placeholder="Write an opening, or let the agent begin the story. You can edit any passage."
        rows={10}
        onMouseUp={(event) => savePaneHeight('manuscript', event.currentTarget)}
        onTouchEnd={(event) => savePaneHeight('manuscript', event.currentTarget)}
        style={{ height: paneHeights.manuscript, minHeight: 160, maxHeight: 720, boxSizing: 'border-box', resize: 'vertical', overflow: 'auto', background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text-primary)', padding: '14px 16px', fontSize: 14, lineHeight: 1.7, fontFamily: 'inherit' }}
      />
      <div style={{ color: 'var(--text-secondary)', fontSize: 12, fontWeight: 600 }}>Continuity memory</div>
      <textarea
        value={draft.memory}
        onChange={(event) => update('memory', event.target.value)}
        disabled={isRunning}
        aria-label="Story memory"
        placeholder="Story memory: characters, setting, facts, and unresolved threads"
        rows={10}
        onMouseUp={(event) => savePaneHeight('memory', event.currentTarget)}
        onTouchEnd={(event) => savePaneHeight('memory', event.currentTarget)}
        style={{ height: paneHeights.memory, minHeight: 160, maxHeight: 720, boxSizing: 'border-box', resize: 'vertical', overflow: 'auto', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 7, color: 'var(--text-primary)', padding: '10px 12px', fontSize: 12, lineHeight: 1.5, fontFamily: 'inherit' }}
      />
      <div className="flex gap-2">
        <textarea
          value={draft.nextBeat}
          onChange={(event) => update('nextBeat', event.target.value)}
          disabled={isRunning}
          aria-label="What happens next"
          placeholder="What should happen next? (optional)"
          rows={2}
          style={{ flex: 1, resize: 'vertical', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 5, color: 'var(--text-primary)', padding: '7px 9px', fontSize: 12, fontFamily: 'inherit' }}
        />
        <button className="btn btn-primary" onClick={generate} disabled={isRunning || !modelAvailable} style={{ alignSelf: 'stretch', minWidth: 112 }}>
          {isRunning ? 'Writing…' : 'Continue story'}
        </button>
      </div>
    </section>
  );
}
