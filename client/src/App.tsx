import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { FileExplorer } from './components/FileExplorer';
import { FileEditor } from './components/FileEditor';
import { TerminalPane } from './components/TerminalPane';
import { ModelSelector } from './components/ModelSelector';
import { StatusIndicator } from './components/StatusIndicator';
import { ProgressBar } from './components/ProgressBar';
import { NovelStudio } from './components/NovelStudio';
import { useOllamaStream, type InferenceBackend } from './hooks/useOllamaStream';
import { formatTokenRate } from './lib/tokenCounter';
import { playChime, unlockAudio } from './lib/chime';

function generateSessionId() {
  return `session_${Math.random().toString(36).slice(2, 11)}`;
}

type RightTab = 'editor' | 'terminal' | 'files' | 'novel';
type PersonaType = 'coder' | 'researcher' | 'creative' | 'system' | 'novelist';
const CONTEXT_SIZES = [16_384, 32_768, 65_536, 131_072, 262_144];

const personaBadges: Record<PersonaType, { label: string; color: string }> = {
  coder: { label: 'Coder', color: 'var(--blue)' },
  researcher: { label: 'Researcher', color: 'var(--green)' },
  creative: { label: 'Creative', color: 'var(--amber)' },
  system: { label: 'System', color: 'var(--purple)' },
  novelist: { label: 'Novelist', color: 'var(--teal, #14b8a6)' },
};

export default function App() {
  const {
    messages,
    status,
    statusText,
    tokenUsage,
    tokenRates,
    sendMessage,
    stop,
    killModel,
    resetSession,
    loadSessionHistory,
    sessionsList,
    fetchSessionsList,
    deleteSession,
  } = useOllamaStream();

  const [model, setModel] = useState('');
  const [inferenceBackend, setInferenceBackend] = useState<InferenceBackend>(() =>
    localStorage.getItem('ollama_inference_backend') === 'llamacpp' ? 'llamacpp' : 'ollama'
  );
  const [contextSize, setContextSize] = useState(() => Number(localStorage.getItem('ollama_context_size')) || 32_768);
  const [thinkingMode, setThinkingMode] = useState(false);
  const [numThread, setNumThread] = useState(() => Number(localStorage.getItem('ollama_num_thread')) === 8 ? 8 : 6);
  const [cavemanMode, setCavemanMode] = useState(() => localStorage.getItem('ollama_caveman_mode') === 'true');
  const [soundEnabled, setSoundEnabled] = useState(() => localStorage.getItem('ollama_sound_enabled') !== 'false');
  const [modelMap, setModelMap] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  // Folder open/closed state lives at the App root so it is preserved when the
  // user switches between the right-panel tabs (Editor / Terminal / Files /
  // Novel). It is also persisted to localStorage so the tree reopens exactly
  // as the user left it across reloads. Defaults to fully collapsed.
  const [folderOpenState, setFolderOpenState] = useState<Record<string, boolean>>(() => {
    try {
      const saved = localStorage.getItem('ollama_folder_open_state');
      if (!saved) return {};
      const parsed = JSON.parse(saved);
      return parsed && typeof parsed === 'object' ? (parsed as Record<string, boolean>) : {};
    } catch {
      return {};
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('ollama_folder_open_state', JSON.stringify(folderOpenState));
    } catch {
      // Ignore quota/serialization failures — the in-memory state still works.
    }
  }, [folderOpenState]);

  // Fetch model map on startup
  useEffect(() => {
    fetch('/api/ollama/model-map')
      .then((res) => res.json())
      .then((data) => setModelMap(data))
      .catch((err) => console.error('Failed to load model map:', err));
  }, []);

  // Sync thinking mode and context size with the selected model's capabilities
  useEffect(() => {
    if (inferenceBackend === 'llamacpp') {
      setThinkingMode(false);
    }
    if (!modelMap || !model) return;
    const currentModelDef = modelMap.models?.find((m: any) => m.id === model);
    if (!currentModelDef) return;

    // ── Thinking mode ──────────────────────────────────────────────────────
    if (!currentModelDef.capabilities?.thinking_mode) {
      // Model does not support thinking at all — turn it off
      setThinkingMode(false);
    }

    // ── Context size ────────────────────────────────────────────────────────
    // Adjust contextSize only if current selection exceeds model's max_context
    const modelMax: number | undefined = currentModelDef.max_context;
    if (modelMax && contextSize > modelMax) {
      const best = [...CONTEXT_SIZES].reverse().find((s) => s <= modelMax);
      if (best) setContextSize(best);
    }
  }, [model, modelMap, inferenceBackend, contextSize]);

  // Active Session Persistence
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    return localStorage.getItem('ollama_active_session') || generateSessionId();
  });

  // Selected Persona for current/new sessions
  const [activePersona, setActivePersona] = useState<PersonaType>('coder');
  // A new session has no saved metadata yet. Keep the persona selected by its
  // quick-action button until it has been persisted by the first request.
  const newSessionPersonas = useRef(new Map<string, PersonaType>());
  const newSessionModels = useRef(new Map<string, string>());

  // Left sidebar and right tools panel collapse via slim switches built
  // into their edges (Alt+C / Alt+T also work).
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState<RightTab>('terminal');
  const [novelOutlineReady, setNovelOutlineReady] = useState(false);
  const [novelFirstChapter, setNovelFirstChapter] = useState<{ number: number; title: string } | null>(null);

  // Hotkeys for toggling sidebars: Alt+C (Chats), Alt+T (Tools)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && (e.key === 'c' || e.key === 'C')) {
        e.preventDefault();
        setSidebarOpen((o) => !o);
      }
      if (e.altKey && (e.key === 't' || e.key === 'T')) {
        e.preventDefault();
        setRightPanelOpen((o) => !o);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Load session list on startup
  useEffect(() => {
    fetchSessionsList();
  }, [fetchSessionsList]);

  useEffect(() => {
    localStorage.setItem('ollama_context_size', String(contextSize));
  }, [contextSize]);

  useEffect(() => {
    localStorage.setItem('ollama_caveman_mode', String(cavemanMode));
  }, [cavemanMode]);

  useEffect(() => {
    localStorage.setItem('ollama_num_thread', String(numThread));
  }, [numThread]);

  useEffect(() => {
    localStorage.setItem('ollama_inference_backend', inferenceBackend);
  }, [inferenceBackend]);

  useEffect(() => {
    localStorage.setItem('ollama_sound_enabled', String(soundEnabled));
  }, [soundEnabled]);

  // Play a soft chime when a streaming session reaches a terminal state.
  // Only fires on the transition out of an active run — page loads and
  // session switches do not retrigger it.
  const prevStatusRef = useRef<typeof status>(status);
  useEffect(() => {
    const prev = prevStatusRef.current;
    const wasActive = prev === 'thinking' || prev === 'tool';
    const nowTerminal = status === 'idle' || status === 'stopped' || status === 'error';
    if (wasActive && nowTerminal && soundEnabled) {
      playChime();
    }
    prevStatusRef.current = status;
  }, [status, soundEnabled]);

  // Load selected session history and restore its active persona and model.
  useEffect(() => {
    let cancelled = false;
    const requestedPersona = newSessionPersonas.current.get(activeSessionId);
    const requestedModel = newSessionModels.current.get(activeSessionId);
    localStorage.setItem('ollama_active_session', activeSessionId);
    loadSessionHistory(activeSessionId).then((session) => {
      if (!cancelled) {
        setActivePersona(requestedPersona ?? session.persona as PersonaType);
        setModel((current) => requestedModel || session.model || current);
        if (requestedPersona) newSessionPersonas.current.delete(activeSessionId);
        if (requestedModel !== undefined) newSessionModels.current.delete(activeSessionId);
      }
    });
    return () => { cancelled = true; };
  }, [activeSessionId, loadSessionHistory]);

  useEffect(() => {
    setNovelOutlineReady(false);
    setNovelFirstChapter(null);
  }, [activeSessionId]);

  const handleFileSelect = (path: string) => {
    setSelectedFile(path);
    setRightPanelTab('editor');
    setRightPanelOpen(true);
  };

  const handleFolderOpenChange = (path: string, open: boolean) => {
    setFolderOpenState((current) => ({ ...current, [path]: open }));
  };

  const startNewChat = (persona: PersonaType = 'coder') => {
    const newId = generateSessionId();
    newSessionPersonas.current.set(newId, persona);
    newSessionModels.current.set(newId, model);
    setActivePersona(persona);
    setActiveSessionId(newId);
  };

  const createNovelChat = async (title: string): Promise<string> => {
    const newId = generateSessionId();
    const res = await fetch('/api/ollama/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: newId,
        title: `${title} — Outline`,
        persona: 'novelist',
        model,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      throw new Error(data.error || 'Failed to create dedicated novel chat');
    }
    newSessionPersonas.current.set(newId, 'novelist');
    newSessionModels.current.set(newId, model);
    await fetchSessionsList();
    return newId;
  };

  const activateNovelChat = (sessionId: string) => {
    setActivePersona('novelist');
    setActiveSessionId(sessionId);
  };

  const handleNovelOutlineReady = useCallback((
    hasOutline: boolean,
    firstChapter?: { number: number; title: string },
  ) => {
    setNovelOutlineReady(hasOutline && Boolean(firstChapter));
    setNovelFirstChapter(firstChapter ? { number: firstChapter.number, title: firstChapter.title } : null);
  }, []);

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this chat history?')) {
      await deleteSession(id);
      if (activeSessionId === id) {
        startNewChat(activePersona);
      }
    }
  };

  const isRunning = status === 'thinking' || status === 'tool';

  return (
    <div style={{ height: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-base)' }}>
      {/* ── Top Bar ── */}
      <header style={{
        height: 52,
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '0 16px',
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
        zIndex: 10,
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 8 }}>
          <div style={{
            width: 30, height: 30, borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--accent), var(--cyan))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 16, boxShadow: '0 0 12px var(--accent-glow)',
          }}>✦</div>
          <span style={{ fontWeight: 700, fontSize: 14, background: 'linear-gradient(90deg, var(--accent-light), var(--cyan))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Ollama Agent
          </span>
        </div>

        {/* Model selector */}
        <ModelSelector value={model} onChange={setModel} disabled={isRunning} persona={activePersona} backend={inferenceBackend} />

        <div className="flex items-center gap-2" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Runtime</span>
          <select
            value={inferenceBackend}
            onChange={(event) => setInferenceBackend(event.target.value as InferenceBackend)}
            disabled={isRunning}
            title="Select inference runtime backend"
            style={{ padding: '5px 8px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)' }}
          >
            <option value="ollama">Ollama (CPU/Vulkan)</option>
            <option value="llamacpp">llama.cpp SYCL</option>
          </select>
        </div>

        {(() => {
          const currentModelDef = modelMap?.models?.find((m: any) => m.id === model);
          const modelMax: number | undefined = currentModelDef?.max_context;
          const allowedSizes = modelMax
            ? CONTEXT_SIZES.filter((s) => s <= modelMax)
            : CONTEXT_SIZES;
          return (
            <div className="flex items-center gap-2" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Context</span>
              <select
                value={contextSize}
                onChange={(event) => setContextSize(Number(event.target.value))}
                disabled={isRunning}
                title="Context window used for the next model run"
                style={{ padding: '5px 8px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)' }}
              >
                {allowedSizes.map((size) => <option key={size} value={size}>{size / 1024}k</option>)}
              </select>
            </div>
          );
        })()}


        {/* Persona selector — placed right after context, before the toggles */}
        <div className="flex items-center gap-2" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Persona</span>
          <select
            id="persona-selector"
            value={activePersona}
            onChange={(e) => setActivePersona(e.target.value as PersonaType)}
            disabled={isRunning || messages.length > 0}
            style={{
              padding: '5px 10px',
              fontSize: 12,
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              cursor: messages.length > 0 ? 'not-allowed' : 'pointer',
            }}
            title={messages.length > 0 ? "Reset conversation to change persona" : "Select AI Agent Persona"}
          >
            <option value="coder">💻 Coder</option>
            <option value="researcher">🔍 Researcher</option>
            <option value="creative">✍ Creative</option>
            <option value="novelist">📖 Novelist</option>
            <option value="system">⚙ System</option>
          </select>
        </div>

        {/* Thinking mode toggle */}
        {(() => {
          const selectedModelDef = modelMap?.models?.find((m: any) => m.id === model);
          const supportsThinking = inferenceBackend === 'ollama' && (selectedModelDef ? (selectedModelDef.capabilities?.thinking_mode ?? false) : true);
          return (
            <label
              className="flex items-center gap-2"
              style={{
                fontSize: 12,
                color: supportsThinking ? 'var(--text-secondary)' : 'var(--text-muted)',
                cursor: (isRunning || !supportsThinking) ? 'not-allowed' : 'pointer',
                opacity: supportsThinking ? 1 : 0.5,
              }}
              title={supportsThinking
                ? 'Use model reasoning when supported'
                : inferenceBackend === 'llamacpp'
                  ? 'Thinking mode is disabled on llama.cpp SYCL runtime'
                  : 'Thinking mode not supported by this model'}
            >
              <input
                type="checkbox"
                checked={thinkingMode}
                onChange={(event) => setThinkingMode(event.target.checked)}
                disabled={isRunning || !supportsThinking}
              />
              Thinking mode
            </label>
          );
        })()}

        {/* Thread tuning selector */}
        <div className="flex items-center gap-2" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Threads</span>
          <select
            value={numThread}
            onChange={(event) => setNumThread(Number(event.target.value))}
            disabled={isRunning || inferenceBackend === 'llamacpp'}
            title={inferenceBackend === 'ollama'
              ? 'Thread count sent as num_thread on each request'
              : 'Thread count applies to Ollama runtime only'}
            style={{ padding: '5px 8px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)' }}
          >
            <option value={6}>6 (P-core focus)</option>
            <option value={8}>8 (broader)</option>
          </select>
        </div>

        {/* Caveman mode toggle */}
        <label
          className="flex items-center gap-2"
          style={{
            fontSize: 12,
            color: cavemanMode ? 'var(--amber, #f59e0b)' : 'var(--text-secondary)',
            cursor: isRunning ? 'not-allowed' : 'pointer',
            opacity: isRunning ? 0.5 : 1,
            transition: 'color 0.2s',
          }}
          title="Caveman mode — same answers, 65% fewer output tokens. Drops filler words and pleasantries, keeps full technical accuracy."
        >
          <input
            type="checkbox"
            checked={cavemanMode}
            onChange={(e) => setCavemanMode(e.target.checked)}
            disabled={isRunning}
          />
          🪨 Caveman
        </label>

        {/* Spacer — pushes token bar to the right */}
        <div style={{ flex: 1 }} />

        {/* Token bar with Context label, plus live read/write token speeds */}
        <div
          className="flex items-stretch gap-2"
          style={{
            borderLeft: '1px solid var(--border)',
            paddingLeft: 12,
            minWidth: 240,
          }}
        >
          <div className="flex flex-col justify-center" style={{ gap: 1 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Context</span>
            <div
              className="flex items-center"
              style={{ gap: 10, fontSize: 10.5, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}
              title="Live prompt prefill (read) and token generation (write) throughput, computed from the last 8 SSE token-counter samples."
            >
              <span style={{ color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                👁 <span style={{ color: 'var(--text-muted)' }}>Read</span>
                <span style={{ color: tokenRates.read > 0 ? 'var(--cyan)' : 'var(--text-muted)' }}>
                  {formatTokenRate(tokenRates.read)} t/s
                </span>
              </span>
              <span style={{ color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                ⌨ <span style={{ color: 'var(--text-muted)' }}>Write</span>
                <span style={{ color: tokenRates.write > 0 ? 'var(--accent-light)' : 'var(--text-muted)' }}>
                  {formatTokenRate(tokenRates.write)} t/s
                </span>
              </span>
            </div>
          </div>
          <ProgressBar input={tokenUsage.input} output={tokenUsage.output} />
        </div>


      </header>

      {/* ── Main Body ── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* Left Sidebar — collapse switch on its right edge */}
        <div style={{
          display: 'flex',
          flexShrink: 0,
          position: 'relative',
        }}>
          {sidebarOpen && (
            <aside style={{
              width: 230,
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-surface)',
          overflow: 'hidden',
        }}>
              {/* New Chat Button (split with persona selectors) */}
              <div style={{ padding: 12, borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 6 }}>
                <button
                  className="btn btn-primary"
                  style={{ width: '100%', justifyContent: 'center' }}
                  onClick={() => startNewChat(activePersona)}
                >
                  + New Chat
                </button>
                <div style={{ display: 'flex', gap: 4, width: '100%' }}>
                  {(['coder', 'researcher', 'creative', 'novelist', 'system'] as const).map((p) => (
                    <button
                      key={p}
                      onClick={() => {
                        if (p === 'novelist') {
                          setRightPanelTab('novel');
                          setRightPanelOpen(true);
                        } else {
                          startNewChat(p);
                        }
                      }}
                      style={{
                        flex: 1,
                        padding: '4px',
                        fontSize: '10px',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-card)',
                        color: 'var(--text-secondary)',
                        cursor: 'pointer',
                      }}
                      title={p === 'novelist' ? 'Open Novel Studio' : `Create new ${p} chat`}
                    >
                      {p === 'coder' && '💻'}
                      {p === 'researcher' && '🔍'}
                      {p === 'creative' && '✍'}
                      {p === 'novelist' && '📖'}
                      {p === 'system' && '⚙'}
                    </button>
                  ))}
                </div>
              </div>

              {/* List of chat sessions */}
              <div className="flex-1 overflow-auto" style={{ padding: '8px 6px', display: 'flex', flexDirection: 'column', gap: 2 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', paddingLeft: 8, paddingBottom: 6, letterSpacing: '0.08em' }}>
                  Recent Chats
                </div>
                {sessionsList.length === 0 && (
                  <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 11, textAlign: 'center' }}>
                    No past chats
                  </div>
                )}
                {sessionsList.map((s) => {
                  const isActive = s.sessionId === activeSessionId;
                  const badge = personaBadges[s.persona as PersonaType] || { label: 'Agent', color: 'var(--text-muted)' };
                  return (
                    <div
                      key={s.sessionId}
                      onClick={() => setActiveSessionId(s.sessionId)}
                      className="flex items-center justify-between"
                      style={{
                        padding: '8px 10px',
                        borderRadius: 'var(--radius-md)',
                        background: isActive ? 'var(--accent-glow)' : 'transparent',
                        color: isActive ? 'var(--accent-light)' : 'var(--text-primary)',
                        cursor: 'pointer',
                        fontSize: 12.5,
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--bg-hover)'; }}
                      onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
                    >
                      <div className="flex flex-col flex-1 truncate" style={{ marginRight: 6 }}>
                        <span className="truncate" style={{ fontWeight: 500 }}>
                          {s.title}
                        </span>
                        <span style={{ fontSize: 9, color: badge.color, marginTop: 2, display: 'inline-block' }}>
                          {badge.label}
                        </span>
                      </div>
                      <button
                        onClick={(e) => handleDeleteSession(e, s.sessionId)}
                        style={{
                          background: 'none', border: 'none', color: 'var(--text-muted)',
                          cursor: 'pointer', fontSize: 11, opacity: isActive ? 1 : 0,
                        }}
                        className="delete-chat-btn"
                        title="Delete chat"
                      >
                        ✕
                      </button>
                    </div>
                  );
                })}
              </div>

              {/* Sound toggle at bottom of sidebar */}
              <div style={{
                padding: '10px 12px',
                borderTop: '1px solid var(--border)',
                background: 'var(--bg-surface)',
                flexShrink: 0,
              }}>
                <label
                  className="flex items-center justify-between"
                  style={{
                    fontSize: 12,
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    userSelect: 'none',
                  }}
                  onClick={() => {
                    // Unlock the audio context on first user interaction so
                    // subsequent end-of-session chimes can play under browser
                    // autoplay policies.
                    unlockAudio();
                    setSoundEnabled((prev) => !prev);
                  }}
                  title="Play a soft chime when a session finishes"
                >
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 13 }}>{soundEnabled ? '🔔' : '🔕'}</span>
                    <span>Session end sound</span>
                  </span>
                  <span
                    style={{
                      width: 30,
                      height: 16,
                      borderRadius: 10,
                      background: soundEnabled ? 'var(--accent)' : 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      position: 'relative',
                      transition: 'background 0.2s',
                      flexShrink: 0,
                    }}
                  >
                    <span
                      style={{
                        position: 'absolute',
                        top: 1,
                        left: soundEnabled ? 14 : 1,
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        background: soundEnabled ? 'var(--bg-surface)' : 'var(--text-muted)',
                        transition: 'left 0.2s, background 0.2s',
                      }}
                    />
                  </span>
                </label>
              </div>
            </aside>
          )}

          {/* Collapse switch — right edge of sidebar */}
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            title={sidebarOpen ? 'Hide chats (Alt+C)' : 'Show chats (Alt+C)'}
            aria-label={sidebarOpen ? 'Hide chats' : 'Show chats'}
            style={{
              width: 16,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 4,
              background: 'var(--bg-surface)',
              border: 'none',
              borderRight: '1px solid var(--border)',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              fontSize: 9,
              padding: '4px 0',
              transition: 'color 0.15s, background 0.15s',
              flexShrink: 0,
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.background = 'var(--bg-hover)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.background = 'var(--bg-surface)'; }}
          >
            <span style={{ fontSize: 11, lineHeight: 1 }}>💬</span>
            <span style={{ fontSize: 10, lineHeight: 1 }}>{sidebarOpen ? '◀' : '▶'}</span>
          </button>
        </div>

        {/* Workspace Layout */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minWidth: 0 }}>
          {/* Main Area: Chat Panel (Always Visible) */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
            <ChatPanel
              model={model}
              sessionId={activeSessionId}
              messages={messages}
              status={status}
              sendMessage={sendMessage}
              stop={stop}
              killModel={killModel}
              resetSession={resetSession}
              persona={activePersona}
              contextSize={contextSize}
              thinkingMode={thinkingMode}
              numThread={numThread}
              inferenceBackend={inferenceBackend}
              cavemanMode={cavemanMode}
              sidebarOpen={true}
              rightPanelOpen={true}
              hasNovelOutline={novelOutlineReady}
              onStartChapter={() => {
                if (!novelFirstChapter) return;
                sendMessage(
                  `[NOVEL_DRAFT:${activeSessionId}:${novelFirstChapter.number}]`,
                  model,
                  activeSessionId,
                  'novelist',
                  { isolated: true, numCtx: contextSize, think: thinkingMode, numThread, inferenceBackend, caveman: false }
                );
              }}
              onToggleSidebar={() => setSidebarOpen((o) => !o)}
              onToggleRightPanel={() => setRightPanelOpen((o) => !o)}
            />
          </div>

          {/* Right Panel: collapse switch on its left edge */}
          <div style={{
            display: 'flex',
            flexShrink: 0,
          }}>
            <button
              onClick={() => setRightPanelOpen((o) => !o)}
              title={rightPanelOpen ? 'Hide tools (Alt+T)' : 'Show tools (Alt+T)'}
              aria-label={rightPanelOpen ? 'Hide tools' : 'Show tools'}
              style={{
                width: 16,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 4,
                background: 'var(--bg-surface)',
                border: 'none',
                borderRight: '1px solid var(--border)',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                fontSize: 9,
                padding: '4px 0',
                transition: 'color 0.15s, background 0.15s',
                flexShrink: 0,
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-primary)'; e.currentTarget.style.background = 'var(--bg-hover)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.background = 'var(--bg-surface)'; }}
            >
              <span style={{ fontSize: 10, lineHeight: 1 }}>{rightPanelOpen ? '▶' : '◀'}</span>
              <span style={{ fontSize: 11, lineHeight: 1 }}>⊞</span>
            </button>

            <div style={{
              display: rightPanelOpen ? 'flex' : 'none',
              width: '50vw',
              minWidth: 400,
              flexDirection: 'column',
              overflow: 'hidden',
              background: 'var(--bg-surface)',
              borderLeft: '1px solid var(--border)',
            }}>
                {/* Tab headers */}
                <div style={{
                  display: 'flex',
                  borderBottom: '1px solid var(--border)',
                  background: 'var(--bg-surface)',
                  flexShrink: 0,
                }}>
                  {([
                    { id: 'editor', label: '📝 Editor' },
                    { id: 'terminal', label: '⌨ Terminal' },
                    { id: 'files', label: '📁 Files' },
                    { id: 'novel', label: '📖 Novel' },
                  ] as const).map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setRightPanelTab(tab.id)}
                      style={{
                        padding: '9px 18px',
                        fontSize: 12,
                        fontWeight: 500,
                        background: 'none',
                        border: 'none',
                        borderBottom: `2px solid ${rightPanelTab === tab.id ? 'var(--accent)' : 'transparent'}`,
                        color: rightPanelTab === tab.id ? 'var(--accent-light)' : 'var(--text-muted)',
                        cursor: 'pointer',
                        transition: 'all 0.15s',
                      }}
                    >
                      {tab.label}
                    </button>
                  ))}
                  {selectedFile && rightPanelTab === 'editor' && (
                    <span style={{
                      alignSelf: 'center', marginLeft: 'auto', marginRight: 12,
                      fontSize: 11, fontFamily: 'var(--font-mono)',
                      color: 'var(--text-muted)', maxWidth: 200,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {selectedFile}
                    </span>
                  )}
                </div>

                {/* Panel content */}
                <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                  {rightPanelTab === 'editor' && <FileEditor filePath={selectedFile} />}
                  {rightPanelTab === 'terminal' && <TerminalPane label="User Terminal" />}
                  {rightPanelTab === 'files' && (
                    <FileExplorer
                      onSelectFile={handleFileSelect}
                      selectedPath={selectedFile ?? undefined}
                      folderOpenState={folderOpenState}
                      onFolderOpenChange={handleFolderOpenChange}
                    />
                  )}
                  <div style={{
                    display: rightPanelTab === 'novel' ? 'flex' : 'none',
                    flex: 1,
                    minHeight: 0,
                    flexDirection: 'column',
                  }}>
                    <NovelStudio
                      sessionId={activeSessionId}
                      modelId={model}
                      thinkingMode={thinkingMode}
                      contextSize={contextSize}
                      isRunning={isRunning}
                      stop={stop}
                      killModel={killModel}
                      inferenceBackend={inferenceBackend}
                      onCreateNovelSession={createNovelChat}
                      onActivateNovelSession={activateNovelChat}
                      onOutlineReady={handleNovelOutlineReady}
                      onDraftChapter={(novelId, chapterNum, chapterTitle) => {
                        sendMessage(
                          `[NOVEL_DRAFT:${novelId}:${chapterNum}]`,
                          model,
                          activeSessionId,
                          'novelist',
                          { isolated: true, numCtx: contextSize, think: thinkingMode, numThread, inferenceBackend, caveman: false }
                        );
                      }}
                    />
                  </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Status footer ── */}
      <footer style={{
        height: 24,
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        background: 'var(--bg-surface)',
        borderTop: '1px solid var(--border)',
        fontSize: 11,
        color: 'var(--text-muted)',
        gap: 16,
        flexShrink: 0,
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span className="dot dot-green" style={{ width: 5, height: 5 }} />
          Ollama Agent UI v1.0
        </span>
        <span>workspace: agent-workspace/</span>
        {selectedFile && <span>editing: {selectedFile}</span>}
      </footer>
    </div>
  );
}
