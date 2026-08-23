import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { FileExplorer } from './components/FileExplorer';
import { FileEditor } from './components/FileEditor';
import { TerminalPane } from './components/TerminalPane';
import { ModelSelector } from './components/ModelSelector';
import { StatusIndicator } from './components/StatusIndicator';
import { ProgressBar } from './components/ProgressBar';
import { NovelStudio } from './components/NovelStudio';
import { CloudProviderModal } from './components/CloudProviderModal';
import { useOllamaStream, type InferenceBackend } from './hooks/useOllamaStream';
import { formatTokenRate } from './lib/tokenCounter';
import { playChime, playPauseChime, unlockAudio } from './lib/chime';
import { useCloudProviderSettings } from './lib/providerConfig';

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
    rerunFrom,
    stop,
    killModel,
    resetSession,
    loadSessionHistory,
    sessionsList,
    fetchSessionsList,
    deleteSession,
    stalledMs,
    autoStopNotice,
    clearAutoStopNotice,
    pendingAskUser,
    resumeSession,
    clearPendingAskUser,
  } = useOllamaStream();

  const [model, setModel] = useState('');
  const { settings: cloudSettings, setSettings: setCloudSettings } = useCloudProviderSettings();
  const [cloudMode, setCloudMode] = useState(() => localStorage.getItem('ollama_cloud_mode') === 'true');
  const [cloudModalOpen, setCloudModalOpen] = useState(false);
  const [askUserGuidance, setAskUserGuidance] = useState('');
  const activeModel = cloudMode ? cloudSettings.model : model;
  const activeCloudProvider = cloudMode ? cloudSettings : undefined;
  const [inferenceBackend, setInferenceBackend] = useState<InferenceBackend>(() =>
    localStorage.getItem('ollama_inference_backend') === 'llamacpp' ? 'llamacpp' : 'ollama'
  );
  const [contextSize, setContextSize] = useState(() => Number(localStorage.getItem('ollama_context_size')) || 32_768);
  const [thinkingMode, setThinkingMode] = useState(false);
  const numThread = 8; // hardcoded — threads selector removed from UI
  const [cavemanMode, setCavemanMode] = useState(() => localStorage.getItem('ollama_caveman_mode') === 'true');
  const [soundEnabled, setSoundEnabled] = useState(() => localStorage.getItem('ollama_sound_enabled') !== 'false');
  // Autopilot for installs: when on, side-effecting installs (npm/pip/winget/
  // etc.) run without per-command approval for the rest of the run. Opt in at
  // session start; persists across reloads and stays on until toggled off.
  const [autopilotInstalls, setAutopilotInstalls] = useState(() => localStorage.getItem('ollama_autopilot_installs') === 'true');
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
    if (cloudMode || inferenceBackend === 'llamacpp') {
      setThinkingMode(false);
    }
    if (cloudMode || !modelMap || !model) return;
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
  }, [model, modelMap, inferenceBackend, contextSize, cloudMode]);

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

  // Drag-to-resize panel widths (persisted to localStorage)
  const [sidebarWidth, setSidebarWidth] = useState(() =>
    Number(localStorage.getItem('ollama_sidebar_width')) || 230
  );
  const [rightPanelWidth, setRightPanelWidth] = useState(() =>
    Number(localStorage.getItem('ollama_right_panel_width')) || 480
  );
  const dragRef = useRef<{
    target: 'sidebar' | 'right';
    startX: number;
    startWidth: number;
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
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

  // Panel drag-to-resize
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      e.preventDefault();
      const delta = e.clientX - dragRef.current.startX;
      if (dragRef.current.target === 'sidebar') {
        const next = Math.max(160, Math.min(520, dragRef.current.startWidth + delta));
        setSidebarWidth(next);
        localStorage.setItem('ollama_sidebar_width', String(next));
      } else {
        const next = Math.max(280, Math.min(window.innerWidth - 320, dragRef.current.startWidth - delta));
        setRightPanelWidth(next);
        localStorage.setItem('ollama_right_panel_width', String(next));
      }
    };
    const onMouseUp = () => {
      if (dragRef.current) {
        dragRef.current = null;
        setIsDragging(false);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
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
    localStorage.setItem('ollama_inference_backend', inferenceBackend);
  }, [inferenceBackend]);

  useEffect(() => {
    localStorage.setItem('ollama_cloud_mode', String(cloudMode));
  }, [cloudMode]);

  useEffect(() => {
    localStorage.setItem('ollama_sound_enabled', String(soundEnabled));
  }, [soundEnabled]);

  useEffect(() => {
    localStorage.setItem('ollama_autopilot_installs', String(autopilotInstalls));
  }, [autopilotInstalls]);

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
    // Distinct ascending cue when the agent pauses for user input (ask_user).
    // Same sound toggle as the end-of-session chime — no separate setting.
    if (wasActive && status === 'paused' && soundEnabled) {
      playPauseChime();
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
    newSessionModels.current.set(newId, activeModel);
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
        model: activeModel,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      throw new Error(data.error || 'Failed to create dedicated novel chat');
    }
    newSessionPersonas.current.set(newId, 'novelist');
    newSessionModels.current.set(newId, activeModel);
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
      {/* ── Top Bar (single row — icon toggles keep it compact) ── */}
      <header className="toolbar-row" style={{
        height: 52,
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '0 14px',
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0, zIndex: 10,
        overflowX: 'auto',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexShrink: 0, marginRight: 4 }}>
          <div style={{
            width: 28, height: 28, borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--accent), var(--cyan))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 15, boxShadow: '0 0 10px var(--accent-glow)',
          }}>✦</div>
          <span style={{ fontWeight: 700, fontSize: 13, whiteSpace: 'nowrap', background: 'linear-gradient(90deg, var(--accent-light), var(--cyan))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Ollama Agent
          </span>
        </div>

        {/* Model */}
        <div style={{ flexShrink: 0 }}>
          {cloudMode ? (
            <button className="btn" onClick={() => setCloudModalOpen(true)} disabled={isRunning}
              title="Configure cloud provider and model"
              style={{ maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--cyan)' }}>
              ☁ {cloudSettings.model || 'Configure cloud model'}
            </button>
          ) : (
            <ModelSelector value={model} onChange={setModel} disabled={isRunning} persona={activePersona} backend={inferenceBackend} />
          )}
        </div>

        {/* Runtime */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, borderLeft: '1px solid var(--border)', paddingLeft: 10, flexShrink: 0 }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Runtime</span>
          <select value={inferenceBackend} onChange={(e) => setInferenceBackend(e.target.value as InferenceBackend)}
            disabled={isRunning || cloudMode} title="Select inference runtime backend"
            style={{ padding: '4px 7px', fontSize: 11, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)' }}>
            <option value="ollama">Ollama (CPU/Vulkan)</option>
            <option value="llamacpp">llama.cpp SYCL</option>
          </select>
        </div>

        {/* Cloud */}
        <button className={cloudMode ? 'btn btn-primary' : 'btn'}
          onClick={() => { if (cloudMode) setCloudMode(false); else setCloudModalOpen(true); }}
          disabled={isRunning}
          title={cloudMode ? 'Switch back to local runtime' : 'Configure cloud provider'}
          style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>
          {cloudMode ? '☁ Cloud on' : '☁ Cloud'}
        </button>

        <div style={{ width: 1, height: 22, background: 'var(--border)', flexShrink: 0 }} />

        {/* Context size */}
        {(() => {
          const currentModelDef = cloudMode ? undefined : modelMap?.models?.find((m: any) => m.id === model);
          const modelMax: number | undefined = currentModelDef?.max_context;
          const allowedSizes = cloudMode ? CONTEXT_SIZES : modelMax
            ? CONTEXT_SIZES.filter((s) => s <= modelMax) : CONTEXT_SIZES;
          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Context</span>
              <select value={contextSize} onChange={(e) => setContextSize(Number(e.target.value))}
                disabled={isRunning} title="Context window used for the next model run"
                style={{ padding: '4px 6px', fontSize: 11, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)' }}>
                {allowedSizes.map((size) => <option key={size} value={size}>{size / 1024}k</option>)}
              </select>
            </div>
          );
        })()}

        <div style={{ width: 1, height: 22, background: 'var(--border)', flexShrink: 0 }} />

        {/* Mode (persona) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Mode</span>
          <select id="persona-selector" value={activePersona}
            onChange={(e) => setActivePersona(e.target.value as PersonaType)}
            disabled={isRunning || messages.length > 0}
            style={{ padding: '4px 7px', fontSize: 11, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', cursor: messages.length > 0 ? 'not-allowed' : 'pointer' }}
            title={messages.length > 0 ? 'Reset conversation to change mode' : 'Select AI Agent Mode'}>
            <option value="coder">💻 Coder</option>
            <option value="researcher">🔍 Researcher</option>
            <option value="creative">✍ Creative</option>
            <option value="novelist">📖 Novelist</option>
            <option value="system">⚙ System</option>
          </select>
        </div>

        {/* Spacer */}
        <div style={{ flex: 1, minWidth: 8 }} />

        {/* ── Icon toggles: Thinking + Caveman ── */}
        {(() => {
          const selectedModelDef = cloudMode ? undefined : modelMap?.models?.find((m: any) => m.id === model);
          const supportsThinking = !cloudMode && inferenceBackend === 'ollama'
            && (selectedModelDef ? (selectedModelDef.capabilities?.thinking_mode ?? false) : true);

          const thinkTitle = supportsThinking
            ? (thinkingMode ? 'Thinking — ON (click to disable)' : 'Thinking — OFF (click to enable)')
            : cloudMode ? 'Thinking — not available for cloud providers'
            : inferenceBackend === 'llamacpp' ? 'Thinking — not available on llama.cpp'
            : 'Thinking — not supported by this model';

          const cavemanTitle = cavemanMode
            ? 'Caveman — ON · 65% fewer tokens, same accuracy (click to disable)'
            : 'Caveman — OFF · click to enable compressed responses';

          return (
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0 }}>
              {/* 🧠 Thinking toggle */}
              <button
                onClick={() => { if (supportsThinking && !isRunning) setThinkingMode((v) => !v); }}
                title={thinkTitle}
                aria-label="Thinking"
                style={{
                  width: 32, height: 32, borderRadius: '50%', border: '1.5px solid',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 16, padding: 0,
                  cursor: (!supportsThinking || isRunning) ? 'not-allowed' : 'pointer',
                  opacity: isRunning ? 0.5 : (!supportsThinking ? 0.3 : 1),
                  background: !supportsThinking
                    ? 'var(--bg-card)'
                    : thinkingMode
                      ? 'rgba(57,217,138,0.15)'
                      : 'rgba(240,96,112,0.12)',
                  borderColor: !supportsThinking
                    ? 'var(--border)'
                    : thinkingMode ? 'var(--green)' : 'rgba(240,96,112,0.55)',
                  transition: 'background 0.2s, border-color 0.2s',
                }}
              >🧠</button>

              {/* 🪵 Caveman toggle */}
              <button
                onClick={() => { if (!isRunning) setCavemanMode((v) => !v); }}
                title={cavemanTitle}
                aria-label="Caveman"
                style={{
                  width: 32, height: 32, borderRadius: '50%', border: '1.5px solid',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 16, padding: 0,
                  cursor: isRunning ? 'not-allowed' : 'pointer',
                  opacity: isRunning ? 0.5 : 1,
                  background: cavemanMode ? 'rgba(57,217,138,0.15)' : 'rgba(240,96,112,0.12)',
                  borderColor: cavemanMode ? 'var(--green)' : 'rgba(240,96,112,0.55)',
                  transition: 'background 0.2s, border-color 0.2s',
                }}
              >🪵</button>
            </div>
          );
        })()}

        {/* Token bar */}
        <div style={{ display: 'flex', alignItems: 'stretch', gap: 8, flexShrink: 0, borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 1 }}>
            <span style={{ fontSize: 10, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Context usage</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 10, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}
              title="Live prompt prefill (read) and token generation (write) throughput.">
              <span style={{ color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                👁 <span style={{ color: 'var(--text-muted)' }}>R</span>
                <span style={{ color: tokenRates.read > 0 ? 'var(--cyan)' : 'var(--text-muted)' }}>{formatTokenRate(tokenRates.read)} t/s</span>
              </span>
              <span style={{ color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                ⌨ <span style={{ color: 'var(--text-muted)' }}>W</span>
                <span style={{ color: tokenRates.write > 0 ? 'var(--accent-light)' : 'var(--text-muted)' }}>{formatTokenRate(tokenRates.write)} t/s</span>
              </span>
            </div>
          </div>
          <ProgressBar input={tokenUsage.input} output={tokenUsage.output} />
        </div>
      </header>

      {/* ── Run status banner ─────────────────────────────────────────────
           Three states:
           1. autoStopNotice  — run was auto-stopped; persists until dismissed
                                or the next message is sent.
           2. active stall    — run is still live but silent > 90 s; offer
                                manual Force Stop while auto-stop fires within
                                the next 10-s interval.
           Both share the same red strip so the UI is consistent.
      ──────────────────────────────────────────────────────────────────── */}
      {(autoStopNotice || (stalledMs > 0 && isRunning)) && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '7px 16px',
          background: autoStopNotice ? 'rgba(240,96,112,0.05)' : 'rgba(240,96,112,0.08)',
          borderBottom: '1px solid rgba(240,96,112,0.35)',
          borderLeft: '3px solid var(--red)',
          flexShrink: 0, zIndex: 9,
        }}>
          <span style={{ fontSize: 15 }}>{autoStopNotice?.includes('⚡') ? '⚡' : autoStopNotice ? '🛑' : '⏳'}</span>
          <span style={{ fontSize: 12, color: 'var(--red)', fontWeight: 600, whiteSpace: 'nowrap' }}>
            {autoStopNotice?.includes('⚡') ? 'Model Offloaded' : autoStopNotice ? 'Stopped' : 'Quiet Run'}
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1 }}>
            {autoStopNotice ?? (
              `No activity for ${
                Math.floor(stalledMs / 60000) > 0
                  ? `${Math.floor(stalledMs / 60000)}m ${Math.round((stalledMs % 60000) / 1000)}s`
                  : `${Math.round(stalledMs / 1000)}s`
              }. Run continues indefinitely until complete or stopped.`
            )}
          </span>
          {autoStopNotice ? (
            <button
              onClick={clearAutoStopNotice}
              style={{
                padding: '4px 12px', fontSize: 11, fontWeight: 600,
                background: 'transparent', color: 'var(--text-muted)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                cursor: 'pointer', whiteSpace: 'nowrap', flexShrink: 0,
              }}
            >
              Dismiss
            </button>
          ) : (
            <button
              onClick={() => stop(activeSessionId)}
              style={{
                padding: '4px 12px', fontSize: 11, fontWeight: 600,
                background: 'var(--red)', color: '#fff',
                border: 'none', borderRadius: 'var(--radius-md)',
                cursor: 'pointer', whiteSpace: 'nowrap', flexShrink: 0,
              }}
            >
              Force Stop
            </button>
          )}
        </div>
      )}

      {/* ── ask_user pause banner ──────────────────────────────────────────
           The agent paused for human input: either it wants to run a
           side-effecting install (approve/deny) or it has burned 3/4 tool
           failures and needs guidance. The run stays alive server-side
           (heartbeated) until the user responds via /api/ollama/resume.
      ──────────────────────────────────────────────────────────────────── */}
      {pendingAskUser && (
        <div style={{
          display: 'flex', flexDirection: 'column', gap: 10,
          padding: '12px 16px',
          background: 'rgba(250,176,5,0.06)',
          borderBottom: '1px solid rgba(250,176,5,0.4)',
          borderLeft: '3px solid var(--amber, #f59e0b)',
          flexShrink: 0, zIndex: 9,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 15 }}>⏸️</span>
            <span style={{ fontSize: 12, color: 'var(--amber, #f59e0b)', fontWeight: 700, whiteSpace: 'nowrap' }}>
              Agent paused — needs your input
            </span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{pendingAskUser.prompt}</div>
          {pendingAskUser.command && (
            <pre style={{
              margin: 0, padding: '6px 10px',
              background: 'var(--bg-base)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)', fontSize: 12, color: 'var(--text-primary)',
              fontFamily: 'ui-monospace, Consolas, monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
            }}>{pendingAskUser.command}</pre>
          )}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {pendingAskUser.kind === 'install' ? (
              <>
                <button
                  onClick={() => { const id = pendingAskUser.sessionId ?? activeSessionId; const cmd = askUserGuidance.trim() || pendingAskUser.command; setAskUserGuidance(''); void resumeSession(id, 'approve', undefined, cmd); }}
                  style={{ padding: '5px 12px', fontSize: 11, fontWeight: 700, background: 'var(--green, #22c55e)', color: '#04210f', border: 'none', borderRadius: 'var(--radius-md)', cursor: 'pointer', whiteSpace: 'nowrap' }}
                >
                  {askUserGuidance.trim() ? 'Approve edited command' : 'Approve & let agent run'}
                </button>
                <button
                  onClick={() => { const id = pendingAskUser.sessionId ?? activeSessionId; const cmd = askUserGuidance.trim() || pendingAskUser.command; setAskUserGuidance(''); setAutopilotInstalls(true); void resumeSession(id, 'approve_all', undefined, cmd); }}
                  title="Approve this install AND all future installs in this run — no more prompts."
                  style={{ padding: '5px 12px', fontSize: 11, fontWeight: 700, background: 'var(--accent, #6366f1)', color: '#fff', border: 'none', borderRadius: 'var(--radius-md)', cursor: 'pointer', whiteSpace: 'nowrap' }}
                >
                  Allow all installs
                </button>
                <button
                  onClick={() => { const id = pendingAskUser.sessionId ?? activeSessionId; const reason = askUserGuidance.trim() || undefined; setAskUserGuidance(''); void resumeSession(id, 'deny', reason); }}
                  style={{ padding: '5px 12px', fontSize: 11, fontWeight: 700, background: 'var(--red, #ef4444)', color: '#fff', border: 'none', borderRadius: 'var(--radius-md)', cursor: 'pointer', whiteSpace: 'nowrap' }}
                >
                  Deny
                </button>
              </>
            ) : (
              <button
                onClick={() => { const id = pendingAskUser.sessionId ?? activeSessionId; const msg = askUserGuidance.trim() || undefined; setAskUserGuidance(''); void resumeSession(id, 'guidance', msg); }}
                style={{ padding: '5px 12px', fontSize: 11, fontWeight: 700, background: 'var(--accent, #6366f1)', color: '#fff', border: 'none', borderRadius: 'var(--radius-md)', cursor: 'pointer', whiteSpace: 'nowrap' }}
              >
                Send guidance
              </button>
            )}
            <input
              value={askUserGuidance}
              onChange={(e) => setAskUserGuidance(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && pendingAskUser) { const id = pendingAskUser.sessionId ?? activeSessionId; const v = askUserGuidance.trim(); if (!v) return; setAskUserGuidance(''); void resumeSession(id, pendingAskUser.kind === 'install' ? 'approve' : 'guidance', pendingAskUser.kind === 'install' ? undefined : v, pendingAskUser.kind === 'install' ? v : undefined); } }}
              placeholder={pendingAskUser.kind === 'install' ? 'Optional: type a corrected command, then Approve — or a deny reason' : 'Type instructions for the agent, then press Enter'}
              style={{ flex: 1, minWidth: 200, padding: '5px 9px', fontSize: 12, background: 'var(--bg-base)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
            />
            <button
              onClick={() => { const id = pendingAskUser.sessionId ?? activeSessionId; setAskUserGuidance(''); clearPendingAskUser(); void stop(id); }}
              style={{ padding: '5px 12px', fontSize: 11, fontWeight: 600, background: 'transparent', color: 'var(--text-muted)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', cursor: 'pointer', whiteSpace: 'nowrap' }}
            >
              Stop
            </button>
          </div>
        </div>
      )}

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
              width: sidebarWidth,
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          borderRight: 'none',
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
                <label
                  className="flex items-center justify-between"
                  style={{
                    fontSize: 12,
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    userSelect: 'none',
                  }}
                  onClick={() => setAutopilotInstalls((prev) => !prev)}
                  title="When on, the agent runs install commands (npm/pip/winget…) without asking each time. Opt in at session start; stays on for the rest of the session."
                >
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 13 }}>{autopilotInstalls ? '🚀' : '🛑'}</span>
                    <span>Autopilot installs</span>
                  </span>
                  <span
                    style={{
                      width: 30,
                      height: 16,
                      borderRadius: 10,
                      background: autopilotInstalls ? 'var(--accent)' : 'var(--bg-card)',
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
                        left: autopilotInstalls ? 14 : 1,
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        background: autopilotInstalls ? 'var(--bg-surface)' : 'var(--text-muted)',
                        transition: 'left 0.2s, background 0.2s',
                      }}
                    />
                  </span>
                </label>
              </div>
            </aside>
          )}

          {/* Drag handle — resize sidebar width */}
          {sidebarOpen && (
            <div
              className={`panel-drag-handle${isDragging && dragRef.current?.target === 'sidebar' ? ' dragging' : ''}`}
              onMouseDown={(e) => {
                e.preventDefault();
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                dragRef.current = { target: 'sidebar', startX: e.clientX, startWidth: sidebarWidth };
                setIsDragging(true);
              }}
              style={{ borderRight: '1px solid var(--border)' }}
              title="Drag to resize sidebar"
            />
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
              model={activeModel}
              sessionId={activeSessionId}
              messages={messages}
              status={status}
              sendMessage={sendMessage}
              rerunFrom={rerunFrom}
              stop={stop}
              killModel={killModel}
              resetSession={resetSession}
              persona={activePersona}
              contextSize={contextSize}
              thinkingMode={thinkingMode}
              numThread={numThread}
              inferenceBackend={inferenceBackend}
              cloudProvider={activeCloudProvider}
              cavemanMode={cavemanMode}
              autopilotInstalls={autopilotInstalls}
              visionSupported={Boolean(modelMap?.models?.find((m: any) => m.id === activeModel)?.capabilities?.vision)}
              sidebarOpen={true}
              rightPanelOpen={true}
              hasNovelOutline={novelOutlineReady}
              onStartChapter={() => {
                if (!novelFirstChapter) return;
                sendMessage(
                  `[NOVEL_DRAFT:${activeSessionId}:${novelFirstChapter.number}]`,
                  activeModel,
                  activeSessionId,
                  'novelist',
                  { isolated: true, numCtx: contextSize, think: thinkingMode, numThread, inferenceBackend, caveman: false, autopilotInstalls, cloudProvider: activeCloudProvider }
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

            {/* Drag handle — resize right panel width */}
            {rightPanelOpen && (
              <div
                className={`panel-drag-handle${isDragging && dragRef.current?.target === 'right' ? ' dragging' : ''}`}
                onMouseDown={(e) => {
                  e.preventDefault();
                  document.body.style.cursor = 'col-resize';
                  document.body.style.userSelect = 'none';
                  dragRef.current = { target: 'right', startX: e.clientX, startWidth: rightPanelWidth };
                  setIsDragging(true);
                }}
                style={{ borderLeft: '1px solid var(--border)' }}
                title="Drag to resize tools panel"
              />
            )}

            <div style={{
              display: rightPanelOpen ? 'flex' : 'none',
              width: rightPanelWidth,
              flexDirection: 'column',
              overflow: 'hidden',
              background: 'var(--bg-surface)',
              borderLeft: 'none',
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
                      modelId={activeModel}
                      thinkingMode={thinkingMode}
                      contextSize={contextSize}
                      isRunning={isRunning}
                      stop={stop}
                      killModel={cloudMode ? undefined : killModel}
                      inferenceBackend={cloudMode ? undefined : inferenceBackend}
                      onCreateNovelSession={createNovelChat}
                      onActivateNovelSession={activateNovelChat}
                      onOutlineReady={handleNovelOutlineReady}
                      onDraftChapter={(novelId, chapterNum, chapterTitle) => {
                        sendMessage(
                          `[NOVEL_DRAFT:${novelId}:${chapterNum}]`,
                          activeModel,
                          activeSessionId,
                          'novelist',
                          { isolated: true, numCtx: contextSize, think: thinkingMode, numThread, inferenceBackend, caveman: false, autopilotInstalls, cloudProvider: activeCloudProvider }
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
          {cloudMode ? 'Cloud Agent UI v1.0' : 'Ollama Agent UI v1.0'}
        </span>
        <span>workspace: agent-workspace/</span>
        {cloudMode && <span>{cloudSettings.provider}</span>}
        {selectedFile && <span>editing: {selectedFile}</span>}
      </footer>

      <CloudProviderModal
        open={cloudModalOpen}
        disabled={isRunning}
        settings={cloudSettings}
        onSave={(next) => {
          setCloudSettings(next);
          setCloudMode(true);
        }}
        onClose={() => setCloudModalOpen(false)}
      />
    </div>
  );
}
