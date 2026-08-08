import React, { useState, useEffect, useRef } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { FileExplorer } from './components/FileExplorer';
import { FileEditor } from './components/FileEditor';
import { TerminalPane } from './components/TerminalPane';
import { ModelSelector } from './components/ModelSelector';
import { StatusIndicator } from './components/StatusIndicator';
import { ProgressBar } from './components/ProgressBar';
import { useOllamaStream } from './hooks/useOllamaStream';

function generateSessionId() {
  return `session_${Math.random().toString(36).slice(2, 11)}`;
}

type RightTab = 'editor' | 'terminal' | 'files';
type PersonaType = 'coder' | 'researcher' | 'creative';
const CONTEXT_SIZES = [16_384, 32_768, 65_536, 131_072, 262_144];

const personaBadges: Record<PersonaType, { label: string; color: string }> = {
  coder: { label: 'Coder', color: 'var(--blue)' },
  researcher: { label: 'Researcher', color: 'var(--green)' },
  creative: { label: 'Creative', color: 'var(--amber)' },
};

export default function App() {
  const {
    messages,
    status,
    statusText,
    tokenUsage,
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
  const [contextSize, setContextSize] = useState(() => Number(localStorage.getItem('ollama_context_size')) || 32_768);
  const [thinkingMode, setThinkingMode] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [folderOpenState, setFolderOpenState] = useState<Record<string, boolean>>({});

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

  // Layout states: Sidebar (left) and Panel (right) are toggleable.
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState<RightTab>('terminal');

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

  // Load selected session history and restore its active persona and model.
  useEffect(() => {
    let cancelled = false;
    const requestedPersona = newSessionPersonas.current.get(activeSessionId);
    const requestedModel = newSessionModels.current.get(activeSessionId);
    localStorage.setItem('ollama_active_session', activeSessionId);
    loadSessionHistory(activeSessionId).then((session) => {
      if (!cancelled) {
        setActivePersona(requestedPersona ?? session.persona as PersonaType);
        setModel(requestedModel ?? session.model);
        if (requestedPersona) newSessionPersonas.current.delete(activeSessionId);
        if (requestedModel !== undefined) newSessionModels.current.delete(activeSessionId);
      }
    });
    return () => { cancelled = true; };
  }, [activeSessionId, loadSessionHistory]);

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
        <ModelSelector value={model} onChange={setModel} disabled={isRunning} />

        <div className="flex items-center gap-2" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Context</span>
          <select
            value={contextSize}
            onChange={(event) => setContextSize(Number(event.target.value))}
            disabled={isRunning}
            title="Context window used for the next model run"
            style={{ padding: '5px 8px', fontSize: 12, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', color: 'var(--text-primary)' }}
          >
            {CONTEXT_SIZES.map((size) => <option key={size} value={size}>{size / 1024}k</option>)}
          </select>
        </div>

        <label className="flex items-center gap-2" style={{ fontSize: 12, color: 'var(--text-secondary)', cursor: isRunning ? 'not-allowed' : 'pointer' }} title="Use model reasoning when supported">
          <input type="checkbox" checked={thinkingMode} onChange={(event) => setThinkingMode(event.target.checked)} disabled={isRunning} />
          Thinking mode
        </label>

        {/* Persona selector */}
        <div className="flex items-center gap-2" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Persona</span>
          <select
            id="persona-selector"
            value={activePersona}
            onChange={(e) => setActivePersona(e.target.value as PersonaType)}
            disabled={isRunning || messages.length > 0} // lock persona once conversation starts
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
            <option value="creative">✍ Creative Novelist</option>
          </select>
        </div>

        {/* Status */}
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 12, justifyContent: 'center' }}>
          <StatusIndicator status={status} statusText={statusText} />
        </div>

        {/* Token bar */}
        <ProgressBar input={tokenUsage.input} output={tokenUsage.output} />

        {/* Layout controls */}
        <div className="flex items-center gap-1" style={{ borderLeft: '1px solid var(--border)', paddingLeft: 12, marginLeft: 4 }}>
          {/* Toggle Left Sidebar */}
          <button
            className="btn"
            style={{
              padding: '5px 10px',
              fontSize: 12,
              background: sidebarOpen ? 'var(--accent-glow)' : 'transparent',
              borderColor: sidebarOpen ? 'var(--accent)' : 'var(--border)',
              color: sidebarOpen ? 'var(--accent-light)' : 'var(--text-secondary)',
            }}
            onClick={() => setSidebarOpen((o) => !o)}
            title="Toggle Chats List"
          >
            💬 Chats
          </button>

          {/* Toggle Right Panel */}
          <button
            className="btn"
            style={{
              padding: '5px 10px',
              fontSize: 12,
              background: rightPanelOpen ? 'var(--accent-glow)' : 'transparent',
              borderColor: rightPanelOpen ? 'var(--accent)' : 'var(--border)',
              color: rightPanelOpen ? 'var(--accent-light)' : 'var(--text-secondary)',
            }}
            onClick={() => setRightPanelOpen((o) => !o)}
            title="Toggle Workspace Tools"
          >
            ⊞ Workspace Tools
          </button>
        </div>
      </header>

      {/* ── Main Body ── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* Left Sidebar — Collapsible Chats Tabs List */}
        {sidebarOpen && (
          <aside style={{
            width: 230,
            flexShrink: 0,
            borderRight: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
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
                {(['coder', 'researcher', 'creative'] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => startNewChat(p)}
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
                    title={`Create new ${p} chat`}
                  >
                    {p === 'coder' && '💻'}
                    {p === 'researcher' && '🔍'}
                    {p === 'creative' && '✍'}
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
          </aside>
        )}

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
              sidebarOpen={sidebarOpen}
              rightPanelOpen={rightPanelOpen}
              onToggleSidebar={() => setSidebarOpen((o) => !o)}
              onToggleRightPanel={() => setRightPanelOpen((o) => !o)}
            />
          </div>

          {/* Right Panel: Collapsible Editor, Terminal & Files */}
          {rightPanelOpen && (
            <div style={{
              width: '50%',
              minWidth: 400,
              display: 'flex',
              flexDirection: 'column',
              borderLeft: '1px solid var(--border)',
              overflow: 'hidden',
              background: 'var(--bg-surface)',
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
              </div>
            </div>
          )}
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
