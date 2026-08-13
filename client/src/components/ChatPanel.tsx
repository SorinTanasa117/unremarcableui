import React, { useEffect, useRef, useState, useCallback, KeyboardEvent } from 'react';
import { MessageBubble } from './MessageBubble';
import { StoryStudio } from './StoryStudio';
import type { ChatMessage, AgentStatus, SendMessageOptions, InferenceBackend } from '../hooks/useOllamaStream';

interface Props {
  model: string;
  sessionId: string;
  messages: ChatMessage[];
  status: AgentStatus;
  sendMessage: (content: string, model: string, sessionId: string, persona: string, options?: SendMessageOptions) => void;
  stop: (sessionId: string) => void;
  killModel: (model: string, inferenceBackend: InferenceBackend) => Promise<void>;
  resetSession: (sessionId: string) => void;
  persona: string;
  contextSize: number;
  thinkingMode: boolean;
  numThread: number;
  inferenceBackend: InferenceBackend;
  cavemanMode: boolean;
  sidebarOpen: boolean;
  rightPanelOpen: boolean;
  hasNovelOutline: boolean;
  onStartChapter: () => void;
  onToggleSidebar: () => void;
  onToggleRightPanel: () => void;
}

export function ChatPanel({
  model,
  sessionId,
  messages,
  status,
  sendMessage,
  stop,
  killModel,
  resetSession,
  persona,
  contextSize,
  thinkingMode,
  numThread,
  inferenceBackend,
  cavemanMode,
  sidebarOpen,
  rightPanelOpen,
  hasNovelOutline,
  onStartChapter,
  onToggleSidebar,
  onToggleRightPanel,
}: Props) {
  const [input, setInput] = useState('');
  const [storyMode, setStoryMode] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isRunning = status === 'thinking' || status === 'tool';
  const showStartChapter = persona === 'novelist' && hasNovelOutline;

  useEffect(() => {
    if (persona !== 'creative') setStoryMode(false);
  }, [persona]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [input]);

  const submit = useCallback(() => {
    const text = input.trim();
    if (!text || !model || isRunning) return;
    setInput('');
    sendMessage(text, model, sessionId, persona, {
      numCtx: contextSize,
      think: thinkingMode,
      numThread,
      inferenceBackend,
      caveman: cavemanMode,
    });
  }, [input, isRunning, model, sessionId, sendMessage, persona, contextSize, thinkingMode, numThread, inferenceBackend, cavemanMode]);

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex-col" style={{ height: '100%', display: 'flex', position: 'relative' }}>

      {persona === 'creative' && (
        <div className="flex items-center justify-between" style={{ padding: '7px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{storyMode ? 'Story Studio' : 'Creative tools'}</span>
          <button
            className="btn"
            onClick={() => setStoryMode((enabled) => !enabled)}
            style={{ padding: '4px 9px', fontSize: 11, background: storyMode ? 'var(--accent-glow)' : 'transparent', color: storyMode ? 'var(--accent-light)' : 'var(--text-secondary)' }}
            aria-pressed={storyMode}
          >
            {storyMode ? '✦ Story Mode on' : '✦ Story Mode'}
          </button>
        </div>
      )}
      {persona === 'creative' && storyMode ? (
        <StoryStudio
          sessionId={sessionId}
          messages={messages}
          status={status}
          modelAvailable={Boolean(model)}
          onGenerate={(prompt) => sendMessage(prompt, model, sessionId, persona, {
            isolated: true,
            numCtx: contextSize,
            think: thinkingMode,
            numThread,
            inferenceBackend,
            caveman: cavemanMode,
          })}
        />
      ) : <>
      {/* Messages */}
      <div
        className="flex-1 overflow-auto"
        style={{ padding: '20px 16px', display: 'flex', flexDirection: 'column', gap: 4 }}
      >
        {messages.length === 0 && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 12,
            color: 'var(--text-muted)',
          }}>
            <div style={{
              width: 56, height: 56,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, var(--accent) 0%, var(--cyan) 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 28, boxShadow: '0 0 30px var(--accent-glow)',
            }}>
              ✦
            </div>
            <p style={{ fontSize: 15, fontWeight: 500, color: 'var(--text-secondary)' }}>
              Ollama Agent
            </p>
            <p style={{ fontSize: 13, maxWidth: 320, textTransform: 'capitalize', color: 'var(--accent-light)', fontWeight: 600 }}>
              Mode: {persona}
            </p>
            <p style={{ fontSize: 13, maxWidth: 320, textAlign: 'center', lineHeight: 1.6 }}>
              {persona === 'coder' && 'High-efficiency software engineering workspace. Focused on writing robust, production-quality code.'}
              {persona === 'researcher' && 'Systematic data intelligence researcher. Ready to search, compile, and cross-reference fact data.'}
              {persona === 'creative' && 'Creative writing environment specializing in atmospheric gothic fantasy, suspense mystery, and drama.'}
            </p>
          </div>
        )}

        {messages.map((msg, i) => {
          const isLastAssistant = msg.role === 'assistant' &&
            [...messages].reverse().find((m) => m.role === 'assistant')?.id === msg.id;
          return (
            <MessageBubble
              key={msg.id}
              message={msg}
              isStreaming={isRunning && isLastAssistant}
            />
          );
        })}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div style={{
        padding: '12px 16px',
        borderTop: '1px solid var(--border)',
        background: 'var(--bg-surface)',
      }}>
        <div style={{
          display: 'flex',
          gap: 10,
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          padding: '10px 14px',
          transition: 'border-color 0.2s, box-shadow 0.2s',
        }}>
          <textarea
            ref={textareaRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={`Message the agent (${persona} mode)… (Shift+Enter for newline)`}
            rows={1}
            disabled={isRunning}
            style={{
              flex: 1,
              background: 'none',
              border: 'none',
              outline: 'none',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: 14,
              resize: 'none',
              lineHeight: 1.6,
              padding: 0,
              boxShadow: 'none',
            }}
          />
          <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end', gap: 6 }}>
            {isRunning ? (
              <button
                className="btn btn-danger btn-icon"
                onClick={() => stop(sessionId)}
                title="Stop agent"
              >
                ■
              </button>
            ) : showStartChapter ? (
              <button
                className="btn btn-primary"
                onClick={onStartChapter}
                disabled={!model}
                title="Start first unwritten chapter"
                style={{ whiteSpace: 'nowrap' }}
              >
                ▶ Start chapter
              </button>
            ) : (
              <button
                className="btn btn-primary btn-icon"
                onClick={submit}
            disabled={!input.trim() || !model}
                title="Send (Enter)"
              >
                ▲
              </button>
            )}
            <button
              className="btn btn-icon"
              onClick={() => void killModel(model, inferenceBackend)}
              disabled={!model}
              title="Eject model from memory"
              style={{ color: 'var(--amber)', borderColor: 'rgba(245,166,35,0.45)' }}
            >
              ⚡
            </button>
          </div>
        </div>

        <div style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', marginTop: 8, padding: '0 4px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              {model || 'No model selected'}
            </span>
          </div>
          <button
            style={{
              background: 'none', border: 'none', color: 'var(--text-muted)',
              fontSize: 11, cursor: 'pointer', padding: '2px 6px',
              borderRadius: 'var(--radius-sm)',
            }}
            onClick={() => resetSession(sessionId)}
            title="Clear conversation"
          >
            Clear
          </button>
        </div>
      </div>
      </>}
    </div>
  );
}
