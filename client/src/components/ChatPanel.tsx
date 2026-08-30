import React, { useEffect, useRef, useState, useCallback, KeyboardEvent } from 'react';
import { MessageBubble } from './MessageBubble';
import { StoryStudio } from './StoryStudio';
import type { ChatMessage, AgentStatus, SendMessageOptions, InferenceBackend, OutgoingAttachment } from '../hooks/useOllamaStream';
import type { CloudProviderSettings } from '../lib/providerConfig';

const IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
const TEXT_MIME_TYPES = ['text/csv', 'text/plain'];
const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
const TEXT_EXTS = ['csv', 'txt'];
const MAX_ATTACHMENTS = 8;

interface PendingAttachment {
  id: string;
  name: string;
  mimeType: string;
  kind: 'image' | 'text';
  /** base64 (image) or raw text (text) — matches OutgoingAttachment.data. */
  data: string;
  /** data URL for the image thumbnail. */
  previewUrl?: string;
}

function newId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function extOf(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.slice(dot + 1).toLowerCase() : '';
}

function classifyFile(file: File): 'image' | 'text' | null {
  const ext = extOf(file.name);
  if (IMAGE_MIME_TYPES.includes(file.type) || IMAGE_EXTS.includes(ext)) return 'image';
  if (TEXT_MIME_TYPES.includes(file.type) || TEXT_EXTS.includes(ext)) return 'text';
  return null;
}

function normalizeMime(file: File, kind: 'image' | 'text'): string {
  if (kind === 'image') {
    if (IMAGE_MIME_TYPES.includes(file.type)) return file.type;
    switch (extOf(file.name)) {
      case 'jpg':
      case 'jpeg': return 'image/jpeg';
      case 'bmp': return 'image/bmp';
      case 'webp': return 'image/webp';
      default: return 'image/png';
    }
  }
  if (TEXT_MIME_TYPES.includes(file.type)) return file.type;
  return extOf(file.name) === 'csv' ? 'text/csv' : 'text/plain';
}

function extForImageMime(mime: string): string {
  switch (mime) {
    case 'image/jpeg': return 'jpg';
    case 'image/bmp': return 'bmp';
    case 'image/webp': return 'webp';
    default: return 'png';
  }
}

// Pasted/loaded images arrive as data URLs; keep only the base64 payload so the
// server can Buffer.from(..., 'base64') it directly.
function readImageBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result ?? '');
      const comma = result.indexOf(',');
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function readTextContent(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

interface Props {
  model: string;
  sessionId: string;
  messages: ChatMessage[];
  status: AgentStatus;
  sendMessage: (content: string, model: string, sessionId: string, persona: string, options?: SendMessageOptions) => void;
  rerunFrom: (sessionId: string, timestamp: number, model: string, persona: string, options?: SendMessageOptions) => Promise<void>;
  stop: (sessionId: string) => void;
  killModel: (model: string, inferenceBackend: InferenceBackend) => Promise<void>;
  resetSession: (sessionId: string) => void;
  persona: string;
  contextSize: number;
  thinkingMode: boolean;
  numThread: number;
  inferenceBackend: InferenceBackend;
  cloudProvider?: CloudProviderSettings;
  cavemanMode: boolean;
  /** When true, installs run without per-command approval for the run. */
  autopilotInstalls: boolean;
  /** Selected model reports vision: true in model_map — enables image attach/paste. */
  visionSupported: boolean;
  /** Configured vision sidecar model id (model_map roles.vision). When set, a
   *  non-vision coder can still take images: the backend describes them with
   *  this model and hands the text to the coder. */
  visionSidecarModel?: string;
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
  rerunFrom,
  stop,
  killModel,
  resetSession,
  persona,
  contextSize,
  thinkingMode,
  numThread,
  inferenceBackend,
  cloudProvider,
  cavemanMode,
  autopilotInstalls,
  visionSupported,
  visionSidecarModel,
  sidebarOpen,
  rightPanelOpen,
  hasNovelOutline,
  onStartChapter,
  onToggleSidebar,
  onToggleRightPanel,
}: Props) {
  const [input, setInput] = useState('');
  const [storyMode, setStoryMode] = useState(false);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [visionWarning, setVisionWarning] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isRunning = status === 'thinking' || status === 'tool';
  const showStartChapter = persona === 'novelist' && hasNovelOutline;

  // Text files (csv/txt) work with any model. Images work when the selected
  // model has native vision OR a vision sidecar is configured (roles.vision) —
  // in the latter case the backend describes the image and hands the text to
  // the non-vision coder. The picker's accept list mirrors that gate.
  const usesVisionSidecar = !visionSupported && Boolean(visionSidecarModel);
  const imagesAllowed = visionSupported || Boolean(visionSidecarModel);
  const acceptAttr = imagesAllowed
    ? '.csv,.txt,.jpg,.jpeg,.png,.bmp,.webp,text/csv,text/plain,image/jpeg,image/png,image/bmp,image/webp'
    : '.csv,.txt,text/csv,text/plain';

  const addFiles = useCallback(async (files: File[]) => {
    const collected: PendingAttachment[] = [];
    let blockedImage = false;
    for (const file of files) {
      const kind = classifyFile(file);
      if (!kind) continue;
      if (kind === 'image' && !imagesAllowed) { blockedImage = true; continue; }
      const mimeType = normalizeMime(file, kind);
      const name = file.name && file.name.trim()
        ? file.name
        : `pasted-${Date.now()}.${extForImageMime(mimeType)}`;
      try {
        if (kind === 'image') {
          const data = await readImageBase64(file);
          collected.push({ id: newId(), name, mimeType, kind, data, previewUrl: `data:${mimeType};base64,${data}` });
        } else {
          const data = await readTextContent(file);
          collected.push({ id: newId(), name, mimeType, kind, data });
        }
      } catch {
        // Skip a file that fails to read rather than blocking the others.
      }
    }
    if (blockedImage) {
      setVisionWarning('No vision available — image attachments were skipped. Select a vision model or configure a vision sidecar (roles.vision) to send images.');
    } else if (usesVisionSidecar && collected.some((c) => c.kind === 'image')) {
      setVisionWarning(`Selected model can't see images; ${visionSidecarModel} will describe them and pass the text to it.`);
    }
    if (collected.length) setAttachments((cur) => [...cur, ...collected].slice(0, MAX_ATTACHMENTS));
  }, [imagesAllowed, usesVisionSidecar, visionSidecarModel]);

  const onFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) void addFiles(Array.from(e.target.files));
    e.target.value = '';
  }, [addFiles]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((cur) => cur.filter((a) => a.id !== id));
  }, []);

  const handlePaste = useCallback((e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const files: File[] = [];
    for (const item of Array.from(items)) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    if (!files.length) return;
    // Don't drop a pasted screenshot into the textarea as junk; consume it.
    e.preventDefault();
    if (!imagesAllowed) {
      setVisionWarning('No vision available — pasted image ignored. Select a vision model or configure a vision sidecar (roles.vision) to send images.');
      return;
    }
    void addFiles(files);
  }, [imagesAllowed, addFiles]);

  // Clear the warning once a vision model is active; auto-dismiss otherwise.
  useEffect(() => {
    if (visionSupported) setVisionWarning(null);
  }, [visionSupported]);
  useEffect(() => {
    if (!visionWarning) return;
    const t = setTimeout(() => setVisionWarning(null), 6000);
    return () => clearTimeout(t);
  }, [visionWarning]);

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
    if (!model || isRunning) return;
    const sendableAttachments = imagesAllowed
      ? attachments
      : attachments.filter((attachment) => attachment.kind !== 'image');
    if (sendableAttachments.length !== attachments.length) {
      setVisionWarning('No vision available — image attachments were skipped. Select a vision model or configure a vision sidecar (roles.vision) to send images.');
      setAttachments(sendableAttachments);
    }
    if (!text && sendableAttachments.length === 0) return;
    setInput('');
    const outgoing: OutgoingAttachment[] = sendableAttachments.map((a) => ({
      name: a.name,
      mimeType: a.mimeType,
      kind: a.kind,
      data: a.data,
    }));
    setAttachments([]);
    sendMessage(text, model, sessionId, persona, {
      numCtx: contextSize,
      think: thinkingMode,
      numThread,
      inferenceBackend,
      caveman: cavemanMode,
      autopilotInstalls,
      cloudProvider,
      ...(outgoing.length ? { attachments: outgoing } : {}),
    });
  }, [input, attachments, isRunning, model, sessionId, sendMessage, persona, contextSize, thinkingMode, numThread, inferenceBackend, cavemanMode, autopilotInstalls, cloudProvider, imagesAllowed]);

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
            autopilotInstalls,
            cloudProvider,
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
              onRerun={msg.role === 'user' && !isRunning ? () => {
                void rerunFrom(sessionId, msg.timestamp, model, persona, {
                  numCtx: contextSize,
                  think: thinkingMode,
                  numThread,
                  inferenceBackend,
                  caveman: cavemanMode,
                  autopilotInstalls,
                  cloudProvider,
                });
              } : undefined}
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
          flexDirection: 'column',
          gap: 8,
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          padding: '10px 14px',
          transition: 'border-color 0.2s, box-shadow 0.2s',
        }}>
          {visionWarning && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '6px 10px', fontSize: 12,
              color: 'var(--amber)', background: 'rgba(245,166,35,0.12)',
              border: '1px solid rgba(245,166,35,0.4)', borderRadius: 'var(--radius-md)',
            }}>
              <span>⚠</span>
              <span style={{ flex: 1 }}>{visionWarning}</span>
              <button
                onClick={() => setVisionWarning(null)}
                title="Dismiss"
                style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: 14, lineHeight: 1, padding: 0 }}
              >
                ×
              </button>
            </div>
          )}
          {attachments.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {attachments.map((att) => (
                <div key={att.id} style={{
                  position: 'relative',
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: att.previewUrl ? 0 : '4px 8px',
                  paddingRight: att.previewUrl ? 0 : 22,
                  background: 'var(--bg-hover)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  maxWidth: 180,
                  overflow: 'hidden',
                }}>
                  {att.previewUrl ? (
                    <img
                      src={att.previewUrl}
                      alt={att.name}
                      style={{ width: 46, height: 46, objectFit: 'cover', display: 'block', borderRadius: 'var(--radius-md)' }}
                    />
                  ) : (
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      📎 {att.name}
                    </span>
                  )}
                  <button
                    onClick={() => removeAttachment(att.id)}
                    title="Remove attachment"
                    style={{
                      position: 'absolute', top: 2, right: 2,
                      width: 16, height: 16, borderRadius: '50%',
                      border: 'none', background: 'rgba(0,0,0,0.6)', color: '#fff',
                      fontSize: 11, lineHeight: '15px', textAlign: 'center',
                      cursor: 'pointer', padding: 0,
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <div style={{ display: 'flex', gap: 10 }}>
            <button
              className="btn btn-icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={isRunning}
              title={imagesAllowed
                ? (usesVisionSidecar
                    ? `Attach files (csv, txt, images). Images are described by ${visionSidecarModel} and passed to this non-vision model.`
                    : 'Attach files (csv, txt, jpg, png, bmp, webp)')
                : 'Attach text files (csv, txt). No vision model or sidecar available for images.'}
              style={{ alignSelf: 'flex-end' }}
            >
              +
            </button>
          <textarea
            ref={textareaRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            onPaste={handlePaste}
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
            disabled={(!input.trim() && attachments.length === 0) || !model}
                title="Send (Enter)"
              >
                ▲
              </button>
            )}
            {!cloudProvider && (
              <button
                className="btn btn-icon"
                onClick={() => void killModel(model, inferenceBackend)}
                disabled={!model}
                title="Eject model from memory"
                style={{ color: 'var(--amber)', borderColor: 'rgba(245,166,35,0.45)' }}
              >
                ⚡
              </button>
            )}
          </div>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept={acceptAttr}
          multiple
          style={{ display: 'none' }}
          onChange={onFileInputChange}
        />

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
