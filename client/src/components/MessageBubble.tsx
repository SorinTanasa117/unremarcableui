import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../hooks/useOllamaStream';

interface Props {
  message: ChatMessage;
  isStreaming?: boolean;
}

const toolIcons: Record<string, string> = {
  web_search: '🔍',
  browse_url: '🌐',
  write_file: '📝',
  read_file: '📄',
  run_terminal: '💻',
};

function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  return date.toTimeString().split(' ')[0]; // HH:MM:SS format
}

function formatDuration(ms?: number): string {
  if (ms === undefined) return '';
  const seconds = Math.floor(ms / 1000);
  if (seconds < 1) return '<1s';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  return [hours && `${hours}h`, minutes && `${minutes}m`, `${remainingSeconds}s`]
    .filter(Boolean)
    .join(' ');
}

// Regex to strip raw chat template control tokens emitted by buggy model tags
const TEMPLATE_TOKENS_REGEX = /<assistant>|<\/assistant>|<user>|<\/user>|<system>|<\/system>|<prompt>|<\/prompt>|\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>/gi;

// Trailing partial control tokens that commonly leak at the end of streams
const TRAILING_PARTIAL_TOKENS = [
  /<\/assistant_?r?e?s?p?o?n?s?e?$/i,
  /<\/assistant?$/i,
  /<\/assistan$/i,
  /<\/assista$/i,
  /<\/assist$/i,
  /<\/assis$/i,
  /<\/ass$/i,
  /<\/as$/i,
  /<\/a$/i,
  /<\/user?$/i,
  /<\/system?$/i,
  /\[\/inst?$/i,
  /<\|im_end?$/i,
  /<\|im_?$/i,
  /<\|endoftext?$/i,
];

function cleanModelOutput(text: string): string {
  if (!text) return '';
  let cleaned = text.replace(TEMPLATE_TOKENS_REGEX, '');
  for (const regex of TRAILING_PARTIAL_TOKENS) {
    cleaned = cleaned.replace(regex, '');
  }
  return cleaned;
}

export function MessageBubble({ message, isStreaming }: Props) {
  const { role, content, thinking, toolName, timestamp, durationMs, durationLabel } = message;
  const cleanedContent = cleanModelOutput(content);

  if (role === 'system') {
    return (
      <div className="fade-in flex items-center gap-2" style={{ padding: '6px 0', color: 'var(--text-muted)', fontSize: 12 }}>
        <span style={{ opacity: 0.5 }}>─</span>
        <span>{content}</span>
        <span style={{ opacity: 0.5 }}>─</span>
      </div>
    );
  }

  if (role === 'tool_result') {
    const icon = toolIcons[toolName ?? ''] ?? '⚙';
    const isDone = durationMs !== undefined;

    return (
      <div className="fade-in flex items-center justify-between" style={{
        padding: '6px 12px',
        borderRadius: 'var(--radius-md)',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        fontSize: 12,
        color: 'var(--text-secondary)',
        margin: '4px 0',
      }}>
        <div className="flex items-center gap-2">
          <span style={{ color: 'var(--amber)' }}>{icon}</span>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleanedContent}</ReactMarkdown>
        </div>
        <div className="flex items-center gap-2" style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          <span>{formatTime(timestamp)}</span>
          {isDone && (
            <span style={{
              background: 'var(--bg-hover)',
              padding: '1px 5px',
              borderRadius: 3,
              color: 'var(--amber)',
              border: '1px solid rgba(245,166,35,0.15)',
            }}>
              {formatDuration(durationMs)}
            </span>
          )}
        </div>
      </div>
    );
  }

  const isUser = role === 'user';

  return (
    <div
      className="fade-in"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        padding: '6px 0',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start' }}>
        {!isUser && (
          <div style={{
            width: 28, height: 28,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--accent), var(--cyan))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, flexShrink: 0, marginRight: 10, marginTop: 2,
          }}>
            ✦
          </div>
        )}
        <div style={{
          padding: isUser ? '10px 14px' : '12px 16px',
          borderRadius: isUser
            ? 'var(--radius-lg) var(--radius-lg) var(--radius-sm) var(--radius-lg)'
            : 'var(--radius-lg) var(--radius-lg) var(--radius-lg) var(--radius-sm)',
          background: isUser
            ? 'linear-gradient(135deg, var(--accent), #5a4fd4)'
            : 'var(--bg-card)',
          border: isUser ? 'none' : '1px solid var(--border)',
          color: 'var(--text-primary)',
          fontSize: 13.5,
          lineHeight: 1.7,
          wordBreak: 'break-word',
        }}>
          {!isUser && thinking && (
            <div style={{
              marginBottom: content ? 10 : 0,
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
                {thinking}
              </pre>
            </div>
          )}
          {isUser ? (
            <span style={{ whiteSpace: 'pre-wrap' }}>{cleanedContent}</span>
          ) : (
            <div className="prose" style={{ maxWidth: '100%' }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleanedContent}</ReactMarkdown>
              {isStreaming && (
                <span style={{
                  display: 'inline-block',
                  width: 2, height: '1em',
                  background: 'var(--accent)',
                  marginLeft: 2,
                  animation: 'pulse-dot 0.7s ease infinite',
                  verticalAlign: 'text-bottom',
                }} />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Timestamp footer for bubbles */}
      <div style={{
        marginTop: 4,
        fontSize: 10,
        color: 'var(--text-muted)',
        fontFamily: 'var(--font-mono)',
        display: 'flex',
        gap: 6,
        paddingLeft: isUser ? 0 : 38,
        paddingRight: isUser ? 8 : 0,
      }}>
        <span>{formatTime(timestamp)}</span>
        {durationMs !== undefined && (
          <>
            <span>•</span>
            <span style={{ color: 'var(--accent-light)' }}>
              {durationLabel ?? 'Generated in'} {formatDuration(durationMs)}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
