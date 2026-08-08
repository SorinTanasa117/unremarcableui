import React, { useEffect, useState, useRef } from 'react';
import { EditorView, basicSetup } from 'codemirror';
import { EditorState } from '@codemirror/state';
import { oneDark } from '@codemirror/theme-one-dark';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { json } from '@codemirror/lang-json';
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';
import { markdown } from '@codemirror/lang-markdown';
import { rust } from '@codemirror/lang-rust';
import { useFileSystem } from '../hooks/useFileSystem';

interface Props {
  filePath: string | null;
}

function getLanguage(path: string) {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  switch (ext) {
    case 'ts': case 'tsx': case 'js': case 'jsx': return javascript({ typescript: true, jsx: true });
    case 'py': return python();
    case 'json': return json();
    case 'css': return css();
    case 'html': return html();
    case 'md': return markdown();
    case 'rs': return rust();
    default: return javascript();
  }
}

export function FileEditor({ filePath }: Props) {
  const { readFile, writeFile } = useFileSystem(-1); // no auto-polling in editor
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load file when path changes
  useEffect(() => {
    if (!filePath) { setContent(null); return; }
    setError(null);
    readFile(filePath).then(setContent).catch((e) => setError(e.message));
  }, [filePath]);

  // Setup/update CodeMirror
  useEffect(() => {
    if (!containerRef.current || content === null || !filePath) return;

    viewRef.current?.destroy();

    const lang = getLanguage(filePath);
    const state = EditorState.create({
      doc: content,
      extensions: [
        basicSetup,
        oneDark,
        lang,
        EditorView.theme({
          '&': { height: '100%', fontFamily: "'JetBrains Mono', monospace", fontSize: '13px' },
          '.cm-scroller': { overflow: 'auto' },
          '&.cm-focused': { outline: 'none' },
          '.cm-gutters': { background: '#0b0d12', borderRight: '1px solid #2a3050' },
        }),
        EditorView.lineWrapping,
      ],
    });

    const view = new EditorView({ state, parent: containerRef.current });
    viewRef.current = view;

    return () => { view.destroy(); viewRef.current = null; };
  }, [content, filePath]);

  const save = async () => {
    if (!filePath || !viewRef.current) return;
    const text = viewRef.current.state.doc.toString();
    setSaving(true);
    try {
      await writeFile(filePath, text);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  if (!filePath) {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-muted)', fontSize: 13, flexDirection: 'column', gap: 8,
      }}>
        <span style={{ fontSize: 32 }}>📄</span>
        <span>Select a file to edit</span>
      </div>
    );
  }

  return (
    <div className="flex-col" style={{ height: '100%', display: 'flex' }}>
      {/* Header */}
      <div className="flex items-center justify-between" style={{ padding: '8px 14px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
          {filePath}
        </span>
        <div className="flex items-center gap-2">
          {error && <span style={{ fontSize: 11, color: 'var(--red)' }}>{error}</span>}
          {saved && <span style={{ fontSize: 11, color: 'var(--green)' }}>✓ Saved</span>}
          <button
            className="btn"
            style={{ fontSize: 12, padding: '4px 12px' }}
            onClick={save}
            disabled={saving}
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>

      {/* Editor */}
      <div ref={containerRef} className="flex-1" style={{ overflow: 'hidden', background: '#0b0d12' }} />
    </div>
  );
}
