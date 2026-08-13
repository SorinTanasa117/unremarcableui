import React from 'react';
import { useFileSystem } from '../hooks/useFileSystem';

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
  size?: number;
}

interface Props {
  onSelectFile: (path: string) => void;
  selectedPath?: string;
  folderOpenState: Record<string, boolean>;
  onFolderOpenChange: (path: string, open: boolean) => void;
}

function extIcon(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  const icons: Record<string, string> = {
    ts: '🔷', tsx: '🔷', js: '🟨', jsx: '🟨',
    py: '🐍', json: '📋', md: '📝', css: '🎨',
    html: '🌐', sh: '⚡', txt: '📄', rs: '🦀',
    go: '🐹', yml: '⚙', yaml: '⚙', env: '🔑',
  };
  return icons[ext] ?? '📄';
}

function FileNode({ node, depth, onSelect, selectedPath, folderOpenState, onFolderOpenChange }: {
  node: FileNode;
  depth: number;
  onSelect: (p: string) => void;
  selectedPath?: string;
  folderOpenState: Record<string, boolean>;
  onFolderOpenChange: (path: string, open: boolean) => void;
}) {
  const isDir = node.type === 'directory';
  // Default collapsed: folders stay closed unless the user has explicitly
  // toggled them. Their open/closed state is owned by App.tsx (so it survives
  // tab switches between Editor/Terminal/Files/Novel) and is also persisted
  // to localStorage so a page reload restores the user's last view.
  const open = isDir ? (folderOpenState[node.path] ?? false) : false;
  const isSelected = node.path === selectedPath;

  return (
    <div>
      <div
        className="flex items-center gap-1"
        style={{
          paddingLeft: depth * 14 + 8,
          paddingRight: 8,
          paddingTop: 3,
          paddingBottom: 3,
          cursor: 'pointer',
          borderRadius: 'var(--radius-sm)',
          background: isSelected ? 'var(--accent-glow)' : 'transparent',
          color: isSelected ? 'var(--accent-light)' : 'var(--text-primary)',
          fontSize: 12.5,
          transition: 'background 0.15s',
          userSelect: 'none',
        }}
        onMouseEnter={(e) => { if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-hover)'; }}
        onMouseLeave={(e) => { if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'transparent'; }}
        onClick={() => {
          if (isDir) onFolderOpenChange(node.path, !open);
          else onSelect(node.path);
        }}
      >
        <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
          {isDir ? (open ? '▾' : '▸') : ' '}
        </span>
        <span style={{ flexShrink: 0 }}>
          {isDir ? (open ? '📂' : '📁') : extIcon(node.name)}
        </span>
        <span className="truncate">{node.name}</span>
        {!isDir && node.size !== undefined && (
          <span style={{ color: 'var(--text-muted)', fontSize: 10, marginLeft: 'auto', flexShrink: 0 }}>
            {node.size < 1024 ? `${node.size}B` : `${(node.size / 1024).toFixed(1)}k`}
          </span>
        )}
      </div>
      {isDir && open && node.children?.map((child) => (
        <FileNode
          key={child.path}
          node={child}
          depth={depth + 1}
          onSelect={onSelect}
          selectedPath={selectedPath}
          folderOpenState={folderOpenState}
          onFolderOpenChange={onFolderOpenChange}
        />
      ))}
    </div>
  );
}

export function FileExplorer({ onSelectFile, selectedPath, folderOpenState, onFolderOpenChange }: Props) {
  const { tree, isLoading, refetch } = useFileSystem();

  return (
    <div className="flex-col" style={{ height: '100%', display: 'flex' }}>
      <div className="flex items-center justify-between" style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)' }}>
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
          Workspace
        </span>
        <button
          className="btn btn-icon"
          style={{ width: 24, height: 24, padding: 0, fontSize: 13, border: 'none', background: 'none', color: 'var(--text-muted)' }}
          onClick={() => refetch()}
          title="Refresh"
        >
          ↻
        </button>
      </div>
      <div className="flex-1 overflow-auto" style={{ padding: '6px 4px' }}>
        {isLoading && (
          <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12 }}>Loading…</div>
        )}
        {!isLoading && tree.length === 0 && (
          <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>
            Agent workspace is empty.<br />
            Ask the agent to create files.
          </div>
        )}
        {tree.map((node: FileNode) => (
          <FileNode
            key={node.path}
            node={node}
            depth={0}
            onSelect={onSelectFile}
            selectedPath={selectedPath}
            folderOpenState={folderOpenState}
            onFolderOpenChange={onFolderOpenChange}
          />
        ))}
      </div>
    </div>
  );
}
