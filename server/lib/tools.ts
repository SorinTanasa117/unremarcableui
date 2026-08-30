import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const WORKSPACE_DIR = path.resolve(__dirname, '../../agent-workspace');

// Tool definitions sent to Ollama
export const TOOL_DEFINITIONS = [
  {
    type: 'function',
    function: {
      name: 'web_search',
      description: 'Search the web using Brave Search. If Brave explicitly reports exhausted credits, fall back to DuckDuckGo. Return titles, URLs, and snippets.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The search query' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'browse_url',
      description: 'Browse a URL with a headless browser and return the visible text content of the page.',
      parameters: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'The URL to visit' },
        },
        required: ['url'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Create a NEW file inside agent-workspace. Never overwrites an existing file. To change existing code, read the relevant chunk and use edit_file for one exact function/component-sized replacement. Compacted-history placeholders are not file content.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace (e.g. src/app.py)' },
          content: { type: 'string', description: 'The file content to write' },
        },
        required: ['filepath', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'edit_file',
      description: 'Fix or change part of an existing file WITHOUT rewriting the whole file. Replaces one exact, unique occurrence of old_str with new_str; CRLF/LF differences are accepted. No automatic write_file fallback follows failures. For functional rewrites, replace one function/component at a time.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace' },
          old_str: { type: 'string', description: 'Exact text to replace. Must match once and be unique; include surrounding context if needed.' },
          new_str: { type: 'string', description: 'Replacement text.' },
        },
        required: ['filepath', 'old_str', 'new_str'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read a file inside agent-workspace in CHUNKS. Returns numbered lines for the requested window (default 100 lines starting at offset). Page through large files with offset/limit instead of dumping everything. An explicit limit of 1-100 on a recovery-locked file unlocks smaller edit_file retries; full-overwrite protection remains until an edit succeeds. limit=0 reads the whole file.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace' },
          offset: { type: 'number', description: '1-based line number to start reading from (default 1)' },
          limit: { type: 'number', description: 'Max lines to return (default 100). Use 0 only for whole-file reads of small files.' },
        },
        required: ['filepath'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_terminal',
      description: 'Run validation, inspection, build, or server commands inside agent-workspace (Windows/cmd.exe) and return stdout+stderr. File-mutation shell commands are blocked; create files with write_file and change existing files with edit_file. Do NOT start a long-running server here as a normal command or chain it with "&" — it will be auto-detached.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The command to run (e.g. "npm install"). Windows cmd.exe syntax.' },
        },
        required: ['command'],
      },
    },
  },
];
