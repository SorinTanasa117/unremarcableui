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
      description: 'Search the web using DuckDuckGo and return a list of results with titles, URLs, and snippets.',
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
      description: 'Write content to a file inside the agent-workspace directory.',
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
      name: 'read_file',
      description: 'Read the content of a file inside the agent-workspace directory.',
      parameters: {
        type: 'object',
        properties: {
          filepath: { type: 'string', description: 'Relative path within agent-workspace' },
        },
        required: ['filepath'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_terminal',
      description: 'Run a shell command inside the agent-workspace directory and return stdout+stderr.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The command to run (e.g. "npm install")' },
        },
        required: ['command'],
      },
    },
  },
];
