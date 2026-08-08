import { Router, Request, Response } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { WORKSPACE_DIR } from '../lib/tools.js';

const router = Router();

// Ensure workspace exists
await fs.mkdir(WORKSPACE_DIR, { recursive: true });

function safePath(rel: string): string | null {
  const resolved = path.resolve(WORKSPACE_DIR, rel);
  if (!resolved.startsWith(WORKSPACE_DIR)) return null;
  return resolved;
}

interface FileTreeNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileTreeNode[];
  size?: number;
}

async function buildTree(dir: string, rel: string = ''): Promise<FileTreeNode[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const nodes: FileTreeNode[] = [];
  for (const entry of entries) {
    const entryRel = rel ? `${rel}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      const children = await buildTree(path.join(dir, entry.name), entryRel);
      nodes.push({ name: entry.name, path: entryRel, type: 'directory', children });
    } else {
      const stat = await fs.stat(path.join(dir, entry.name));
      nodes.push({ name: entry.name, path: entryRel, type: 'file', size: stat.size });
    }
  }
  return nodes.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

// GET /api/files/tree
router.get('/tree', async (_req: Request, res: Response) => {
  try {
    const tree = await buildTree(WORKSPACE_DIR);
    res.json(tree);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/files/read?path=...
router.get('/read', async (req: Request, res: Response) => {
  const rel = req.query.path as string;
  const target = safePath(rel);
  if (!target) { res.status(400).json({ error: 'Invalid path' }); return; }
  try {
    const content = await fs.readFile(target, 'utf-8');
    res.json({ content });
  } catch (err: any) {
    res.status(404).json({ error: err.message });
  }
});

// POST /api/files/write
router.post('/write', async (req: Request, res: Response) => {
  const { path: rel, content } = req.body as { path: string; content: string };
  const target = safePath(rel);
  if (!target) { res.status(400).json({ error: 'Invalid path' }); return; }
  try {
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, content, 'utf-8');
    res.json({ ok: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/files/delete
router.delete('/delete', async (req: Request, res: Response) => {
  const rel = req.query.path as string;
  const target = safePath(rel);
  if (!target) { res.status(400).json({ error: 'Invalid path' }); return; }
  try {
    await fs.rm(target, { recursive: true, force: true });
    res.json({ ok: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
