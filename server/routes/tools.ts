import { Router, Request, Response } from 'express';
import { runTool } from '../lib/toolRunner.js';


const router = Router();

// POST /api/tools/search
router.post('/search', async (req: Request, res: Response) => {
  const { query } = req.body as { query: string };
  const result = await runTool('web_search', { query });
  res.json(result);
});

// POST /api/tools/browse
router.post('/browse', async (req: Request, res: Response) => {
  const { url } = req.body as { url: string };
  const result = await runTool('browse_url', { url });
  res.json(result);
});

export default router;
