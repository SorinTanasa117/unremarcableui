/**
 * Focused regression smoke test for write_file result metadata.
 *
 * Run:  npx tsx server/scripts/testFileWrite.ts
 *
 * Proves a successful write_file reports byte + line counts so the agent can
 * distinguish a real source file from a tiny placeholder / truncated write.
 * Does not touch any app under agent-workspace beyond its own temp folder,
 * which it removes on exit.
 */
import fs from 'fs/promises';
import path from 'path';
import { runTool } from '../lib/toolRunner.js';
import { WORKSPACE_DIR } from '../lib/tools.js';

let failures = 0;
function check(label: string, cond: boolean) {
  if (cond) {
    console.log(`  ok  - ${label}`);
  } else {
    failures += 1;
    console.error(`  FAIL- ${label}`);
  }
}

async function main() {
  const dir = `._writetest_${Date.now()}`;
  const sessionId = `writetest-${Date.now()}`;
  const substantivePath = `${dir}/substantive.txt`;
  const placeholderPath = `${dir}/placeholder.txt`;
  const substantiveContent = 'line1\nline2\nline3'; // 17 bytes, 3 lines
  const placeholderContent = '/* init */'; // 10 bytes, 1 line

  try {
    const r1 = await runTool('write_file', { filepath: substantivePath, content: substantiveContent }, sessionId);
    check('substantive write succeeded', r1.success);
    check('reports byte count', r1.output.includes('17 bytes'));
    check('reports line count', r1.output.includes('3 lines'));

    const r2 = await runTool('write_file', { filepath: placeholderPath, content: placeholderContent }, sessionId);
    check('placeholder write succeeded', r2.success);
    check('placeholder reports small size', r2.output.includes('10 bytes') && r2.output.includes('1 lines'));
    check('placeholder size differs from substantive', !r2.output.includes('17 bytes'));
  } finally {
    try {
      await fs.rm(path.resolve(WORKSPACE_DIR, dir), { recursive: true, force: true });
    } catch {}
  }

  if (failures > 0) {
    console.error(`\n${failures} check(s) failed.`);
    process.exit(1);
  }
  console.log('\nAll write_file metadata checks passed.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
