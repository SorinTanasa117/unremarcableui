import fs from 'fs/promises';
import path from 'path';

/**
 * Chat attachments: files the user adds through the "+" picker or by pasting
 * an image. Bytes are stored on disk under the session's own attachments
 * folder (never inlined into the session JSON), and only lightweight metadata
 * is persisted on the message. Image bytes are re-read from disk and re-encoded
 * when a request is built, so both the current turn and earlier turns keep
 * their images across a multi-turn vision conversation.
 */

export type AttachmentKind = 'image' | 'text';

/** What the client uploads for one attachment on a single turn. */
export interface IncomingAttachment {
  name: string;
  mimeType: string;
  kind: AttachmentKind;
  /** image: base64 without the data-URI prefix. text: raw UTF-8 text. */
  data: string;
}

/** What is persisted on a message and round-tripped to the client. */
export interface AttachmentMeta {
  name: string;
  mimeType: string;
  kind: AttachmentKind;
  /** Stored filename inside the session attachments dir (basename only). */
  file: string;
}

// "web images" in the request = WebP. Only these image types are accepted.
const IMAGE_MIME_EXT: Record<string, string> = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/bmp': 'bmp',
  'image/webp': 'webp',
};
const TEXT_MIME_EXT: Record<string, string> = {
  'text/csv': 'csv',
  'text/plain': 'txt',
};

export const MAX_ATTACHMENTS = 8;
export const MAX_IMAGE_BYTES = 12 * 1024 * 1024; // 12 MB per image
export const MAX_TEXT_CHARS = 100_000; // cap injected text so it can't blow context

export function isSupportedImageMime(mime: string): boolean {
  return Object.prototype.hasOwnProperty.call(IMAGE_MIME_EXT, mime);
}
export function isSupportedTextMime(mime: string): boolean {
  return Object.prototype.hasOwnProperty.call(TEXT_MIME_EXT, mime);
}

/**
 * Attachments live beside the session file, in a per-session folder:
 *   <sessionsDir>/<sessionId>/attachments/
 * basename() on the id blocks path traversal from a crafted session id.
 */
export function sessionAttachmentsDir(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, path.basename(sessionId), 'attachments');
}

function sanitizeName(name: string): string {
  const base = path.basename(name || 'file');
  return base.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 80) || 'file';
}

function extForMime(mime: string): string | null {
  return IMAGE_MIME_EXT[mime] ?? TEXT_MIME_EXT[mime] ?? null;
}

/** Fenced block appended to the user message so the model reads the file text. */
function textAttachmentBlock(name: string, mimeType: string, text: string): string {
  const capped = text.length > MAX_TEXT_CHARS ? `${text.slice(0, MAX_TEXT_CHARS)}\n…[truncated]` : text;
  const lang = mimeType === 'text/csv' ? 'csv' : '';
  return `\n\n[Attached file: ${name}]\n\`\`\`${lang}\n${capped}\n\`\`\``;
}

/**
 * Persist a turn's attachments to disk and produce the message-side artifacts:
 *   - metas:      lightweight records stored on the message (image + text)
 *   - textBlocks: fenced file contents appended to the user message content
 *     so the model actually sees text files (csv/txt)
 * Unsupported types and oversized images are skipped rather than failing the
 * whole request.
 */
export async function ingestAttachments(
  sessionsDir: string,
  sessionId: string,
  incoming: IncomingAttachment[] | undefined,
): Promise<{ metas: AttachmentMeta[]; textBlocks: string }> {
  if (!Array.isArray(incoming) || incoming.length === 0) {
    return { metas: [], textBlocks: '' };
  }
  const dir = sessionAttachmentsDir(sessionsDir, sessionId);
  await fs.mkdir(dir, { recursive: true });

  const metas: AttachmentMeta[] = [];
  let textBlocks = '';
  let index = 0;

  for (const att of incoming.slice(0, MAX_ATTACHMENTS)) {
    index += 1;
    const isImage = att.kind === 'image' && isSupportedImageMime(att.mimeType);
    const isText = att.kind === 'text' && isSupportedTextMime(att.mimeType);
    if (!isImage && !isText) continue;

    const ext = extForMime(att.mimeType);
    if (!ext) continue;

    const safeName = sanitizeName(att.name);
    const hasExt = safeName.toLowerCase().endsWith(`.${ext}`);
    const stored = `${Date.now()}-${index}-${safeName}${hasExt ? '' : `.${ext}`}`;
    const target = path.join(dir, stored);

    if (isImage) {
      const buf = Buffer.from(att.data ?? '', 'base64');
      if (buf.byteLength === 0 || buf.byteLength > MAX_IMAGE_BYTES) continue;
      await fs.writeFile(target, buf);
      metas.push({ name: safeName, mimeType: att.mimeType, kind: 'image', file: stored });
    } else {
      const text = String(att.data ?? '').slice(0, MAX_TEXT_CHARS);
      await fs.writeFile(target, text, 'utf-8');
      metas.push({ name: safeName, mimeType: att.mimeType, kind: 'text', file: stored });
      textBlocks += textAttachmentBlock(safeName, att.mimeType, String(att.data ?? ''));
    }
  }

  return { metas, textBlocks };
}

/** Read one stored attachment and return raw base64 (no data-URI prefix). */
export async function readAttachmentBase64(
  sessionsDir: string,
  sessionId: string,
  file: string,
): Promise<string> {
  const dir = sessionAttachmentsDir(sessionsDir, sessionId);
  const target = path.join(dir, path.basename(file));
  const buf = await fs.readFile(target);
  return buf.toString('base64');
}

/** Read base64 for every image meta on a message, skipping any that fail. */
export async function readImageAttachments(
  sessionsDir: string,
  sessionId: string,
  metas: AttachmentMeta[] | undefined,
): Promise<Array<{ mimeType: string; base64: string }>> {
  const images = (metas ?? []).filter((m) => m.kind === 'image');
  if (images.length === 0) return [];
  const out: Array<{ mimeType: string; base64: string }> = [];
  for (const meta of images) {
    try {
      out.push({ mimeType: meta.mimeType, base64: await readAttachmentBase64(sessionsDir, sessionId, meta.file) });
    } catch {
      // A missing/renamed file must not abort the turn.
    }
  }
  return out;
}

/** MIME type for the attachment serve endpoint, derived from the stored ext. */
export function contentTypeForFile(file: string): string {
  const ext = path.extname(file).toLowerCase().replace('.', '');
  switch (ext) {
    case 'jpg':
    case 'jpeg':
      return 'image/jpeg';
    case 'png':
      return 'image/png';
    case 'bmp':
      return 'image/bmp';
    case 'webp':
      return 'image/webp';
    case 'csv':
      return 'text/csv; charset=utf-8';
    case 'txt':
      return 'text/plain; charset=utf-8';
    default:
      return 'application/octet-stream';
  }
}
