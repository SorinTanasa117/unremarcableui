/**
 * Context manager for the 262k-token rolling window.
 * Uses character estimation (÷4) for speed.
 */

export interface AttachmentMeta {
  name: string;
  mimeType: string;
  kind: 'image' | 'text';
  /** Stored filename inside the session attachments dir (basename only). */
  file: string;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  thinking?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  tool_name?: string;
  duration_ms?: number;
  created_at?: number;
  /** User-attached files (images/text). Bytes live on disk; this is metadata. */
  attachments?: AttachmentMeta[];
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

const MAX_TOKENS = 262_144;
const CHARS_PER_TOKEN = 4;

export interface TokenUsage {
  input: number;
  output: number;
  total: number;
}

function toolCallText(toolCalls: ToolCall[] | undefined): string {
  if (!toolCalls?.length) return '';
  return toolCalls.map((toolCall) => {
    const args = typeof toolCall.function.arguments === 'string'
      ? toolCall.function.arguments
      : JSON.stringify(toolCall.function.arguments);
    return `${toolCall.function.name}\n${args}`;
  }).join('\n');
}

function estimateTokens(msgs: Message[]): number {
  return msgs.reduce((acc, message) => {
    const text = `${message.content ?? ''}${message.thinking ?? ''}${toolCallText(message.tool_calls)}`;
    return acc + Math.ceil(text.length / CHARS_PER_TOKEN);
  }, 0);
}

export class ContextManager {
  private messages: Message[] = [];
  private systemPrompt: string;

  constructor(systemPrompt: string) {
    this.systemPrompt = systemPrompt;
    this.messages = [{ role: 'system', content: systemPrompt }];
  }

  push(msg: Message) {
    this.messages.push(msg);
    this.compact();
  }

  getMessages(): Message[] {
    return this.messages;
  }

  getTokenEstimate(): number {
    return estimateTokens(this.messages);
  }

  /**
   * Input is everything supplied to the model (system, user, and tool
   * messages); output is the model's assistant messages, including hidden
   * function-call arguments such as code passed to write_file. An optional
   * pending response lets the UI update while a streamed assistant message is growing.
   */
  getTokenUsage(pendingMessages: Message[] = []): TokenUsage {
    const messages = [...this.messages, ...pendingMessages];
    const output = estimateTokens(messages.filter((message) => message.role === 'assistant'));
    const input = estimateTokens(messages.filter((message) => message.role !== 'assistant'));
    return { input, output, total: input + output };
  }

  private compact() {
    const ratio = estimateTokens(this.messages) / MAX_TOKENS;
    if (ratio < 0.9) return;

    // Drop oldest non-system messages until we're under 75%
    while (
      estimateTokens(this.messages) / MAX_TOKENS > 0.75 &&
      this.messages.length > 2
    ) {
      // Keep system prompt (index 0) — remove index 1
      if (this.messages[1]?.role !== 'system') {
        this.messages.splice(1, 1);
      } else {
        break;
      }
    }

    // Insert a compaction notice
    this.messages.splice(1, 0, {
      role: 'system',
      content:
        '[Context compacted: earlier messages were removed to stay within the token limit. Continue naturally.]',
    });
  }

  reset() {
    this.messages = [{ role: 'system', content: this.systemPrompt }];
  }
}
