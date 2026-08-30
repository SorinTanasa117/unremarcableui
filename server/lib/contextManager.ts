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

const CHARS_PER_TOKEN = 4;
const IMAGE_TOKEN_ESTIMATE = 2_048;

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

export function estimateMessageTokens(msgs: Message[]): number {
  return msgs.reduce((acc, message) => {
    const text = `${message.content ?? ''}${message.thinking ?? ''}${toolCallText(message.tool_calls)}`;
    const imageTokens = message.attachments?.filter((attachment) => attachment.kind === 'image').length
      ?? 0;
    return acc + Math.ceil(text.length / CHARS_PER_TOKEN) + imageTokens * IMAGE_TOKEN_ESTIMATE;
  }, 0);
}

export interface ModelContextWindow {
  messages: Message[];
  estimatedTokens: number;
  compacted: boolean;
  droppedMessages: number;
}

export function compactToolCall(toolCall: ToolCall): ToolCall {
  try {
    const rawArguments = toolCall.function.arguments as unknown;
    const args = (typeof rawArguments === 'string'
      ? JSON.parse(rawArguments)
      : rawArguments) as Record<string, unknown>;
    if (!args || typeof args !== 'object' || Array.isArray(args)) return toolCall;

    const compacted: Record<string, unknown> = {};
    if (typeof args.filepath === 'string') compacted.filepath = args.filepath;
    if (typeof args.command === 'string') compacted.command = args.command.slice(0, 500);
    if (typeof args.query === 'string') compacted.query = args.query.slice(0, 500);
    if (typeof args.url === 'string') compacted.url = args.url;
    if (typeof args.offset === 'number') compacted.offset = args.offset;
    if (typeof args.limit === 'number') compacted.limit = args.limit;
    const omittedFields = ['content', 'old_str', 'new_str'].filter((field) => field in args);
    if (omittedFields.length > 0) {
      // Do not keep placeholders under executable argument names. Weak models
      // copy those receipts into new write_file calls and can mistake them for
      // real source. An unknown metadata key preserves the history signal but
      // cannot satisfy any file-mutation tool schema.
      compacted._history_receipt = `${omittedFields.join(', ')} omitted; operation already completed. Re-read the current file before making a new targeted edit.`;
    }

    return {
      ...toolCall,
      function: {
        ...toolCall.function,
        arguments: JSON.stringify(compacted),
      },
    };
  } catch {
    return toolCall;
  }
}

function compactHistoricalMessage(message: Message): Message {
  if (message.role === 'assistant') {
    return {
      ...message,
      thinking: undefined,
      tool_calls: message.tool_calls?.map(compactToolCall),
    };
  }

  if (message.role === 'tool' && message.content.length > 2_000) {
    return {
      ...message,
      content: `${message.content.slice(0, 2_000)}\n\n[Historical tool output truncated. Re-read current workspace state if needed.]`,
    };
  }

  return { ...message };
}

function groupUserTurns(messages: Message[]): Message[][] {
  const turns: Message[][] = [];
  for (const message of messages) {
    if (message.role === 'user' || turns.length === 0) {
      turns.push([message]);
    } else {
      turns[turns.length - 1].push(message);
    }
  }
  return turns;
}

function lastUserIndex(messages: Message[]): number {
  for (let index = messages.length - 1; index >= 0; index--) {
    if (messages[index].role === 'user') return index;
  }
  return -1;
}

function removeMessageWithToolPair(messages: Message[], index: number) {
  const message = messages[index];
  if (!message) return;

  if (message.role === 'assistant' && message.tool_calls?.length) {
    const callIds = new Set(message.tool_calls.map((toolCall) => toolCall.id));
    for (let cursor = messages.length - 1; cursor >= 0; cursor--) {
      if (cursor !== index && messages[cursor].role === 'tool' && callIds.has(messages[cursor].tool_call_id ?? '')) {
        messages.splice(cursor, 1);
        if (cursor < index) index--;
      }
    }
  } else if (message.role === 'tool' && message.tool_call_id) {
    const assistantIndex = messages.findIndex((candidate) =>
      candidate.role === 'assistant'
      && candidate.tool_calls?.some((toolCall) => toolCall.id === message.tool_call_id),
    );
    if (assistantIndex >= 0) {
      const first = Math.max(index, assistantIndex);
      const second = Math.min(index, assistantIndex);
      messages.splice(first, 1);
      messages.splice(second, 1);
      return;
    }
  }

  messages.splice(index, 1);
}

function buildCompactionNotice(droppedTurns: Message[][]): Message {
  const userGoals = droppedTurns
    .map((turn) => turn.find((message) => message.role === 'user')?.content.trim())
    .filter((content): content is string => Boolean(content))
    .slice(-3)
    .map((content) => `- ${content.slice(0, 400)}`);

  const fileOperations: string[] = [];
  for (const turn of droppedTurns) {
    for (const message of turn) {
      for (const toolCall of message.tool_calls ?? []) {
        if (!['write_file', 'edit_file', 'read_file'].includes(toolCall.function.name)) continue;
        try {
          const rawArguments = toolCall.function.arguments as unknown;
          const args = (typeof rawArguments === 'string'
            ? JSON.parse(rawArguments)
            : rawArguments) as Record<string, unknown>;
          if (typeof args.filepath === 'string') {
            fileOperations.push(`- ${toolCall.function.name}: ${args.filepath}`);
          }
        } catch {}
      }
    }
  }

  const sections = [
    '[Earlier chat turns compacted to fit the active model context.]',
    userGoals.length ? `Recent goals from compacted turns:\n${userGoals.join('\n')}` : '',
    fileOperations.length
      ? `Recent workspace operations:\n${fileOperations.slice(-12).join('\n')}`
      : '',
    'Workspace files are the source of truth. Use read_file before editing existing code.',
  ].filter(Boolean);

  return { role: 'system', content: sections.join('\n\n') };
}

export class ContextManager {
  private messages: Message[] = [];
  private systemPrompt: string;
  // A durable, model-generated brief of earlier work. Pinned right after the
  // system prompt and never dropped, so continuity survives when raw turns are
  // shed from the outbound window on a small-RAM machine.
  private durableSummary?: string;

  constructor(systemPrompt: string) {
    this.systemPrompt = systemPrompt;
    this.messages = [{ role: 'system', content: systemPrompt }];
  }

  push(msg: Message) {
    this.messages.push(msg);
  }

  getMessages(): Message[] {
    return this.messages;
  }

  setDurableSummary(summary: string) {
    this.durableSummary = summary.trim() || undefined;
  }

  getDurableSummary(): string | undefined {
    return this.durableSummary;
  }

  private summaryMessage(): Message | null {
    return this.durableSummary
      ? { role: 'system', content: `[Rolling summary of earlier work — the raw turns it covers may be dropped from this window. Treat workspace files as source of truth.]\n${this.durableSummary}` }
      : null;
  }

  getTokenEstimate(): number {
    return estimateMessageTokens(this.messages);
  }

  /**
   * Build an outbound-only context window for the active runtime budget.
   * Full session history remains persisted for the UI. Historical thinking and
   * large tool arguments are removed because current workspace files, not old
   * generated source blobs, are authoritative.
   */
  getModelContext(maxTokens: number): ModelContextWindow {
    const safeMax = Math.max(1_024, Math.floor(maxTokens));
    const firstSystemIndex = this.messages.findIndex((message) => message.role === 'system');
    const firstSystem = firstSystemIndex >= 0
      ? this.messages[firstSystemIndex]
      : { role: 'system' as const, content: this.systemPrompt };
    const nonSystem = this.messages
      .filter((message, index) =>
        index !== firstSystemIndex
        && (message.role !== 'system' || message.created_at !== undefined),
      )
      .map(compactHistoricalMessage);
    const pinnedSummary = this.summaryMessage();
    const pinned = pinnedSummary ? [{ ...firstSystem }, pinnedSummary] : [{ ...firstSystem }];
    const compactedAll = [...pinned, ...nonSystem];
    const compactedAllTokens = estimateMessageTokens(compactedAll);
    const originalTokens = estimateMessageTokens(this.messages);

    if (compactedAllTokens <= safeMax) {
      return {
        messages: compactedAll,
        estimatedTokens: compactedAllTokens,
        compacted: compactedAllTokens < originalTokens || compactedAll.length < this.messages.length,
        droppedMessages: this.messages.length - compactedAll.length,
      };
    }

    const turns = groupUserTurns(nonSystem);
    const keptTurns: Message[][] = [];
    let keptTokens = estimateMessageTokens(pinned);

    for (let index = turns.length - 1; index >= 0; index--) {
      const turnTokens = estimateMessageTokens(turns[index]);
      const noticeReserve = 1_024;
      if (keptTurns.length > 0 && keptTokens + turnTokens + noticeReserve > safeMax) break;
      keptTurns.unshift(turns[index]);
      keptTokens += turnTokens;
    }

    const droppedTurnCount = Math.max(0, turns.length - keptTurns.length);
    const droppedTurns = turns.slice(0, droppedTurnCount);
    const notice = droppedTurns.length ? buildCompactionNotice(droppedTurns) : null;
    let outbound = [
      { ...firstSystem },
      ...(pinnedSummary ? [pinnedSummary] : []),
      ...(notice ? [notice] : []),
      ...keptTurns.flat(),
    ];

    // A single recent turn can still exceed the budget. Remove its oldest
    // messages while preserving the latest user request.
    while (estimateMessageTokens(outbound) > safeMax && outbound.length > 3) {
      const latestUserIndex = lastUserIndex(outbound);
      const removableIndex = outbound.findIndex((message, index) =>
        index > (notice ? 1 : 0)
        && index !== latestUserIndex
        && message.role !== 'system',
      );
      if (removableIndex < 0) break;
      removeMessageWithToolPair(outbound, removableIndex);
    }

    if (estimateMessageTokens(outbound) > safeMax) {
      const latestUserIndex = lastUserIndex(outbound);
      if (latestUserIndex >= 0) {
        const latestUser = outbound[latestUserIndex];
        const otherTokens = estimateMessageTokens(outbound.filter((_, index) => index !== latestUserIndex));
        const availableChars = Math.max(1_000, (safeMax - otherTokens) * CHARS_PER_TOKEN);
        if (latestUser.content.length > availableChars) {
          const half = Math.floor((availableChars - 120) / 2);
          outbound[latestUserIndex] = {
            ...latestUser,
            content: `${latestUser.content.slice(0, half)}\n\n[Middle of oversized user message omitted to fit context]\n\n${latestUser.content.slice(-half)}`,
          };
        }
      }
    }

    return {
      messages: outbound,
      estimatedTokens: estimateMessageTokens(outbound),
      compacted: true,
      droppedMessages: Math.max(0, this.messages.length - outbound.length),
    };
  }

  /**
   * Input is everything supplied to the model (system, user, and tool
   * messages); output is the model's assistant messages, including hidden
   * function-call arguments such as code passed to write_file. An optional
   * pending response lets the UI update while a streamed assistant message is growing.
   */
  getTokenUsage(pendingMessages: Message[] = []): TokenUsage {
    const messages = [...this.messages, ...pendingMessages];
    const output = estimateMessageTokens(messages.filter((message) => message.role === 'assistant'));
    const input = estimateMessageTokens(messages.filter((message) => message.role !== 'assistant'));
    return { input, output, total: input + output };
  }

  reset() {
    this.messages = [{ role: 'system', content: this.systemPrompt }];
  }
}
