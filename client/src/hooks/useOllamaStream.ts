import { useState, useRef, useCallback, useEffect } from 'react';
import { TokenRateTracker } from '../lib/tokenCounter';

export type MessageRole = 'user' | 'assistant' | 'tool_result' | 'system';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  isActivity?: boolean;
  durationLabel?: string;
  toolName?: string;
  toolSuccess?: boolean;
  timestamp: number;
  durationMs?: number;
}

export interface ToolEvent {
  name: string;
  args?: Record<string, string>;
  output?: string;
  success?: boolean;
}

export type AgentStatus = 'idle' | 'thinking' | 'tool' | 'stopped' | 'error';

export interface ChatSessionInfo {
  sessionId: string;
  title: string;
  persona: string;
  model: string;
  updatedAt: number;
}

export interface TokenUsage {
  input: number;
  output: number;
  total: number;
}

export interface TokenRates {
  /** Prompt prefill (input) tokens per second over the active window. */
  read: number;
  /** Output generation (decoded) tokens per second over the active window. */
  write: number;
  /** True while cumulative input or output counts have moved within the window. */
  active: boolean;
}

export interface SessionLoadResult {
  persona: string;
  model: string;
}

export interface SendMessageOptions {
  /** Send only this request plus the persona prompt; do not grow chat history. */
  isolated?: boolean;
  numCtx?: number;
  think?: boolean;
  /** Caveman mode — compress every reply, same technical content, fewer tokens. */
  caveman?: boolean;
}

interface ActiveRunSnapshot {
  status: 'thinking' | 'tool';
  statusText: string;
  partialContent: string;
  hasStartedGenerating?: boolean;
  startedAt: number;
  updatedAt: number;
}

function activeRunLabel(run: ActiveRunSnapshot, messages: ChatMessage[]): string {
  if (run.partialContent) return run.partialContent;
  if (!run.hasStartedGenerating) return 'Starting AI agent…';
  if (run.status === 'tool') return `Executing action: ${run.statusText}`;

  const lastCompletedStep = [...messages].reverse().find((message) => message.role === 'tool_result');
  return lastCompletedStep
    ? `Reviewing: ${lastCompletedStep.content}`
    : 'Starting task…';
}

interface UseOllamaStreamReturn {
  messages: ChatMessage[];
  status: AgentStatus;
  statusText: string;
  tokenUsage: TokenUsage;
  tokenRates: TokenRates;
  sendMessage: (content: string, model: string, sessionId: string, persona: string, options?: SendMessageOptions) => void;
  stop: (sessionId: string) => void;
  killModel: (model: string) => Promise<void>;
  resetSession: (sessionId: string) => void;
  currentTool: ToolEvent | null;
  loadSessionHistory: (sessionId: string) => Promise<SessionLoadResult>;
  sessionsList: ChatSessionInfo[];
  fetchSessionsList: () => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
}

function uid() { return Math.random().toString(36).slice(2); }

function toolStartLabel(name: string, args?: Record<string, string>): string {
  const file = args?.filepath;
  const query = args?.query;
  const url = args?.url;
  const cmd = args?.command;
  switch (name) {
    case 'write_file': return file ? `Writing file \`${file}\`…` : "Writing a file…";
    case 'read_file': return file ? `Reading file \`${file}\`…` : "Reading a file…";
    case 'web_search': return query ? `Searching the web for "${query}"…` : "Searching the web…";
    case 'browse_url': return url ? `Browsing webpage: ${url}…` : "Browsing webpage…";
    case 'run_terminal': return cmd ? `Running terminal command \`${cmd}\`…` : "Running terminal command…";
    default: return `Executing ${name}…`;
  }
}

function toolResultProcessingLabel(name: string, args?: Record<string, string>, success = true): string {
  if (!success) {
    return `Reviewing failure of **${name}** and adjusting plan…`;
  }
  const file = args?.filepath;
  const query = args?.query;
  const url = args?.url;
  const cmd = args?.command;
  switch (name) {
    case 'write_file': return file ? `Processing write results for \`${file}\`…` : "Processing file write results…";
    case 'read_file': return file ? `Processing content read from \`${file}\`…` : "Processing read file content…";
    case 'web_search': return query ? `Processing web search results for "${query}"…` : "Processing search results…";
    case 'browse_url': return url ? `Processing browsed content from ${url}…` : "Processing browsed webpage…";
    case 'run_terminal': return cmd ? `Processing terminal execution results for \`${cmd}\`…` : "Processing terminal output…";
    default: return `Processing results of ${name}…`;
  }
}

function completedStepLabel(name: string, args?: Record<string, string>, success = true): string {
  const file = args?.filepath;
  const outcome = success ? '' : ' but it reported an error';
  switch (name) {
    case 'write_file': return file ? `Wrote \`${file}\`${outcome}.` : `Wrote a file${outcome}.`;
    case 'read_file': return file ? `Read \`${file}\`${outcome}.` : `Read a file${outcome}.`;
    case 'web_search': return `Completed a web search${outcome}.`;
    case 'browse_url': return `Inspected the requested page${outcome}.`;
    case 'run_terminal': return `Ran a terminal command${outcome}.`;
    default: return `Completed the requested action${outcome}.`;
  }
}

function savedNextStepLabel(message: any): string | null {
  const call = message.tool_calls?.[0];
  if (!call?.function?.name) return null;
  let args: Record<string, string> | undefined;
  try {
    args = typeof call.function.arguments === 'string'
      ? JSON.parse(call.function.arguments)
      : call.function.arguments;
  } catch {}
  return toolStartLabel(call.function.name, args);
}

function savedToolName(messages: any[], toolIndex: number): string | undefined {
  const tool = messages[toolIndex];
  if (tool.tool_name) return tool.tool_name;
  for (let index = toolIndex - 1; index >= 0; index -= 1) {
    const calls = messages[index].tool_calls ?? [];
    const match = calls.find((call: any) => call.id === tool.tool_call_id);
    if (match?.function?.name) return match.function.name;
  }
  return undefined;
}

function savedToolArgs(messages: any[], toolIndex: number): Record<string, string> | undefined {
  const tool = messages[toolIndex];
  for (let index = toolIndex - 1; index >= 0; index -= 1) {
    const calls = messages[index].tool_calls ?? [];
    const match = calls.find((call: any) => call.id === tool.tool_call_id);
    if (!match?.function) continue;
    try {
      return typeof match.function.arguments === 'string'
        ? JSON.parse(match.function.arguments)
        : match.function.arguments;
    } catch {
      return undefined;
    }
  }
  return undefined;
}

export function useOllamaStream(): UseOllamaStreamReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamStatus, setStatus] = useState<AgentStatus>('idle');
  const [streamStatusText, setStatusText] = useState('');
  const [visibleSessionId, setVisibleSessionId] = useState('');
  const [tokenUsage, setTokenUsage] = useState<TokenUsage>({ input: 0, output: 0, total: 0 });
  const [tokenRates, setTokenRates] = useState<TokenRates>({ read: 0, write: 0, active: false });
  const [currentTool, setCurrentTool] = useState<ToolEvent | null>(null);
  const [sessionsList, setSessionsList] = useState<ChatSessionInfo[]>([]);
  const abortRef = useRef<(() => void) | null>(null);

  // Time tracking refs
  const activeAssistantId = useRef<string>('');
  const exchangeStartTime = useRef<number>(0);
  const toolStartTime = useRef<number>(0);
  const activeToolMsgId = useRef<string>('');
  const activeTool = useRef<ToolEvent | null>(null);
  const sessionLoadVersion = useRef(0);
  const streamSessionId = useRef<string | null>(null);
  const visibleSessionRef = useRef('');
  const messagesBySession = useRef(new Map<string, ChatMessage[]>());
  const tokensBySession = useRef(new Map<string, TokenUsage>());

  // Token-rate tracking. The server emits a `tokens` event after each streamed
  // chunk. We record (timestamp, input, output) and compute read (prefill) and
  // write (decode) speeds over a recent sliding window so the top-bar readout
  // stays smooth even when the underlying SSE event cadence is bursty.
  const rateTrackerRef = useRef<TokenRateTracker>(new TokenRateTracker());

  const status = streamSessionId.current === visibleSessionId ? streamStatus : 'idle';
  const statusText = streamSessionId.current === visibleSessionId ? streamStatusText : '';

  const updateSessionMessages = useCallback((sessionId: string, update: (current: ChatMessage[]) => ChatMessage[]) => {
    const next = update(messagesBySession.current.get(sessionId) ?? []);
    messagesBySession.current.set(sessionId, next);
    if (visibleSessionRef.current === sessionId) setMessages(next);
  }, []);

  const updateSessionTokens = useCallback((sessionId: string, usage: TokenUsage) => {
    tokensBySession.current.set(sessionId, usage);
    if (visibleSessionRef.current === sessionId) setTokenUsage(usage);
    // Only sample live rates for the active streaming session so a hydrated
    // history load doesn't pollute the in-flight measurement window.
    if (streamSessionId.current === sessionId) {
      const rates = rateTrackerRef.current.record(Date.now(), usage.input, usage.output);
      if (visibleSessionRef.current === sessionId) {
        setTokenRates((current) => (
          current.read === rates.read && current.write === rates.write && current.active === rates.active
            ? current
            : rates
        ));
      }
    }
  }, []);

  const resetTokenRates = useCallback(() => {
    rateTrackerRef.current.reset();
    setTokenRates({ read: 0, write: 0, active: false });
  }, []);

  // Tick once a second so the displayed rate stays fresh between SSE
  // `tokens` events (which arrive at most once per chunk). The tracker
  // already updates instantly on each event, so this only repaints when a
  // long gap would otherwise freeze the readout.
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      if (streamSessionId.current !== visibleSessionRef.current) return;
      const rates = rateTrackerRef.current.rates();
      setTokenRates((current) => (
        current.read === rates.read && current.write === rates.write && current.active === rates.active
          ? current
          : rates
      ));
    }, 500);
    return () => window.clearInterval(intervalId);
  }, []);

  const appendToken = useCallback((token: string) => {
    const id = activeAssistantId.current;
    const sessionId = streamSessionId.current;
    if (!id || !sessionId) return;
    const activeMessage = messagesBySession.current.get(sessionId)?.find((message) => message.id === id);
    if (activeMessage?.isActivity) {
      const responseId = uid();
      activeAssistantId.current = responseId;
      updateSessionMessages(sessionId, (messages) => [
        ...messages.map((message) => message.id === id
          ? {
              ...message,
              isActivity: false,
              durationMs: Date.now() - message.timestamp,
              durationLabel: 'Elapsed',
            }
          : message
        ),
        { id: responseId, role: 'assistant', content: token, timestamp: Date.now() },
      ]);
      return;
    }
    updateSessionMessages(sessionId, (messages) =>
      messages.map((m) => m.id === id ? { ...m, content: m.content + token } : m)
    );
  }, [updateSessionMessages]);

  const addAssistantBubble = useCallback((content = '', isActivity = false): string => {
    const id = uid();
    activeAssistantId.current = id;
    const sessionId = streamSessionId.current;
    if (sessionId) updateSessionMessages(sessionId, (messages) => [...messages, {
      id,
      role: 'assistant',
      content,
      isActivity,
      timestamp: Date.now(),
    }]);
    return id;
  }, [updateSessionMessages]);

  const fetchSessionsList = useCallback(async () => {
    try {
      const res = await fetch('/api/ollama/sessions');
      if (res.ok) {
        const data = await res.json();
        setSessionsList(data);
      }
    } catch {}
  }, []);

  const loadSessionHistory = useCallback(async (sessionId: string): Promise<SessionLoadResult> => {
    setVisibleSessionId(sessionId);
    visibleSessionRef.current = sessionId;
    const loadVersion = ++sessionLoadVersion.current;
    // Never briefly show the previously selected chat's usage for a new chat.
    const cachedMessages = messagesBySession.current.get(sessionId);
    const cachedTokens = tokensBySession.current.get(sessionId);
    setMessages(cachedMessages ?? []);
    setTokenUsage(cachedTokens ?? { input: 0, output: 0, total: 0 });
    setCurrentTool(null);
    activeAssistantId.current = '';
    try {
      const res = await fetch(`/api/ollama/session?sessionId=${encodeURIComponent(sessionId)}`);
      if (res.ok) {
        const data = await res.json();
        if (loadVersion !== sessionLoadVersion.current) return { persona: 'coder', model: '' };
        // Preserve the local SSE stream when this tab owns it. A refreshed tab
        // has no AbortController, so it must continue hydrating from the
        // server-side run snapshot instead.
        if (streamSessionId.current === sessionId && cachedMessages && abortRef.current) {
          return { persona: data.persona || 'coder', model: data.model || '' };
        }
        const mapped: ChatMessage[] = (data.messages || [])
          .filter((m: any) => m.role !== 'system' || m.content.startsWith('[Context compacted'))
          .map((m: any, index: number, sessionMessages: any[]) => {
            const roleMap: Record<string, MessageRole> = {
              user: 'user',
              assistant: 'assistant',
              tool: 'tool_result',
              system: 'system',
            };
            return {
              id: m.tool_call_id || uid(),
              role: roleMap[m.role] || 'assistant',
              content: m.role === 'tool'
                ? completedStepLabel(
                    savedToolName(sessionMessages, index) ?? 'tool',
                    savedToolArgs(sessionMessages, index),
                  )
                : (m.content || savedNextStepLabel(m) || ''),
              toolName: m.role === 'tool' ? savedToolName(sessionMessages, index) : undefined,
              timestamp: m.created_at ?? Date.now(),
              durationMs: m.role === 'tool' ? (m.duration_ms ?? 0) : undefined,
            };
          });

        const run = data.run as ActiveRunSnapshot | undefined;
        if (run) {
          const activeMessageId = `active-${sessionId}`;
          mapped.push({
            id: activeMessageId,
            role: 'assistant',
            content: activeRunLabel(run, mapped),
            timestamp: run.startedAt,
          });
          activeAssistantId.current = activeMessageId;
          streamSessionId.current = sessionId;
          setStatus(run.status);
          setStatusText(run.statusText);
        } else if (streamSessionId.current === sessionId) {
          streamSessionId.current = null;
          activeAssistantId.current = '';
          setStatus('idle');
          setStatusText('');
          setCurrentTool(null);
        }
        messagesBySession.current.set(sessionId, mapped);
        setMessages(mapped);

        // Fetch this session's input/output context breakdown.
        const tokenRes = await fetch(`/api/ollama/tokens?sessionId=${encodeURIComponent(sessionId)}`);
        if (tokenRes.ok) {
          const tData = await tokenRes.json();
          if (loadVersion === sessionLoadVersion.current) {
            updateSessionTokens(sessionId, { input: tData.input ?? 0, output: tData.output ?? 0, total: tData.total ?? 0 });
          }
        }
        return { persona: data.persona || 'coder', model: data.model || '' };
      }
    } catch {}
    return { persona: 'coder', model: '' };
  }, [updateSessionTokens]);

  const refreshActiveRun = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`/api/ollama/session?sessionId=${encodeURIComponent(sessionId)}`);
      if (!res.ok) return;

      const data = await res.json();
      const run = data.run as ActiveRunSnapshot | undefined;
      if (!run) {
        if (streamSessionId.current === sessionId) {
          streamSessionId.current = null;
          activeAssistantId.current = '';
          setStatus('idle');
          setStatusText('');
          setCurrentTool(null);
          // The completed response is now persisted in the session history.
          void loadSessionHistory(sessionId);
        }
        return;
      }

      const activeMessageId = `active-${sessionId}`;
      activeAssistantId.current = activeMessageId;
      streamSessionId.current = sessionId;
      setStatus(run.status);
      setStatusText(run.statusText);
      updateSessionMessages(sessionId, (messages) => {
        const existing = messages.find((message) => message.id === activeMessageId);
        const content = run.partialContent || existing?.content || activeRunLabel(run, messages);
        if (existing) {
          return messages.map((message) => message.id === activeMessageId
            ? { ...message, content, timestamp: run.startedAt }
            : message
          );
        }
        return [...messages, {
          id: activeMessageId,
          role: 'assistant',
          content,
          timestamp: run.startedAt,
        }];
      });
    } catch {}
  }, [loadSessionHistory, updateSessionMessages]);

  // A page refresh ends the browser's SSE connection but not the server-side
  // run. Rehydrate that run from its persisted snapshot, then update only its
  // active bubble until the server reports completion.
  useEffect(() => {
    const isVisibleRun = !abortRef.current && streamSessionId.current === visibleSessionId &&
      (streamStatus === 'thinking' || streamStatus === 'tool');
    if (!visibleSessionId || !isVisibleRun) return;

    const intervalId = window.setInterval(() => {
      void refreshActiveRun(visibleSessionId);
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, [refreshActiveRun, streamStatus, visibleSessionId]);

  const deleteSession = useCallback(async (sessionId: string) => {
    try {
      const res = await fetch(`/api/ollama/session?sessionId=${encodeURIComponent(sessionId)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        await fetchSessionsList();
      }
    } catch {}
  }, [fetchSessionsList]);

  const sendMessage = useCallback(async (
    content: string,
    model: string,
    sessionId: string,
    persona: string,
    options: SendMessageOptions = {},
  ) => {
    // A user message belongs to the current view, so a late history request
    // must not replace it with another session's messages or counters.
    sessionLoadVersion.current += 1;
    const start = Date.now();
    // The initial history request can still be pending when the user sends
    // the first prompt. Claim the view before appending so that prompt is not
    // stored only in the per-session cache while the visible session ref is
    // still empty.
    visibleSessionRef.current = sessionId;
    setVisibleSessionId(sessionId);
    streamSessionId.current = sessionId;
    exchangeStartTime.current = start;

    // Reset rate tracking so the first SSE `tokens` event establishes a fresh
    // baseline. A pre-seeded readout would divide the entire session's
    // accumulated tokens by a tiny elapsed window and produce a huge burst.
    rateTrackerRef.current.start(start);
    setTokenRates({ read: 0, write: 0, active: false });

    // Add user bubble
    const userMsg: ChatMessage = { id: uid(), role: 'user', content, timestamp: start };
    updateSessionMessages(sessionId, (messages) => [...messages, userMsg]);

    // Make startup visible instead of showing an unexplained empty cursor.
    addAssistantBubble('Starting AI agent…', true);

    setStatus('thinking');
    setStatusText('Thinking…');
    setCurrentTool(null);

    const controller = new AbortController();
    abortRef.current = () => controller.abort();

    try {
      const res = await fetch('/api/ollama/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, model, message: content, persona, isolated: options.isolated, numCtx: options.numCtx, think: options.think, caveman: options.caveman }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        let detail = `Server error: ${res.status}`;
        try {
          const body = await res.json();
          if (body.error) detail = body.error;
        } catch {}
        throw new Error(detail);
      }
      fetchSessionsList();

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          const lines = part.split('\n');
          const eventLine = lines.find((l) => l.startsWith('event:'));
          const dataLine = lines.find((l) => l.startsWith('data:'));
          if (!eventLine || !dataLine) continue;

          const event = eventLine.slice(6).trim();
          let data: any;
          try { data = JSON.parse(dataLine.slice(5).trim()); } catch { continue; }

          switch (event) {
            case 'thinking_started': {
              const currentAssistantId = activeAssistantId.current;
              updateSessionMessages(sessionId, (messages) => messages.map((message) =>
                message.id === currentAssistantId && message.isActivity
                  ? { ...message, content: 'Thinking…' }
                  : message
              ));
              setStatus('thinking');
              setStatusText('Thinking…');
              break;
            }

            case 'token':
              appendToken(data.content);
              break;

            case 'tool_start': {
              toolStartTime.current = Date.now();
              const currentAssistantId = activeAssistantId.current;
              const descriptiveText = toolStartLabel(data.name, data.args);
              let updatedActivity = false;
              updateSessionMessages(sessionId, (messages) => messages.map((message) => {
                if (message.id !== currentAssistantId || !message.isActivity) return message;
                updatedActivity = true;
                return {
                  ...message,
                  content: descriptiveText,
                  isActivity: false,
                  durationMs: toolStartTime.current - message.timestamp,
                  durationLabel: 'Elapsed',
                };
              }));
              if (!updatedActivity) {
                updateSessionMessages(sessionId, (messages) => [...messages, {
                  id: uid(),
                  role: 'assistant',
                  content: descriptiveText,
                  timestamp: toolStartTime.current,
                  durationMs: 0,
                  durationLabel: 'Elapsed',
                }]);
              }
              const toolMsgId = uid();
              activeToolMsgId.current = toolMsgId;

              const toolMsg: ChatMessage = {
                id: toolMsgId,
                role: 'tool_result',
                content: `Running **${data.name}**…`,
                toolName: data.name,
                timestamp: toolStartTime.current,
              };
              updateSessionMessages(sessionId, (messages) => [...messages, toolMsg]);
              const tool = { name: data.name, args: data.args };
              activeTool.current = tool;
              setCurrentTool(tool);

              const labels: Record<string, string> = {
                web_search: `🔍 Searching: "${data.args?.query}"`,
                browse_url: `🌐 Browsing ${data.args?.url}`,
                write_file: `📝 Writing ${data.args?.filepath}`,
                read_file: `📄 Reading ${data.args?.filepath}`,
                run_terminal: `💻 Running: ${data.args?.command}`,
              };
              setStatusText(labels[data.name as string] ?? `Running ${data.name}…`);
              setStatus('tool');

              break;
            }

            case 'tool_result': {
              const duration = data.durationMs ?? (Date.now() - toolStartTime.current);
              const toolMsgId = activeToolMsgId.current;

              updateSessionMessages(sessionId, (messages) =>
                messages.map((m) =>
                  m.id === toolMsgId
                    ? {
                      ...m,
                        durationMs: duration,
                        content: data.success === false
                          ? `⚠️ ${data.name} failed: ${data.output ?? 'No details returned.'}`
                          : completedStepLabel(
                            m.toolName ?? data.name ?? 'tool',
                            activeTool.current?.args,
                            true,
                          ),
                      }
                    : m
                )
              );

              const currentArgs = activeTool.current?.args;
              setCurrentTool((prev) => prev ? { ...prev, output: data.output, success: data.success } : null);
              activeTool.current = null;
              setStatus('thinking');
              setStatusText('Thinking…');
              addAssistantBubble(
                toolResultProcessingLabel(data.name, currentArgs, data.success !== false),
                true,
              );
              break;
            }

            case 'tokens':
              updateSessionTokens(sessionId, { input: data.input ?? 0, output: data.output ?? 0, total: data.total ?? 0 });
              break;

            case 'done': {
              const totalDuration = Date.now() - exchangeStartTime.current;
              const lastAssId = activeAssistantId.current;
              updateSessionMessages(sessionId, (messages) => messages
                .filter((m) => m.id !== lastAssId || m.content.trim())
                .map((m) => m.id === lastAssId ? { ...m, durationMs: totalDuration } : m)
              );
              setStatus('idle');
              setStatusText('');
              streamSessionId.current = null;
              setCurrentTool(null);
              activeAssistantId.current = '';
              resetTokenRates();
              fetchSessionsList();
              break;
            }

            case 'stopped':
              setStatus('stopped');
              setStatusText('Stopped');
              setCurrentTool(null);
              activeAssistantId.current = '';
              resetTokenRates();
              break;

            case 'error':
              setStatus('error');
              setStatusText(data.message ?? 'Error');
              activeAssistantId.current = '';
              resetTokenRates();
              updateSessionMessages(sessionId, (messages) => [...messages, {
                id: uid(), role: 'system',
                content: `⚠️ Error: ${data.message}`,
                timestamp: Date.now(),
              }]);
              break;
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus('error');
        setStatusText(err.message);
      } else {
        setStatus('stopped');
        setStatusText('Stopped');
      }
      activeAssistantId.current = '';
      resetTokenRates();
      updateSessionMessages(sessionId, (messages) => [...messages, {
        id: uid(), role: 'system', content: `⚠️ Error: ${err.message}`, timestamp: Date.now(),
      }]);
    }
  }, [appendToken, addAssistantBubble, fetchSessionsList, updateSessionMessages, updateSessionTokens, resetTokenRates]);

  const stop = useCallback(async (sessionId: string) => {
    if (streamSessionId.current !== sessionId) return;
    abortRef.current?.();
    await fetch('/api/ollama/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    }).catch(() => {});
    setStatus('stopped');
    setStatusText('Stopped');
    resetTokenRates();
  }, [resetTokenRates]);

  const killModel = useCallback(async (model: string) => {
    if (!model) return;
    try {
      const res = await fetch('/api/ollama/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });
      if (!res.ok) throw new Error(`Unable to unload model (${res.status})`);
      setStatus('stopped');
      setStatusText('Model unloaded');
      resetTokenRates();
    } catch (err: any) {
      setStatus('error');
      setStatusText(err.message);
    }
  }, [resetTokenRates]);

  const resetSession = useCallback(async (sessionId: string) => {
    await fetch('/api/ollama/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    }).catch(() => {});
    messagesBySession.current.delete(sessionId);
    tokensBySession.current.set(sessionId, { input: 0, output: 0, total: 0 });
    if (visibleSessionRef.current === sessionId) setMessages([]);
    setStatus('idle');
    setStatusText('');
    setTokenUsage({ input: 0, output: 0, total: 0 });
    setCurrentTool(null);
    activeAssistantId.current = '';
    resetTokenRates();
    await fetchSessionsList();
  }, [fetchSessionsList, resetTokenRates]);

  return {
    messages,
    status,
    statusText,
    tokenUsage,
    tokenRates,
    sendMessage,
    stop,
    killModel,
    resetSession,
    currentTool,
    loadSessionHistory,
    sessionsList,
    fetchSessionsList,
    deleteSession
  };
}
