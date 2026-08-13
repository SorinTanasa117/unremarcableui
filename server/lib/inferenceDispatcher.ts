import { Agent } from 'undici';

// Local inference servers (Ollama, llama.cpp) can go silent for many minutes
// while prefilling a large context or generating a buffered tool call on slow
// hardware (e.g. a 20B model on an iGPU). Node's global fetch (undici) aborts
// any response whose body pauses for more than 5 minutes by default
// (BodyTimeoutError), which kills legitimate long generations mid-stream.
// Inference calls use this dispatcher with those idle timeouts disabled;
// cancellation stays with each request's AbortSignal.
// The undici package's Agent and the Dispatcher type that @types/node bundles
// for RequestInit.dispatcher come from two parallel type packages and are not
// structurally assignable. At runtime Node's global fetch IS undici, so a real
// Agent instance is exactly what it expects; the cast only bridges the types.
type FetchDispatcher = NonNullable<RequestInit['dispatcher']>;

export const inferenceDispatcher = new Agent({
  bodyTimeout: 0,
  headersTimeout: 0,
}) as unknown as FetchDispatcher;
