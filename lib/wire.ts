// The two client wires and the one upstream wire.
//
// Clients speak OpenAI Chat Completions (POST /v1/chat/completions) or Anthropic Messages
// (POST /v1/messages), each as one JSON answer or as server-sent events. The gateway speaks Chat
// Completions only, so an Anthropic request is translated into a Chat Completions body on the way
// out and the gateway's answer is translated back on the way in; an OpenAI request and its answer
// pass through with only the model field set to the id this server dialled.
//
// The model field matters: the house scores the model a response names against its own record of
// the call, and a provider's answer spells a free model without the gateway's `openrouter/` prefix.
// Every function here therefore takes the dialled id and puts it back — on the JSON answer, on
// every streamed chunk, and in the Anthropic message_start.
//
// All functions are pure and work on complete, buffered bodies: the caller collects an upstream
// stream in full, checks it, and only then renders the client's answer from what was collected.
// Nothing here reads a clock or a random source — the model lists carry fixed timestamps — so the
// same upstream answer renders to the same client bytes on a live pass and on its replay.

import { isRecord, type Json } from './providers.ts';

/** One model this server serves, as the two /v1/models answers list it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** Fixed listing timestamps: a `created` read from the clock would differ between two passes. */
const OPENAI_CREATED = 1767225600;
const ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models for an `Authorization: Bearer` caller: OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models for an `x-api-key` caller: Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: ANTHROPIC_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** An error body in OpenAI's envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error body in Anthropic's envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The text of a message content: a string itself, or its text blocks joined. */
export function textOfContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part (text or image); null for anything else. */
function openAIPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as one string: its text blocks, else its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOfContent(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as the Chat Completions body the gateway is dialled with: the
 * system prompt becomes a system message, image blocks become image_url parts, tool_use blocks
 * become tool_calls, tool_result blocks become tool messages, and the tools, tool choice, output
 * cap, temperature, top_p, stop sequences and streaming flag carry over. Streaming asks the
 * gateway for the usage chunk as well, since the Messages wire reports usage at the end.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOfContent(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (typeof raw.content === 'string') {
      messages.push({ role, content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOfContent(raw.content);
      messages.push({ role, content: text === '' ? null : text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

/** A tool call's arguments string as an object; {} when it does not parse to one. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input object.
    }
  }
  return {};
}

/** OpenAI's finish_reason in Anthropic's stop_reason vocabulary. */
export function anthropicStopReason(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A Chat Completions JSON answer as an Anthropic Messages object naming the dialled model. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOfContent(message.content);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: num(usage.prompt_tokens), output_tokens: num(usage.completion_tokens) },
  };
}

/** One SSE event: the `event:` line when named, the JSON data line, the closing blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A buffered Chat Completions SSE body with every chunk's `model` set to the dialled id. Lines
 * that are not JSON data payloads (comments, the [DONE] sentinel, blank lines) pass through as
 * they came, so the frame structure the provider sent is the one the client reads.
 */
export function rewriteOpenAIStream(text: string, model: string): string {
  return text
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      const payload = trimmed.slice('data:'.length).trim();
      if (payload === '' || payload === '[DONE]') return line;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
      } catch {
        // Not JSON: relay the line untouched.
      }
      return line;
    })
    .join('\n');
}

/** The parsed chunks of a buffered Chat Completions SSE body, and how the stream ended. */
export interface ParsedStream {
  /** Every data payload that parsed to an object, in order ([DONE] excluded). */
  chunks: Json[];
  /** A [DONE] sentinel was present. */
  sawDone: boolean;
  /** A data payload was present that did not parse to an object. */
  malformed: boolean;
}

/** Split a buffered SSE body into its chunk objects, the way the env's own reader splits frames. */
export function parseOpenAIStream(text: string): ParsedStream {
  const chunks: Json[] = [];
  let sawDone = false;
  let malformed = false;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length === 0) continue;
    const payload = data.join('\n').trim();
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) chunks.push(chunk);
      else malformed = true;
    } catch {
      malformed = true;
    }
  }
  return { chunks, sawDone, malformed };
}

/**
 * The Anthropic event stream for a collected set of Chat Completions chunks: message_start with
 * the stream's real input-token count, one text block and one tool_use block per upstream tool
 * call (text deltas and input_json_delta fragments as they arrived), every block closed before
 * message_delta carries the stop reason and the output-token count, then message_stop.
 */
export function anthropicStreamFromChunks(
  chunks: readonly Json[],
  model: string,
  usage: { input_tokens: number; output_tokens: number },
  stopReason: string,
): string {
  const out: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input_tokens, output_tokens: 0 },
      },
    }),
  ];
  let nextIndex = 0;
  let textIndex = -1;
  const toolIndexByUpstream = new Map<number, number>();
  const openBlocks: number[] = [];
  const closeBlock = (index: number): void => {
    out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    const at = openBlocks.indexOf(index);
    if (at !== -1) openBlocks.splice(at, 1);
  };

  for (const chunk of chunks) {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        openBlocks.push(textIndex);
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexByUpstream.size;
      let index = toolIndexByUpstream.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          closeBlock(textIndex);
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexByUpstream.set(upstreamIndex, index);
        openBlocks.push(index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
  }
  for (const index of [...openBlocks].sort((a, b) => a - b)) closeBlock(index);
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: usage.output_tokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}
