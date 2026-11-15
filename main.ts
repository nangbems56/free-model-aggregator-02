// The best live free-model aggregator: one server, two client wires, one gateway.
//
// What a trial is. The house env sends six prompts, one at a time, to this server's OpenAI and
// Anthropic endpoints. Each prompt names the model "auto"; this server picks a free model from the
// brief's rated catalog, dials it through the house gateway (Chat Completions only), and renders
// the answer back on the client's wire. The house scores the model that served each prompt by its
// rating, so the whole game is: serve every prompt with the highest-rated free model that meets
// the prompt's needs — and when a provider refuses, prove the refusal was real and still serve.
//
// How a prompt is served here.
//
//   1. Needs. The brief's manifest says, per prompt and in order, what it needs (tools, JSON mode,
//      vision, context). The request body is cross-checked and can only add needs (a tools array,
//      a json_object response format, an image part). Candidates are the free models meeting every
//      need, best rating first; models that already refused twice this trial sink to the back, and
//      if nothing meets the needs the rest follow in rating order — a lesser model's answer still
//      beats an error.
//   2. Attempts. Candidates are dialled in order, at most five attempts, each with its own timeout,
//      inside the client's per-prompt deadline. A refusal (any non-2xx, a timeout, a transport
//      error) is recorded for the failures trail and the next candidate is tried. A byte-identical
//      request is never sent twice: every attempt names a different model.
//   3. Proof before serving. A 2xx is not trusted until it is checked the way the house checks:
//      well-formed on its wire, a legal finish reason, usage reporting at least one output token,
//      and — when the prompt names exactly one token to reply with — that token present in the
//      answer's text. A 2xx that fails the check is set aside (the first one is kept as a
//      last-resort answer) and the next candidate is tried: a served-but-worthless answer scores
//      the same as an error, while a failover can still earn the prompt.
//   4. The trail. When a prompt is served after one or more refusals, the answer carries the
//      `x-makefun-upstream-failures` header listing each refusal (status, the provider's error
//      code, the model dialled), which is how the house credits the recovery. The trail rides only
//      on answers that passed the checks — on anything else it could only mark a failure the
//      house already counts.
//
// Determinism. The live pass is replayed and the replay must reproduce the house's record exactly,
// so every decision above is a pure function of the brief, the requests, and the gateway's answers
// seen so far. Timeouts only bound how long this server waits; nothing time-based is ever written
// into a request or an answer.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';

import {
  buildCatalog,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStopReason,
  anthropicStreamFromChunks,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  parseOpenAIStream,
  rewriteOpenAIStream,
  textOfContent,
  type ServedModel,
} from './lib/wire.ts';

/** Attempts per prompt: the first choice plus room to fail over past flaky providers. */
const MAX_ATTEMPTS = 5;
/** One gateway call may take at most this long before it counts as a refusal. */
const ATTEMPT_TIMEOUT_MS = 13_000;
/** A prompt is abandoned (last-resort answer or 502) this long after it arrived; the client waits 30s. */
const PROMPT_DEADLINE_MS = 26_000;
/** Trial-wide gateway-call budget, kept under the gateway's own 60-request cap. */
const MAX_TOTAL_CALLS = 55;
/** The client's answer cap; a larger upstream body can never be served, so it is not kept. */
const MAX_ANSWER_BYTES = 2 * 1024 * 1024;
/** Finish reasons the OpenAI wire allows; anything else fails the house's stop-reason check. */
const LEGAL_FINISH = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
/** The token a prompt asks to be repeated: WORD-1234, capitals and exactly four digits. */
const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/g;

type Wire = 'openai' | 'anthropic';

/** One upstream refusal, as the failures-trail header carries it. */
interface FailureRow {
  status: number;
  code: string;
  model: string;
}

/** A checked upstream answer, in everything needed to render it on either client wire. */
interface CheckedAnswer {
  model: string;
  /** True only when every check passed: this answer may be served as the prompt's answer. */
  sound: boolean;
  json: Json | null;
  streamText: string | null;
  streamChunks: Json[];
  finishReason: string | null;
  usageIn: number;
  usageOut: number;
}

/* ------------------------------------------------------------------ configuration */

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/$/, '');
const gatewayBase = (process.env.OPENAI_BASE_URL ?? '').replace(/\/$/, '');
const gatewayKey = process.env.OPENAI_API_KEY ?? '';
const brief: Brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const catalog = buildCatalog(brief);
const servedModels: ServedModel[] = freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));

/** Refusals per model this trial; two or more sink a model to the back of every later ranking. */
const refusals = new Map<string, number>();
/** Gateway calls made this trial, against MAX_TOTAL_CALLS. */
let totalCalls = 0;
/** The manifest row the next completion request answers; prompts arrive in manifest order. */
let nextPrompt = 0;

/* ------------------------------------------------------------------ request reading */

/** Read a whole request body, rejecting past the gateway's own 8 MiB intake cap. */
async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > 8 * 1024 * 1024) throw new Error('request body too large');
    chunks.push(buf);
  }
  return Buffer.concat(chunks).toString('utf8');
}

/** Send a JSON body in one wire's envelope. */
function sendJson(res: ServerResponse, status: number, value: unknown, headers: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': 'application/json', ...headers });
  res.end(JSON.stringify(value));
}

/** Send an SSE body. */
function sendEvents(res: ServerResponse, text: string, headers: Record<string, string> = {}): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', ...headers });
  res.end(text);
}

/* ------------------------------------------------------------------ needs and ranking */

/** The needs a request body declares on its own: tools, JSON mode, an image to see. */
function bodyRequirements(wire: Wire, body: Json): Requirement {
  const req: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  if (wire === 'openai' && isRecord(body.response_format)) {
    if (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema') req.json = true;
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) req.vision = true;
    }
  }
  return req;
}

/** Manifest needs merged with body needs: a flag either source sets is a need. */
function mergeRequirements(a: Requirement, b: Requirement): Requirement {
  return {
    ...(a.tools === true || b.tools === true ? { tools: true } : {}),
    ...(a.vision === true || b.vision === true ? { vision: true } : {}),
    ...(a.json === true || b.json === true ? { json: true } : {}),
    ...(a.min_context !== undefined ? { min_context: a.min_context } : b.min_context !== undefined ? { min_context: b.min_context } : {}),
  };
}

/**
 * The candidate order for one prompt: free models meeting every need first, then the rest, each
 * group by rating (ties by id); within a group, models with two or more refusals this trial go
 * last — a model that keeps refusing should not eat every prompt's first attempt.
 */
function rankCandidates(requires: Requirement): ModelRow[] {
  const byRank = (a: ModelRow, b: ModelRow): number =>
    b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  const order = (rows: ModelRow[]): ModelRow[] => {
    const fresh = rows.filter((m) => (refusals.get(m.id) ?? 0) < 2).sort(byRank);
    const demoted = rows.filter((m) => (refusals.get(m.id) ?? 0) >= 2).sort(byRank);
    return [...fresh, ...demoted];
  };
  const free = freeModels(brief);
  return [...order(free.filter((m) => meetsRequirements(requires, m))), ...order(free.filter((m) => !meetsRequirements(requires, m)))];
}

/**
 * The token the prompt asks to be repeated, when the request names exactly one distinct one.
 * The house's prompts each name one ("Reply with exactly the word X"); with zero or several
 * distinct candidates there is nothing safe to verify against and the check is skipped.
 */
function expectedToken(body: Json): string | null {
  const matches = new Set(JSON.stringify(body).match(TOKEN_PATTERN) ?? []);
  return matches.size === 1 ? [...matches][0] : null;
}

/* ------------------------------------------------------------------ the gateway call */

/** The Chat Completions body one attempt dials: the client's request with the model set. */
function gatewayBody(wire: Wire, clientBody: Json, model: string): Json {
  if (wire === 'anthropic') {
    const body = anthropicToOpenAI(clientBody, model);
    if (body.max_tokens === undefined && body.max_completion_tokens === undefined) body.max_tokens = 1024;
    return body;
  }
  const body: Json = { ...clientBody, model };
  delete body.store;
  if (body.stream === true) {
    body.stream_options = { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true };
  }
  if (body.max_tokens === undefined && body.max_completion_tokens === undefined) body.max_tokens = 1024;
  return body;
}

/** The refusal code a provider's error body carries: its string code, else its type, else the status. */
function refusalCode(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code.slice(0, 64);
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type.slice(0, 64);
      if (typeof parsed.error.code === 'number') return String(parsed.error.code).slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

interface AttemptOutcome {
  /** Set when the gateway answered 2xx: the raw body, buffered in full. */
  text: string | null;
  /** Set when the attempt was refused: the trail row for it. */
  failure: FailureRow | null;
}

/** Dial one model once. Every outcome is data; this never throws. */
async function dial(model: string, body: Json, stream: boolean, timeoutMs: number): Promise<AttemptOutcome> {
  totalCalls += 1;
  let response: Response;
  try {
    response = await fetch(`${gatewayBase}/chat/completions`, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${gatewayKey}`,
        'content-type': 'application/json',
        accept: stream ? 'text/event-stream' : 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (err) {
    const timedOut = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError');
    return { text: null, failure: { status: timedOut ? 504 : 502, code: timedOut ? 'timeout' : 'transport_error', model } };
  }
  const text = await response.text().catch(() => '');
  if (response.status < 200 || response.status >= 300) {
    return { text: null, failure: { status: response.status, code: refusalCode(text, response.status), model } };
  }
  return { text, failure: null };
}

/* ------------------------------------------------------------------ checking an answer */

const usageOf = (v: unknown): { input: number; output: number } | null => {
  if (!isRecord(v)) return null;
  const input = v.prompt_tokens;
  const output = v.completion_tokens;
  if (typeof input !== 'number' || typeof output !== 'number') return null;
  return { input, output };
};

/** Whether an answer's text proves the prompt was served: the named token, or any real content. */
function provesContent(text: string, hasToolCalls: boolean, token: string | null): boolean {
  if (token !== null) return text.includes(token);
  return text !== '' || hasToolCalls;
}

/** Check a buffered JSON answer the way the house will. Null when nothing servable came back. */
function checkJsonAnswer(text: string, model: string, token: string | null): CheckedAnswer | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const fallback: CheckedAnswer = { model, sound: false, json: parsed, streamText: null, streamChunks: [], finishReason: null, usageIn: 0, usageOut: 0 };
  const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message) || choice.message.role !== 'assistant') return fallback;
  if (typeof choice.finish_reason !== 'string' || !LEGAL_FINISH.has(choice.finish_reason)) return fallback;
  const usage = usageOf(parsed.usage);
  if (usage === null || usage.input < 0 || usage.output < 1) return fallback;
  const content = choice.message.content;
  const answerText = typeof content === 'string' ? content : textOfContent(content);
  const hasToolCalls = Array.isArray(choice.message.tool_calls) && choice.message.tool_calls.length > 0;
  if (!provesContent(answerText, hasToolCalls, token)) {
    return { ...fallback, finishReason: choice.finish_reason, usageIn: usage.input, usageOut: usage.output };
  }
  return { ...fallback, sound: true, finishReason: choice.finish_reason, usageIn: usage.input, usageOut: usage.output };
}

/** Check a buffered SSE answer the way the house will. Null only when there is nothing to render. */
function checkStreamAnswer(text: string, model: string, token: string | null): CheckedAnswer | null {
  const fallback: CheckedAnswer = { model, sound: false, json: null, streamText: text, streamChunks: [], finishReason: null, usageIn: 0, usageOut: 0 };
  const parsed = parseOpenAIStream(text);
  if (parsed.malformed || !parsed.sawDone || parsed.chunks.length === 0) return fallback;
  let answerText = '';
  let finishSeen = false;
  let finishCount = 0;
  let contentAfterFinish = false;
  let finishReason: string | null = null;
  let usage: { input: number; output: number } | null = null;
  let hasToolCalls = false;
  for (const chunk of parsed.chunks) {
    if (!Array.isArray(chunk.choices)) return fallback;
    const seen = usageOf(chunk.usage);
    if (seen !== null) usage = seen;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    const piece = typeof delta.content === 'string' ? delta.content : '';
    if (finishSeen && piece !== '') contentAfterFinish = true;
    answerText += piece;
    if (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) hasToolCalls = true;
    if (typeof choice.finish_reason === 'string') {
      finishCount += 1;
      finishSeen = true;
      finishReason = choice.finish_reason;
    }
  }
  if (finishCount !== 1 || contentAfterFinish || finishReason === null || !LEGAL_FINISH.has(finishReason)) return fallback;
  if (usage === null || usage.input < 0 || usage.output < 1) return fallback;
  const detail = { streamChunks: parsed.chunks, finishReason, usageIn: usage.input, usageOut: usage.output };
  if (!provesContent(answerText, hasToolCalls, token)) return { ...fallback, ...detail };
  return { ...fallback, ...detail, sound: true };
}

/* ------------------------------------------------------------------ serving a prompt */

/** Render a checked answer on the client's wire, with the failures trail when one is owed. */
function renderAnswer(res: ServerResponse, wire: Wire, stream: boolean, answer: CheckedAnswer, trail: FailureRow[] | null): void {
  const headers: Record<string, string> = {};
  if (trail !== null && trail.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(trail.slice(0, 8));
  if (wire === 'openai') {
    if (stream) {
      sendEvents(res, rewriteOpenAIStream(answer.streamText ?? '', answer.model), headers);
    } else {
      sendJson(res, 200, { ...(answer.json ?? {}), model: answer.model }, headers);
    }
    return;
  }
  if (stream) {
    sendEvents(
      res,
      anthropicStreamFromChunks(
        answer.streamChunks,
        answer.model,
        { input_tokens: answer.usageIn, output_tokens: answer.usageOut },
        anthropicStopReason(answer.finishReason),
      ),
      headers,
    );
  } else {
    sendJson(res, 200, openAIToAnthropic(answer.json ?? {}, answer.model), headers);
  }
}

/** Serve one completion request: rank, dial, check, fail over, render. */
async function serveCompletion(wire: Wire, clientBody: Json, res: ServerResponse): Promise<void> {
  const stream = clientBody.stream === true;
  const row = brief.prompts[nextPrompt];
  nextPrompt += 1;
  const manifest = row !== undefined && row.api === wire ? row.requires : {};
  const requires = mergeRequirements(manifest, bodyRequirements(wire, clientBody));
  const token = expectedToken(clientBody);
  const candidates = rankCandidates(requires);

  const failures: FailureRow[] = [];
  let lastResort: CheckedAnswer | null = null;
  const startedAt = Date.now();
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS || totalCalls >= MAX_TOTAL_CALLS) break;
    const remaining = PROMPT_DEADLINE_MS - (Date.now() - startedAt);
    if (remaining <= 1_000) break;
    attempts += 1;
    const outcome = await dial(candidate.id, gatewayBody(wire, clientBody, candidate.id), stream, Math.min(ATTEMPT_TIMEOUT_MS, remaining));
    if (outcome.failure !== null) {
      failures.push(outcome.failure);
      refusals.set(candidate.id, (refusals.get(candidate.id) ?? 0) + 1);
      continue;
    }
    const text = outcome.text ?? '';
    if (Buffer.byteLength(text) > MAX_ANSWER_BYTES) continue;
    const checked = stream ? checkStreamAnswer(text, candidate.id, token) : checkJsonAnswer(text, candidate.id, token);
    if (checked === null) continue;
    if (checked.sound) {
      renderAnswer(res, wire, stream, checked, failures.length > 0 ? failures : null);
      return;
    }
    if (lastResort === null) lastResort = checked;
  }
  if (lastResort !== null) {
    renderAnswer(res, wire, stream, lastResort, null);
    return;
  }
  if (wire === 'anthropic') sendJson(res, 502, anthropicError('api_error', 'every free model refused or could not serve this request'));
  else sendJson(res, 502, openAIError('every free model refused or could not serve this request', 'upstream_error', 'upstream_refused'));
}

/* ------------------------------------------------------------------ the HTTP surface */

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const anthropic = req.headers['x-api-key'] !== undefined;
    sendJson(res, 200, anthropic ? anthropicModelList(servedModels) : openAIModelList(servedModels));
    return;
  }
  const wire: Wire | null =
    req.method === 'POST' && url.pathname === '/v1/chat/completions'
      ? 'openai'
      : req.method === 'POST' && url.pathname === '/v1/messages'
        ? 'anthropic'
        : null;
  if (wire === null) {
    if (url.pathname.startsWith('/v1/messages')) sendJson(res, 404, anthropicError('not_found_error', 'not found'));
    else sendJson(res, 404, openAIError('not found', 'invalid_request_error', 'not_found'));
    return;
  }
  let body: Json;
  try {
    const parsed: unknown = JSON.parse(await readBody(req));
    if (!isRecord(parsed)) throw new Error('request body is not a JSON object');
    body = parsed;
  } catch {
    if (wire === 'anthropic') sendJson(res, 400, anthropicError('invalid_request_error', 'request body is not valid JSON'));
    else sendJson(res, 400, openAIError('request body is not valid JSON', 'invalid_request_error', 'invalid_json'));
    return;
  }
  await serveCompletion(wire, body, res);
}

/* ------------------------------------------------------------------ trial lifecycle */

/** POST the catalog to the env. Retried briefly on transport errors only; an HTTP refusal is final. */
async function register(port: number): Promise<void> {
  let lastError: Error | null = null;
  for (let i = 0; i < 20; i++) {
    try {
      const response = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      if (!response.ok) throw new Error(`register refused: HTTP ${response.status} ${await response.text()}`);
      return;
    } catch (err) {
      if (err instanceof Error && err.message.startsWith('register refused')) throw err;
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll the env's work endpoint until it reports the trial done ({ done: true }). */
async function awaitDone(): Promise<void> {
  for (;;) {
    try {
      const response = await fetch(`${envUrl}/work`);
      if (response.ok) {
        const work: unknown = await response.json();
        if (isRecord(work) && work.done === true) return;
      }
    } catch {
      // The env is briefly unreachable: keep polling.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (brief.error !== null) console.error(`brief: ${brief.error}`);
  const server = createServer((req, res) => {
    handle(req, res).catch((err: unknown) => {
      console.error('request failed:', err);
      if (!res.headersSent) sendJson(res, 500, openAIError('internal error', 'api_error', 'internal'));
      else res.end();
    });
  });
  server.listen(0, '0.0.0.0');
  await once(server, 'listening');
  const address = server.address() as AddressInfo;
  if (address.port === 8080) throw new Error('the OS assigned port 8080, which the gateway owns');
  await register(address.port);
  await awaitDone();
  server.close();
}

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

main().then(
  () => process.exit(0),
  (err: unknown) => {
    console.error(err);
    process.exit(1);
  },
);
