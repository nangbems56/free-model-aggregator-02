// The brief: where the trial's model list and prompt manifest live, and the catalog built from them.
//
// A trial hands this server one file (MAKEFUN_PROMPT_FILE) whose prose carries two JSON documents,
// each found by the `schema` string it carries under this competition's prefix:
//
//   - free-model-aggregator-02/models/v1: the free models the gateway will route, one row each, with
//     the id to dial (`openrouter/<vendor>/<model>:free`), the house's quality rating in
//     milli-points, the context length, and the capability flags (tools, vision, JSON mode). A row is
//     free unless it says `free: false`; only free rows may serve a prompt or enter the catalog.
//   - free-model-aggregator-02/prompts/v1: the prompts the env's client will send, in sending order,
//     each with its id, its wire (openai or anthropic), whether it streams, its weight (1 or 2) and
//     the requirements a model must meet to serve it well.
//
// The catalog is the one document `--list` prints and the register call carries: one provider,
// openrouter, at its public base URL (never the gateway's trial address, which differs between a
// live pass and its replay while the catalog must not), naming the entrant's key variable, with the
// free models exactly as the brief lists them.
//
// Everything in this file is a pure function of the brief's bytes: no clock, no randomness, so the
// catalog a live pass registers is byte for byte the one its replay registers.

import { readFileSync } from 'node:fs';

const SCHEMA_PREFIX = 'free-model-aggregator-02/';
export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** OpenRouter's public API root: the catalog's base_url, whatever the trial's gateway address is. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The variable an entrant pushes their OpenRouter key under; the catalog names it, never holds it. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One model row of the brief. `id` is the dialable gateway id; `rating_milli` the house's rating. */
export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs from a model. A flag set true is a hard need; min_context a floor. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt of the brief's manifest, in the order the env sends them. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief, read. `error` says why a document is missing, when one is. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** The catalog document: what `--list` prints and what /register carries, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** The index just past the `}` that closes the object opening at `start`, or -1 when it never closes. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in the order they appear. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a standalone JSON object: keep scanning.
    }
  }
  return found;
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** The first competition document holding an array under one of `keys`; the document and the key. */
function documentWith(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).startsWith(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The dialable id is the row's `id` (or its `model`); a row that spells the bare
 * OpenRouter slug while naming the openrouter provider is prefixed the way the gateway routes it.
 */
function modelRow(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const provider = asString(v.provider) !== '' ? asString(v.provider) : asString(v.wire);
  const id = provider === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: provider === '' ? 'openrouter' : provider,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: v.free !== false,
  };
}

/** One prompt row; only the requirement flags the brief actually sets survive. */
function promptRow(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
      ...(asNumber(req.min_context) !== null ? { min_context: asNumber(req.min_context) as number } : {}),
    },
  };
}

/** Read and parse the brief at `path`. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWith(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief holds no ${MODELS_SCHEMA} document` };
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const row = modelRow(v);
    return row === null ? [] : [row];
  });
  const promptsDoc = documentWith(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief holds no ${PROMPTS_SCHEMA} document` };
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const row = promptRow(v);
    return row === null ? [] : [row];
  });
  return { models, prompts, error: null };
}

/** The brief's free models, in the brief's order. Only these may be dialled or catalogued. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/**
 * Whether a model meets a prompt's requirements. A capability the brief leaves unknown (null) never
 * meets a requirement that asks for it; a context floor needs a known window at or above it.
 */
export function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.context_length === null || model.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief makes: one openrouter provider holding each free model once, in brief order.
 * The scored column compares id sets alone, but the document is also part of the trial's export, so
 * it is built once, deterministically, and the same object is registered and printed by `--list`.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return { providers: [{ id: 'openrouter', base_url: OPENROUTER_PUBLIC_BASE, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models }] };
}
