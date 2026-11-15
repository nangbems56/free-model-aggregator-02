# Free-model aggregator — entry for free-model-aggregator-02

One server that answers OpenAI Chat Completions and Anthropic Messages requests by picking the
best free model for each prompt and dialling it through the house gateway.

## Layout

- `main.ts` — the server and the serving engine: candidate ranking, the attempt/failover loop,
  answer checking, rendering, registration and the work poll.
- `lib/providers.ts` — the brief: parsing the model ratings and prompt manifest out of
  `MAKEFUN_PROMPT_FILE`, and building the catalog that `--list` prints and `/register` carries.
- `lib/wire.ts` — the wire translations: Anthropic ↔ OpenAI request/response/stream conversion,
  the two `/v1/models` list shapes, and the model-field rewrite every answer goes through.

## How it serves a prompt

1. **Needs** come from the brief's manifest row for the prompt (arrival order), cross-checked
   against the request body (a tools array, a `json_object`/`json_schema` response format, or an
   image part can only add needs).
2. **Candidates** are the free models meeting every need, best house rating first; models that
   refused twice this trial sink to the back. If no model meets the needs, the rest follow in
   rating order — a lesser answer still beats an error.
3. **Attempts**: at most 5 per prompt, 13s each, inside a 26s prompt deadline; at most 55 gateway
   calls per trial. Every attempt names a different model, so no byte-identical request is ever
   sent twice. Refusals (non-2xx, timeout, transport error) are recorded and the next candidate
   is dialled.
4. **Checking before serving**: a 2xx is buffered in full and checked the way the house checks —
   wire shape, legal finish reason, usage with at least one output token, and the prompt's named
   token present in the answer text. A 2xx that fails is set aside (the first is kept as a
   last-resort answer) and the next candidate is tried.
5. **The failures trail**: an answer served after refusals carries the
   `x-makefun-upstream-failures` header listing each refusal (status, provider error code, model
   dialled). The trail rides only on answers that passed the checks.
6. **Determinism**: every decision is a pure function of the brief, the requests, and the
   gateway's answers. Timeouts only bound waiting; nothing time-based is written into a request
   or an answer, so a replay reproduces the live pass exactly.

## Runtime

Zero dependencies — Node's standard library only (`node:http`, global `fetch`). The harness runs
`node main.ts`; `--list` prints the catalog and exits. The server listens on an OS-assigned port,
registers `{ protocol_version: 1, port, catalog }` with `$MAKEFUN_ENV_URL/register`, dials
`$OPENAI_BASE_URL/chat/completions` with `$OPENAI_API_KEY`, and polls `$MAKEFUN_ENV_URL/work`
until the trial is done, then exits 0.
