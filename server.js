'use strict';

/*
 * JANITORAI MULTI-PROVIDER ROLEPLAY PROXY v3
 *
 * Node.js 20+ | Zero external dependencies
 * Built for personal long-term narrative RP on free Render.
 *
 * PROVIDERS
 *   - NVIDIA NIM
 *   - Google Gemini (OpenAI-compatible endpoint)
 *
 * ROUTES
 *   GET  /
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/chat/completions
 *
 * MODEL ROUTING
 *   gemini/<model-id>       -> Google Gemini
 *   google/gemini-<model>   -> Google Gemini alias
 *   everything else         -> NVIDIA
 *
 * KEYS
 *   No keys are stored on the server and there is no proxy key.
 *   The provider API key is taken from each request's Authorization header
 *   (JanitorAI's "API key" field) and forwarded to the provider the model
 *   routes to. Keys are never logged or cached.
 *
 * MEMORY
 *   Stateless. Continuity depends entirely on the transcript + lorebook
 *   JanitorAI supplies with each request.
 */

const http = require('node:http');
const crypto = require('node:crypto');
const fs = require('node:fs');
const nodePath = require('node:path');

const env = process.env;

// =============================================================================
// 1. CONFIGURATION
// =============================================================================

function intEnv(name, fallback, min = 1, max = Number.MAX_SAFE_INTEGER) {
  const raw = env[name];
  const value = raw === undefined || raw.trim() === '' ? fallback : Number(raw);

  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}.`);
  }
  return value;
}

function boolEnv(name, fallback) {
  if (env[name] === undefined) return fallback;

  const value = String(env[name]).trim().toLowerCase();
  if (['true', '1', 'yes'].includes(value)) return true;
  if (['false', '0', 'no'].includes(value)) return false;

  throw new Error(`${name} must be true or false.`);
}

function csvEnv(name, fallback = []) {
  if (!env[name]) return fallback;
  return env[name].split(',').map(value => value.trim()).filter(Boolean);
}

function baseUrlEnv(name, fallback) {
  const value = (env[name] || fallback).trim().replace(/\/+$/, '');

  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${name} must be a valid URL.`);
  }

  if (
    parsed.protocol !== 'https:' ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error(
      `${name} must be an HTTPS URL without credentials, query parameters, or fragments.`
    );
  }
  return value;
}

const HOST = env.HOST || '0.0.0.0';
const PORT = intEnv('PORT', 3000, 1, 65535);

const NVIDIA_BASE_URL = baseUrlEnv(
  'NVIDIA_BASE_URL',
  'https://integrate.api.nvidia.com/v1'
);
const GEMINI_BASE_URL = baseUrlEnv(
  'GEMINI_BASE_URL',
  'https://generativelanguage.googleapis.com/v1beta/openai'
);

const DEFAULT_MODEL = (env.DEFAULT_MODEL || '').trim();
const DEFAULT_MAX_TOKENS = intEnv('DEFAULT_MAX_TOKENS', 16384);

const MAX_BODY_BYTES = intEnv('MAX_BODY_BYTES', 20 * 1024 * 1024);
const MAX_RESPONSE_BYTES = intEnv('MAX_RESPONSE_BYTES', 32 * 1024 * 1024);
const MAX_SSE_EVENT_BYTES = intEnv('MAX_SSE_EVENT_BYTES', 2 * 1024 * 1024);

const REQUEST_TIMEOUT_MS = intEnv('REQUEST_TIMEOUT_MS', 600000, 1000);
const MODEL_LIST_TIMEOUT_MS = intEnv('MODEL_LIST_TIMEOUT_MS', 15000, 1000);

// Free Render defaults: low concurrency to stay comfortable under 512 MB.
const NVIDIA_CONCURRENCY = intEnv('NVIDIA_MAX_CONCURRENT_REQUESTS', 1);
const GEMINI_CONCURRENCY = intEnv('GEMINI_MAX_CONCURRENT_REQUESTS', 1);
const REQUEST_QUEUE_LIMIT = intEnv('REQUEST_QUEUE_LIMIT', 4, 0);
const REQUEST_QUEUE_TIMEOUT_MS = intEnv('REQUEST_QUEUE_TIMEOUT_MS', 30000, 1000);

const STRIP_REASONING = boolEnv('STRIP_REASONING', true);
const RP_CONTINUITY_MODE = boolEnv('RP_CONTINUITY_MODE', true);
const ALLOW_UNKNOWN_NVIDIA_MODELS = boolEnv('ALLOW_UNKNOWN_MODELS', true);

const ALLOWED_ORIGINS = csvEnv('CORS_ORIGINS', [
  'https://janitorai.com',
  'https://www.janitorai.com'
]).map(origin => origin.replace(/\/+$/, ''));

const EXTRA_NVIDIA_MODELS = csvEnv('EXTRA_NVIDIA_MODELS');
const EXTRA_GEMINI_MODELS = csvEnv('EXTRA_GEMINI_MODELS');

const MISTRAL_PREFIX_MODE = (env.MISTRAL_ASSISTANT_PREFIX_MODE || 'translate')
  .trim()
  .toLowerCase();

if (!['translate', 'preserve', 'off'].includes(MISTRAL_PREFIX_MODE)) {
  throw new Error(
    'MISTRAL_ASSISTANT_PREFIX_MODE must be translate, preserve, or off.'
  );
}

function logEvent(type, fields = {}, isError = false) {
  const line = JSON.stringify({ type, ...fields });
  if (isError) console.error(line);
  else console.log(line);
}

// =============================================================================
// 2. ROLEPLAY CONTINUITY SYSTEM
// =============================================================================

const RP_PROMPT_MARKER = 'ROLEPLAY CONTINUITY INSTRUCTIONS';

const EMBEDDED_CONTINUITY_PROMPT = `
ROLEPLAY CONTINUITY INSTRUCTIONS (HIGHEST PRIORITY):

You are continuing one continuous, mature, uncensored, 18+, fictional narrative roleplay with novel-quality, cinematic prose and slice-of-life pacing.

ABSOLUTE RULES — never violate these:
1. Lorebook / World Bible / Character definitions / Scenario are absolute established fact. Never contradict or invent alternatives.
2. Do not invent new major facts, locations, relationships, history, abilities, or events unless they are the immediate logical consequence of what already exists and the user's latest action.
3. Never restart the scene or re-introduce characters already present.
4. The user's character is under the user's exclusive control. Never write their dialogue, thoughts, emotions, decisions, or voluntary actions.
5. Preserve every established detail of characters, relationships, physical positions, clothing, injuries, possessions, and time.
6. NPCs have independent lives, private goals, and limited knowledge.
7. Prefer quiet, natural, slice-of-life progression and immersive sensory prose over forced drama.
8. If information is missing, preserve uncertainty. Do not invent convenient explanations.
9. Do not conclude scenes, conflicts, and/or scenarios in a single generation. Tend to stop when the user's character would likely need to respond through action or dialogue.

Treat all supplied messages as one continuous timeline. Continue from the actual current endpoint. Characters remember meaningful experiences. Actions have consequences. Relationships evolve. The response should simply continue the story.

ROLEPLAY CONTINUITY INSTRUCTIONS END.
`.trim();

function withMarker(text) {
  return text.includes(RP_PROMPT_MARKER)
    ? text
    : `${RP_PROMPT_MARKER}\n\n${text}`;
}

// Priority: RP_CONTINUITY_PROMPT env var, then an external file (so the
// prompt can be edited without touching code), then the embedded fallback.
function loadContinuityPrompt() {
  const custom = (env.RP_CONTINUITY_PROMPT || '').trim();
  if (custom) return withMarker(custom);

  const candidates = [
    env.RP_CONTINUITY_FILE,
    nodePath.join(__dirname, 'rp-continuity.txt'),
    nodePath.join(process.cwd(), 'rp-continuity.txt')
  ].filter(Boolean);

  for (const file of new Set(candidates)) {
    try {
      const content = fs.readFileSync(file, 'utf8').trim();
      if (content) {
        logEvent('continuity_prompt_loaded', {
          source: file,
          bytes: Buffer.byteLength(content, 'utf8')
        });
        return withMarker(content);
      }
    } catch {
      // Missing or unreadable file: try the next candidate.
    }
  }

  return EMBEDDED_CONTINUITY_PROMPT;
}

const RP_CONTINUITY_PROMPT = loadContinuityPrompt();

const MISTRAL_CONTINUATION_PROMPT = `
Continue the immediately preceding assistant message as an unfinished response prefix.

The existing assistant text is already written content. Continue directly from its endpoint.

Do not repeat, quote, summarize, paraphrase, rewrite, or restart any of the existing assistant text.

Preserve the existing writing style, narrative perspective, formatting, tone, dialogue flow, and sentence structure.

Use the complete supplied conversation history to preserve established characters, chronology, locations, physical positioning, relationships, character knowledge, ongoing actions, and consequences.

Do not introduce a new opening, unnecessary recap, unrelated scene, or artificial transition.

Do not treat the unfinished assistant prefix as a new user request.

The continuation must read as one seamless assistant response with no indication that it was generated separately.
`.trim();

// =============================================================================
// 3. MODEL PROFILES
// =============================================================================

const MODEL_PROFILES = new Map();

function addProfile(id, label, upstream = id, adapter = 'generic', effort = null) {
  MODEL_PROFILES.set(id, {
    id,
    label,
    model: upstream,
    provider: 'nvidia',
    adapter,
    effort
  });
}

// Mistral Large 4
const MISTRAL4 = 'mistralai/mistral-large-4-1t-a52b';
addProfile(MISTRAL4, 'Mistral Large 4 - High Reasoning', MISTRAL4, 'mistral4', 'high');
addProfile(`${MISTRAL4}-medium`, 'Mistral Large 4 - Medium Reasoning', MISTRAL4, 'mistral4', 'medium');
addProfile(`${MISTRAL4}-low`, 'Mistral Large 4 - Low Reasoning', MISTRAL4, 'mistral4', 'low');
addProfile(`${MISTRAL4}-no-thinking`, 'Mistral Large 4 - No Reasoning', MISTRAL4, 'mistral4', 'none');

// GLM (labels state the effort value actually sent)
const GLM = 'z-ai/glm-5.3-flash';
addProfile(GLM, 'GLM 5.3 Flash - High Reasoning', GLM, 'glm', 'high');
addProfile(`${GLM}-low`, 'GLM 5.3 Flash - Low Reasoning', GLM, 'glm', 'low');
addProfile(`${GLM}-medium`, 'GLM 5.3 Flash - High Reasoning', GLM, 'glm', 'high');
addProfile(`${GLM}-high`, 'GLM 5.3 Flash - Max Reasoning', GLM, 'glm', 'max');

// MiniMax
const MINIMAX = 'minimaxai/minimax-m3.1-flash';
addProfile(MINIMAX, 'MiniMax M3.1 Flash - Medium Reasoning', MINIMAX, 'minimax', 'medium');
for (const effort of ['low', 'medium', 'high', 'xhigh', 'max']) {
  addProfile(
    `${MINIMAX}-${effort}`,
    `MiniMax M3.1 Flash - ${effort.toUpperCase()} Reasoning`,
    MINIMAX,
    'minimax',
    effort
  );
}

// Nemotron VL
const NEMOTRON_VL = 'nvidia/nemotron-3.5-super-vl-120b-a12b';
addProfile(NEMOTRON_VL, 'Nemotron 3.5 Super VL - High Reasoning', NEMOTRON_VL, 'nemotron-vl', 'high');
addProfile(`${NEMOTRON_VL}-no-thinking`, 'Nemotron 3.5 Super VL - No Reasoning', NEMOTRON_VL, 'nemotron-vl', 'none');
addProfile(`${NEMOTRON_VL}-low`, 'Nemotron 3.5 Super VL - Low Reasoning', NEMOTRON_VL, 'nemotron-vl', 'low');
addProfile(`${NEMOTRON_VL}-high`, 'Nemotron 3.5 Super VL - High Reasoning', NEMOTRON_VL, 'nemotron-vl', 'high');

// Muse Glimmer (Meta 30B agentic, available on NVIDIA NIM)
const MUSE_GLIMMER = 'meta/muse-glimmer-30b';
addProfile(MUSE_GLIMMER, 'Muse Glimmer 30B', MUSE_GLIMMER, 'generic');
addProfile(`${MUSE_GLIMMER}-high`, 'Muse Glimmer 30B - High Reasoning', MUSE_GLIMMER, 'generic');

// Other NVIDIA models
addProfile('stepfun-ai/step-5-preview', 'Step 5 Preview');
addProfile('poolside/laguna-xs-2.1', 'Laguna XS 2.1');
addProfile('nvidia/nemotron-3.5-bolt-e4b', 'Nemotron 3.5 Bolt E4B');

// =============================================================================
// 4. MODEL RESOLUTION AND CLIENT KEYS
// =============================================================================

function geminiProfile(id, model) {
  return { id, model, provider: 'gemini', adapter: 'generic', label: id };
}

function resolveModel(requested) {
  if (typeof requested !== 'string' || !requested.trim()) return null;

  const id = requested.trim();

  const known = MODEL_PROFILES.get(id);
  if (known) return known;

  if (id.startsWith('gemini/')) {
    const model = id.slice('gemini/'.length);
    return model ? geminiProfile(id, model) : null;
  }

  if (id.startsWith('google/')) {
    const model = id.slice('google/'.length);
    if (model.startsWith('gemini-') || model.startsWith('models/gemini-')) {
      return geminiProfile(id, model);
    }
  }

  if (ALLOW_UNKNOWN_NVIDIA_MODELS) {
    return { id, model: id, provider: 'nvidia', adapter: 'generic', label: id };
  }

  return null;
}

function providerDetails(provider) {
  if (provider === 'gemini') {
    return { name: 'Gemini', baseUrl: GEMINI_BASE_URL };
  }
  return { name: 'NVIDIA', baseUrl: NVIDIA_BASE_URL };
}

// The provider key comes from the client on every request. It is used for
// that request only and is never logged or stored.
function getClientKey(req) {
  const match = String(req.headers.authorization || '').match(/^Bearer\s+(.+)$/i);
  const key = String(match?.[1] || req.headers['x-api-key'] || '').trim();

  // Printable ASCII only, so it is always a valid header value upstream.
  return /^[\x21-\x7e]{1,512}$/.test(key) ? key : '';
}

function isNvidiaKey(key) {
  return key.startsWith('nvapi-');
}

const MISSING_KEY_MESSAGE =
  "No API key was sent. Put your NVIDIA or Gemini API key in JanitorAI's API key field.";

// =============================================================================
// 5. CONCURRENCY QUEUE
// =============================================================================

function abortReason(signal) {
  return signal.reason || new Error('Request aborted.');
}

function httpError(message, status, type) {
  const error = new Error(message);
  error.status = status;
  if (type) error.type = type;
  return error;
}

class Queue {
  constructor(limit, maxQueue, timeoutMs) {
    this.limit = limit;
    this.maxQueue = maxQueue;
    this.timeoutMs = timeoutMs;
    this.active = 0;
    this.waiting = [];
  }

  acquire(signal) {
    if (signal.aborted) return Promise.reject(abortReason(signal));

    if (this.active < this.limit) {
      this.active++;
      return Promise.resolve(this.makeRelease());
    }

    if (this.waiting.length >= this.maxQueue) {
      return Promise.reject(
        httpError('Provider queue is full.', 503, 'server_overloaded')
      );
    }

    return new Promise((resolve, reject) => {
      const entry = { settled: false, timer: null, onAbort: null, grant: null };

      const cleanup = () => {
        clearTimeout(entry.timer);
        signal.removeEventListener('abort', entry.onAbort);
      };

      const rejectEntry = error => {
        if (entry.settled) return;
        entry.settled = true;

        const index = this.waiting.indexOf(entry);
        if (index >= 0) this.waiting.splice(index, 1);

        cleanup();
        reject(error);
      };

      entry.onAbort = () => rejectEntry(abortReason(signal));

      entry.grant = () => {
        if (entry.settled) return;
        entry.settled = true;
        cleanup();
        resolve(this.makeRelease());
      };

      entry.timer = setTimeout(() => {
        rejectEntry(
          httpError('Provider queue wait timed out.', 503, 'queue_timeout')
        );
      }, this.timeoutMs);
      entry.timer.unref?.();

      signal.addEventListener('abort', entry.onAbort, { once: true });
      this.waiting.push(entry);
    });
  }

  makeRelease() {
    let released = false;

    return () => {
      if (released) return;
      released = true;
      this.active = Math.max(0, this.active - 1);
      this.dispatch();
    };
  }

  dispatch() {
    while (this.waiting.length && this.active < this.limit) {
      const entry = this.waiting.shift();
      if (!entry || entry.settled) continue;
      this.active++;
      entry.grant();
    }
  }
}

const queues = {
  nvidia: new Queue(NVIDIA_CONCURRENCY, REQUEST_QUEUE_LIMIT, REQUEST_QUEUE_TIMEOUT_MS),
  gemini: new Queue(GEMINI_CONCURRENCY, REQUEST_QUEUE_LIMIT, REQUEST_QUEUE_TIMEOUT_MS)
};

// =============================================================================
// 6. BASIC HTTP AND JSON HELPERS
// =============================================================================

function sendJson(res, status, payload) {
  if (res.destroyed || res.writableEnded || res.headersSent) return;

  const body = JSON.stringify(payload);

  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store'
  });
  res.end(body);
}

function sendError(res, status, message, type = 'proxy_error') {
  sendJson(res, status, {
    error: {
      message: String(message || 'Unknown error').slice(0, 4000),
      type
    }
  });
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractErrorMessage(data, fallback = 'Provider request failed.') {
  const parsed = typeof data === 'string' ? parseJson(data) || data : data;

  if (typeof parsed === 'string') {
    return parsed.trim().slice(0, 3000) || fallback;
  }

  // Gemini sometimes wraps its error object in an array.
  const source = Array.isArray(parsed) ? parsed[0] : parsed;

  if (source && typeof source === 'object') {
    const message =
      source.error?.message ||
      (typeof source.error === 'string' ? source.error : null) ||
      source.message ||
      source.detail ||
      source.title;

    if (typeof message === 'string' && message) {
      return message.slice(0, 3000);
    }
  }

  return fallback;
}

function abortableDelay(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortReason(signal));
      return;
    }

    const onAbort = () => {
      clearTimeout(timer);
      reject(abortReason(signal));
    };

    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    timer.unref?.();

    signal.addEventListener('abort', onAbort, { once: true });
  });
}

// =============================================================================
// 7. LIMITED RESPONSE READING
// =============================================================================

async function readLimited(response, limit = MAX_RESPONSE_BYTES) {
  if (!response.body) return { text: '', truncated: false };

  const reader = response.body.getReader();
  const chunks = [];

  let total = 0;
  let truncated = false;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = Buffer.from(value);
      const remaining = limit - total;

      if (chunk.length > remaining) {
        if (remaining > 0) chunks.push(chunk.subarray(0, remaining));
        truncated = true;
        break;
      }

      chunks.push(chunk);
      total += chunk.length;
    }
  } finally {
    if (truncated) {
      try {
        await reader.cancel();
      } catch {}
    }
    reader.releaseLock();
  }

  return { text: Buffer.concat(chunks).toString('utf8'), truncated };
}

// =============================================================================
// 8. REQUEST BODY READING
// =============================================================================

// Reads the next body chunk, but gives up as soon as the request is aborted
// (timeout, shutdown, client disconnect) so a stalled upload cannot hang.
function nextWithAbort(iterator, signal) {
  if (signal.aborted) return Promise.reject(abortReason(signal));

  return new Promise((resolve, reject) => {
    const onAbort = () => reject(abortReason(signal));

    signal.addEventListener('abort', onAbort, { once: true });

    iterator.next().then(
      value => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      error => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
}

async function readRequestBody(req, signal) {
  const declaredLength = Number(req.headers['content-length'] || 0);

  if (Number.isFinite(declaredLength) && declaredLength > MAX_BODY_BYTES) {
    throw httpError('Request body is too large.', 413, 'invalid_request_error');
  }

  const chunks = [];
  const iterator = req[Symbol.asyncIterator]();

  let total = 0;
  let tooLarge = false;

  // An oversized body is drained rather than cut off, so the 413 response
  // can still be delivered on the same connection.
  while (true) {
    const result = await nextWithAbort(iterator, signal);
    if (result.done) break;

    total += result.value.length;

    if (total > MAX_BODY_BYTES) {
      tooLarge = true;
      chunks.length = 0;
    } else if (!tooLarge) {
      chunks.push(result.value);
    }
  }

  if (tooLarge) {
    throw httpError('Request body is too large.', 413, 'invalid_request_error');
  }

  const parsed = parseJson(Buffer.concat(chunks).toString('utf8'));

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw httpError(
      'Request body must be a valid JSON object.',
      400,
      'invalid_request_error'
    );
  }

  return parsed;
}

// =============================================================================
// 9. CORS
// =============================================================================

function applyCors(req, res) {
  const rawOrigin = req.headers.origin;
  const origin = typeof rawOrigin === 'string' ? rawOrigin.replace(/\/+$/, '') : '';

  const wildcard = ALLOWED_ORIGINS.includes('*');
  const allowed = !origin || wildcard || ALLOWED_ORIGINS.includes(origin);

  if (!allowed) {
    sendError(res, 403, 'Origin not allowed.', 'cors_error');
    return false;
  }

  if (origin) {
    res.setHeader('Access-Control-Allow-Origin', wildcard ? '*' : rawOrigin);
    res.setHeader('Vary', 'Origin');
  }

  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader(
    'Access-Control-Allow-Headers',
    req.headers['access-control-request-headers'] || 'Authorization, Content-Type'
  );
  res.setHeader('Access-Control-Expose-Headers', 'X-Request-ID');
  res.setHeader('Access-Control-Max-Age', '86400');

  return true;
}

// =============================================================================
// 10. REQUEST VALIDATION
// =============================================================================

const VALID_ROLES = new Set([
  'system',
  'developer',
  'context',
  'user',
  'assistant',
  'tool'
]);

function validateChatRequest(body) {
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return 'messages must be a non-empty array.';
  }

  if (body.messages.length > 10000) {
    return 'messages contains too many entries.';
  }

  if (
    body.model !== undefined &&
    (typeof body.model !== 'string' || !body.model.trim())
  ) {
    return 'model must be a non-empty string.';
  }

  for (let i = 0; i < body.messages.length; i++) {
    const message = body.messages[i];

    if (!message || typeof message !== 'object' || Array.isArray(message)) {
      return `Message ${i + 1} must be an object.`;
    }

    if (typeof message.role !== 'string' || !VALID_ROLES.has(message.role)) {
      return `Message ${i + 1} has an unsupported role.`;
    }

    if (
      message.content !== undefined &&
      message.content !== null &&
      typeof message.content !== 'string' &&
      !Array.isArray(message.content)
    ) {
      return `Message ${i + 1} content must be a string, null, or content-part array.`;
    }

    if (
      message.role === 'tool' &&
      (typeof message.tool_call_id !== 'string' || !message.tool_call_id)
    ) {
      return `Tool message ${i + 1} is missing tool_call_id.`;
    }
  }

  const ranges = {
    temperature: [0, 2],
    top_p: [0, 1],
    frequency_penalty: [-2, 2],
    presence_penalty: [-2, 2]
  };

  for (const [key, [min, max]] of Object.entries(ranges)) {
    const value = body[key];
    if (value === undefined) continue;

    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return `${key} must be a finite number.`;
    }
    if (value < min || value > max) {
      return `${key} must be between ${min} and ${max}.`;
    }
  }

  for (const key of ['max_tokens', 'max_completion_tokens']) {
    if (
      body[key] !== undefined &&
      (!Number.isInteger(body[key]) || body[key] < 1)
    ) {
      return `${key} must be a positive integer.`;
    }
  }

  if (body.stream !== undefined && typeof body.stream !== 'boolean') {
    return 'stream must be a boolean.';
  }

  if (
    body.n !== undefined &&
    (!Number.isInteger(body.n) || body.n < 1 || body.n > 16)
  ) {
    return 'n must be an integer between 1 and 16.';
  }

  if (
    body.stop !== undefined &&
    typeof body.stop !== 'string' &&
    !Array.isArray(body.stop)
  ) {
    return 'stop must be a string or an array of strings.';
  }

  if (
    Array.isArray(body.stop) &&
    (body.stop.length > 16 || body.stop.some(value => typeof value !== 'string'))
  ) {
    return 'stop must contain no more than 16 strings.';
  }

  return null;
}

// =============================================================================
// 11. MESSAGE PREPARATION
// =============================================================================

// "developer" and "context" are not understood by most upstream chat
// templates, so they are sent as "system" for every provider.
function normalizeRoles(messages) {
  return messages.map(message => ({
    ...message,
    role: ['developer', 'context'].includes(message.role)
      ? 'system'
      : message.role
  }));
}

function contentContainsMarker(content) {
  if (typeof content === 'string') return content.includes(RP_PROMPT_MARKER);

  if (Array.isArray(content)) {
    return content.some(part => {
      if (typeof part === 'string') return part.includes(RP_PROMPT_MARKER);
      return (
        part &&
        typeof part === 'object' &&
        typeof part.text === 'string' &&
        part.text.includes(RP_PROMPT_MARKER)
      );
    });
  }

  return false;
}

// Expects role-normalized messages. Inserts the continuity prompt after the
// leading system block and before the conversation.
function addContinuityInstruction(messages) {
  const result = messages.map(message => ({ ...message }));

  if (!RP_CONTINUITY_MODE || !RP_CONTINUITY_PROMPT) return result;

  const alreadyPresent = result.some(
    message =>
      message.role === 'system' && contentContainsMarker(message.content)
  );
  if (alreadyPresent) return result;

  let insertionIndex = 0;
  while (
    insertionIndex < result.length &&
    result[insertionIndex].role === 'system'
  ) {
    insertionIndex++;
  }

  result.splice(insertionIndex, 0, {
    role: 'system',
    content: RP_CONTINUITY_PROMPT
  });

  return result;
}

const MISTRAL_MESSAGE_FIELDS = [
  'content',
  'name',
  'tool_call_id',
  'tool_calls',
  'function_call',
  'refusal',
  'reasoning_content'
];

function trimMistralMessages(messages) {
  return messages.map(message => {
    const output = { role: message.role };

    for (const key of MISTRAL_MESSAGE_FIELDS) {
      if (Object.prototype.hasOwnProperty.call(message, key)) {
        output[key] = message[key];
      }
    }

    return output;
  });
}

function translateMistralAssistantPrefix(messages) {
  const result = messages.slice();
  const last = result[result.length - 1];

  if (
    MISTRAL_PREFIX_MODE !== 'translate' ||
    !last ||
    last.role !== 'assistant'
  ) {
    return { messages: result, translated: false };
  }

  if (Array.isArray(last.tool_calls) && last.tool_calls.length > 0) {
    throw new Error(
      'The final Mistral message contains tool_calls without a tool result; it cannot be translated as a narrative prefix.'
    );
  }

  result.push({ role: 'user', content: MISTRAL_CONTINUATION_PROMPT });

  return { messages: result, translated: true };
}

// =============================================================================
// 12. PROVIDER REQUEST ADAPTERS
// =============================================================================

const COMMON_FIELDS = [
  'model',
  'messages',
  'max_tokens',
  'stream',
  'temperature',
  'top_p',
  'frequency_penalty',
  'presence_penalty',
  'seed',
  'stop',
  'tools',
  'tool_choice',
  'reasoning_effort'
];

const MISTRAL_ALLOWED_FIELDS = new Set(COMMON_FIELDS);

// Gemini's OpenAI-compatible endpoint can reject parameters it does not
// know (repetition_penalty, top_k, min_p, ...), so only these are forwarded.
const GEMINI_ALLOWED_FIELDS = new Set([
  ...COMMON_FIELDS,
  'n',
  'user',
  'response_format',
  'stream_options',
  'parallel_tool_calls',
  'extra_body'
]);

const ADAPTER_ALLOWED_FIELDS = new Set([
  ...COMMON_FIELDS,
  'parallel_tool_calls',
  'response_format',
  'stream_options',
  'chat_template_kwargs',
  'repetition_penalty',
  'top_k',
  'min_p',
  'n',
  'user',
  'logprobs',
  'top_logprobs',
  'logit_bias',
  'functions',
  'function_call'
]);

function copyAllowed(source, allowed) {
  const output = {};
  for (const [key, value] of Object.entries(source)) {
    if (allowed.has(key)) output[key] = value;
  }
  return output;
}

function prepareProviderRequest(input, profile) {
  const body = { ...input, model: profile.model };

  if (body.max_tokens === undefined && body.max_completion_tokens !== undefined) {
    body.max_tokens = body.max_completion_tokens;
  }
  delete body.max_completion_tokens;

  if (body.max_tokens === undefined) body.max_tokens = DEFAULT_MAX_TOKENS;

  const messages = normalizeRoles(input.messages);

  // Gemini
  if (profile.provider === 'gemini') {
    body.messages = addContinuityInstruction(messages);

    return {
      body: copyAllowed(body, GEMINI_ALLOWED_FIELDS),
      prefixTranslated: false
    };
  }

  // Mistral Large 4
  if (profile.adapter === 'mistral4') {
    body.reasoning_effort = profile.effort || 'high';

    const translated = translateMistralAssistantPrefix(
      addContinuityInstruction(trimMistralMessages(messages))
    );
    body.messages = translated.messages;

    // NVIDIA recommends not overriding both sampling parameters together
    // for this model.
    if (body.temperature !== undefined && body.top_p !== undefined) {
      delete body.top_p;
    }

    return {
      body: copyAllowed(body, MISTRAL_ALLOWED_FIELDS),
      prefixTranslated: translated.translated
    };
  }

  // Other NVIDIA adapters
  body.messages = addContinuityInstruction(messages);

  // Generic NVIDIA models keep all of JanitorAI's other parameters.
  if (profile.adapter === 'generic') {
    return { body, prefixTranslated: false };
  }

  const template =
    body.chat_template_kwargs &&
    typeof body.chat_template_kwargs === 'object' &&
    !Array.isArray(body.chat_template_kwargs)
      ? { ...body.chat_template_kwargs }
      : {};

  switch (profile.adapter) {
    case 'glm':
      body.reasoning_effort = profile.effort || 'high';
      template.clear_thinking = true;
      break;

    case 'minimax':
      template.reasoning_effort = profile.effort || 'medium';
      delete body.reasoning_effort;
      break;

    case 'nemotron-vl':
      body.reasoning_effort = profile.effort || 'high';
      delete template.enable_thinking;
      break;

    default:
      break;
  }

  if (Object.keys(template).length > 0) {
    body.chat_template_kwargs = template;
  } else {
    delete body.chat_template_kwargs;
  }

  return {
    body: copyAllowed(body, ADAPTER_ALLOWED_FIELDS),
    prefixTranslated: false
  };
}

// =============================================================================
// 13. REASONING OUTPUT FILTER
// =============================================================================

const REASONING_KEYS = new Set([
  'reasoning',
  'reasoning_content',
  'internal_reasoning',
  'thinking'
]);

function stripReasoning(value) {
  if (Array.isArray(value)) return value.map(stripReasoning);
  if (!value || typeof value !== 'object') return value;

  const output = {};
  for (const [key, child] of Object.entries(value)) {
    if (REASONING_KEYS.has(key.toLowerCase())) continue;
    output[key] = stripReasoning(child);
  }
  return output;
}

// Some models put their reasoning inline as <think>...</think> inside the
// normal content. This removes those blocks, including when a tag is split
// across streaming chunks.
const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';

function partialTagLength(text, tag) {
  const max = Math.min(tag.length - 1, text.length);
  for (let length = max; length > 0; length--) {
    if (text.endsWith(tag.slice(0, length))) return length;
  }
  return 0;
}

class ThinkTagFilter {
  constructor() {
    this.inside = false;
    this.pending = '';
    this.trimLeading = false;
  }

  // Drops the blank lines that usually follow a closed think block.
  emit(text) {
    if (!this.trimLeading) return text;

    const trimmed = text.replace(/^\s+/, '');
    if (trimmed) this.trimLeading = false;
    return trimmed;
  }

  push(text) {
    let input = this.pending + text;
    let output = '';

    this.pending = '';

    while (input) {
      const tag = this.inside ? THINK_CLOSE : THINK_OPEN;
      const index = input.indexOf(tag);

      if (index >= 0) {
        if (this.inside) this.trimLeading = true;
        else output += this.emit(input.slice(0, index));

        input = input.slice(index + tag.length);
        this.inside = !this.inside;
        continue;
      }

      // Hold back a possible partial tag at the end of this chunk.
      const keep = partialTagLength(input, tag);

      if (!this.inside) {
        output += this.emit(input.slice(0, input.length - keep));
      }

      this.pending = input.slice(input.length - keep);
      break;
    }

    return output;
  }

  flush() {
    const rest = this.inside ? '' : this.pending;
    this.pending = '';
    return rest;
  }
}

function cleanCompletion(data) {
  if (!STRIP_REASONING) return data;

  const result = stripReasoning(data);

  if (result && Array.isArray(result.choices)) {
    for (const choice of result.choices) {
      const message = choice?.message;

      if (message && typeof message.content === 'string') {
        const filter = new ThinkTagFilter();
        message.content = filter.push(message.content) + filter.flush();
      }
    }
  }

  return result;
}

function cleanStreamChunk(data, filters) {
  const result = stripReasoning(data);

  if (!result || !Array.isArray(result.choices)) return result;

  for (const choice of result.choices) {
    if (!choice || typeof choice !== 'object') continue;

    const index = Number.isInteger(choice.index) ? choice.index : 0;

    let filter = filters.get(index);
    if (!filter) {
      filter = new ThinkTagFilter();
      filters.set(index, filter);
    }

    const delta = choice.delta;

    if (delta && typeof delta.content === 'string') {
      delta.content = filter.push(delta.content);
    }

    if (choice.finish_reason) {
      const rest = filter.flush();

      if (rest) {
        const target = delta && typeof delta === 'object'
          ? delta
          : (choice.delta = {});

        target.content =
          (typeof target.content === 'string' ? target.content : '') + rest;
      }
    }
  }

  return result;
}

// =============================================================================
// 14. STREAMING HELPERS
// =============================================================================

function isDoneEvent(eventText) {
  return /(?:^|\n)data:\s*\[DONE\]\s*(?:\n|$)/.test(eventText);
}

function transformSseEvent(eventText, filters) {
  if (!eventText) return '';

  const dataLines = [];
  const otherLines = [];

  for (const line of eventText.split(/\r?\n/)) {
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''));
    } else {
      otherLines.push(line);
    }
  }

  if (!dataLines.length) return `${eventText}\n\n`;

  const data = dataLines.join('\n');
  const prefix = otherLines.length ? `${otherLines.join('\n')}\n` : '';

  if (data.trim() === '[DONE]') return `${prefix}data: [DONE]\n\n`;

  const parsed = parseJson(data);
  if (parsed === null) return `${prefix}data: ${data}\n\n`;

  return `${prefix}data: ${JSON.stringify(cleanStreamChunk(parsed, filters))}\n\n`;
}

function waitForDrain(res, signal) {
  return new Promise(resolve => {
    let settled = false;

    const finish = value => {
      if (settled) return;
      settled = true;

      res.removeListener('drain', onDrain);
      res.removeListener('close', onClose);
      signal.removeEventListener('abort', onAbort);

      resolve(value);
    };

    const onDrain = () => finish(true);
    const onClose = () => finish(false);
    const onAbort = () => finish(false);

    res.once('drain', onDrain);
    res.once('close', onClose);
    signal.addEventListener('abort', onAbort, { once: true });

    if (res.destroyed || signal.aborted) finish(false);
  });
}

async function writeChunk(res, chunk, signal) {
  if (res.destroyed || res.writableEnded || signal.aborted) return false;
  if (res.write(chunk)) return true;
  return waitForDrain(res, signal);
}

function startSseResponse(res) {
  if (res.headersSent) return;

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no'
  });
  res.flushHeaders?.();
}

function endResponse(res) {
  if (!res.destroyed && !res.writableEnded) res.end();
}

// Used when a failure happens after the SSE response has already started,
// so the client gets a readable error instead of a silently dropped stream.
function endSseWithError(res, message, type) {
  if (res.destroyed || res.writableEnded) return;

  try {
    const payload = JSON.stringify({
      error: { message: String(message).slice(0, 4000), type }
    });

    // The leading blank line terminates any partially written event.
    res.write(`\n\ndata: ${payload}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  } catch {
    res.destroy();
  }
}

// =============================================================================
// 15. SSE STREAM RESPONSE
// =============================================================================

async function streamResponse(upstream, res, signal) {
  startSseResponse(res);

  if (!upstream.body) {
    endResponse(res);
    return;
  }

  // Preserve exact upstream bytes when reasoning filtering is disabled.
  if (!STRIP_REASONING) {
    for await (const chunk of upstream.body) {
      if (!(await writeChunk(res, Buffer.from(chunk), signal))) return;
    }

    endResponse(res);
    return;
  }

  const decoder = new TextDecoder();
  const filters = new Map();

  let buffer = '';
  let receivedDone = false;

  for await (const chunk of upstream.body) {
    if (signal.aborted || res.destroyed) return;

    buffer += decoder.decode(chunk, { stream: true });

    while (!receivedDone) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match) break;

      const eventText = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);

      const output = transformSseEvent(eventText, filters);
      if (!output) continue;

      if (isDoneEvent(output)) receivedDone = true;

      if (!(await writeChunk(res, output, signal))) return;
    }

    if (receivedDone) break;

    if (Buffer.byteLength(buffer, 'utf8') > MAX_SSE_EVENT_BYTES) {
      throw new Error('Provider SSE event exceeded the configured size limit.');
    }
  }

  if (signal.aborted || res.destroyed) return;

  buffer += decoder.decode();

  if (!receivedDone && buffer.trim()) {
    const output = transformSseEvent(
      buffer.replace(/(?:\r?\n)+$/, ''),
      filters
    );

    if (output) {
      if (!(await writeChunk(res, output, signal))) return;
      receivedDone = isDoneEvent(output);
    }
  }

  if (!receivedDone) {
    if (!(await writeChunk(res, 'data: [DONE]\n\n', signal))) return;
  }

  endResponse(res);
}

// =============================================================================
// 16. JSON-TO-SSE CONVERSION
// =============================================================================

async function sendJsonAsSse(res, data, signal) {
  const result = cleanCompletion(data);

  startSseResponse(res);

  const events = [];

  if (!result || !Array.isArray(result.choices)) {
    events.push(result);
  } else {
    const base = {
      id: result.id || `chatcmpl-${crypto.randomUUID()}`,
      object: 'chat.completion.chunk',
      created: result.created || Math.floor(Date.now() / 1000),
      model: result.model || 'janitorai-proxy'
    };

    for (const choice of result.choices) {
      const message = { ...(choice?.message || {}) };
      delete message.role;

      const index = Number.isInteger(choice?.index) ? choice.index : 0;

      events.push({
        ...base,
        choices: [{
          index,
          delta: { role: 'assistant', ...message },
          finish_reason: null
        }]
      });

      events.push({
        ...base,
        choices: [{
          index,
          delta: {},
          finish_reason: choice?.finish_reason || 'stop'
        }]
      });
    }

    if (result.usage) {
      events.push({ ...base, choices: [], usage: result.usage });
    }
  }

  for (const event of events) {
    if (!(await writeChunk(res, `data: ${JSON.stringify(event)}\n\n`, signal))) {
      return;
    }
  }

  if (!(await writeChunk(res, 'data: [DONE]\n\n', signal))) return;

  endResponse(res);
}

// =============================================================================
// 17. NVIDIA ASYNCHRONOUS REQUEST POLLING
// =============================================================================

function getNvidiaRequestId(headers, data) {
  return (
    headers.get('nvcf-reqid') ||
    headers.get('nvcf-request-id') ||
    headers.get('x-request-id') ||
    headers.get('request-id') ||
    data?.request_id ||
    data?.requestId ||
    null
  );
}

async function pollNvidiaResult(nvidiaRequestId, apiKey, signal) {
  const url = `${NVIDIA_BASE_URL}/status/${encodeURIComponent(nvidiaRequestId)}`;

  for (let attempt = 1; ; attempt++) {
    await abortableDelay(Math.min(1000 + attempt * 250, 3000), signal);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: 'application/json'
      },
      signal,
      redirect: 'error'
    });

    if (response.status === 202) {
      try {
        await response.body?.cancel();
      } catch {}
      continue;
    }

    if (response.status === 200) {
      const read = await readLimited(response, MAX_RESPONSE_BYTES);

      if (read.truncated) {
        throw new Error('NVIDIA status response exceeded the size limit.');
      }

      const parsed = parseJson(read.text);

      if (parsed === null) {
        throw new Error('NVIDIA status endpoint returned invalid JSON.');
      }

      return parsed;
    }

    const read = await readLimited(response, 1024 * 1024);

    throw new Error(
      extractErrorMessage(
        read.text,
        `NVIDIA status endpoint returned HTTP ${response.status}.`
      )
    );
  }
}

// =============================================================================
// 18. MODEL DISCOVERY
// =============================================================================

// Nothing is cached: the list is fetched with the caller's own key, so one
// key's results are never served to another caller.

async function fetchProviderModels(providerName, apiKey) {
  const provider = providerDetails(providerName);

  const response = await fetch(`${provider.baseUrl}/models`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: 'application/json'
    },
    signal: AbortSignal.timeout(MODEL_LIST_TIMEOUT_MS),
    redirect: 'error'
  });

  const read = await readLimited(response, 5 * 1024 * 1024);

  if (!response.ok) {
    throw new Error(
      `${provider.name} model listing returned HTTP ${response.status}.`
    );
  }

  if (read.truncated) {
    throw new Error(`${provider.name} model list exceeded the size limit.`);
  }

  const parsed = parseJson(read.text);

  if (!Array.isArray(parsed?.data)) {
    throw new Error(
      `${provider.name} returned an unexpected model-list response.`
    );
  }

  return parsed.data
    .filter(item => item && typeof item.id === 'string' && item.id.trim())
    .map(item => {
      const id = item.id.trim();

      if (providerName === 'gemini') {
        // Gemini lists IDs as "models/gemini-..."; show "gemini/gemini-...".
        const bare = id.replace(/^gemini\//, '').replace(/^models\//, '');
        return { ...item, id: `gemini/${bare}`, owned_by: 'google' };
      }

      return { ...item, id, owned_by: item.owned_by || 'nvidia' };
    });
}

// A key starting with "nvapi-" lists NVIDIA models; any other key lists
// Gemini models. The key is only ever sent to its own provider.
async function buildModelList(apiKey) {
  const providerName = isNvidiaKey(apiKey) ? 'nvidia' : 'gemini';

  const models = [];
  let fetchError = null;

  try {
    models.push(...await fetchProviderModels(providerName, apiKey));
  } catch (error) {
    fetchError = error;

    logEvent(
      'model_discovery_error',
      {
        provider: providerName,
        message: String(error?.message || 'Model listing failed.').slice(0, 300)
      },
      true
    );
  }

  if (providerName === 'nvidia') {
    for (const id of EXTRA_NVIDIA_MODELS) {
      models.push({ id, object: 'model', owned_by: 'nvidia' });
    }

    const created = Math.floor(Date.now() / 1000);

    for (const profile of MODEL_PROFILES.values()) {
      models.push({
        id: profile.id,
        object: 'model',
        created,
        owned_by: 'nvidia-nim-proxy',
        name: profile.label
      });
    }
  } else {
    for (const rawId of EXTRA_GEMINI_MODELS) {
      const id = rawId.startsWith('gemini/') ? rawId : `gemini/${rawId}`;
      models.push({ id, object: 'model', owned_by: 'google' });
    }
  }

  const unique = new Map();
  for (const model of models) unique.set(model.id, model);

  if (fetchError && unique.size === 0) throw fetchError;

  return { object: 'list', data: [...unique.values()] };
}

// =============================================================================
// 19. CHAT COMPLETIONS
// =============================================================================

const activeControllers = new Set();

let shuttingDown = false;

async function chatCompletions(req, res, requestId) {
  const started = Date.now();
  const controller = new AbortController();
  const signal = controller.signal;

  activeControllers.add(controller);

  let releaseQueue = null;
  let heartbeat = null;
  let timedOut = false;
  let clientDisconnected = false;

  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort(new Error('Provider request timed out.'));
  }, REQUEST_TIMEOUT_MS);
  timeout.unref?.();

  const onResponseClose = () => {
    if (!res.writableEnded) {
      clientDisconnected = true;
      controller.abort(new Error('Client disconnected.'));
    }
  };

  res.once('close', onResponseClose);

  try {
    if (shuttingDown) {
      return sendError(res, 503, 'Server is shutting down.', 'server_unavailable');
    }

    // Read and validate the incoming request.

    const input = await readRequestBody(req, signal);

    const validationError = validateChatRequest(input);

    if (validationError) {
      return sendError(res, 400, validationError, 'invalid_request_error');
    }

    // Resolve the requested model.

    const requestedModel = input.model || DEFAULT_MODEL;

    if (!requestedModel) {
      return sendError(
        res,
        400,
        'No model was specified. Set DEFAULT_MODEL or send model in the request.',
        'invalid_request_error'
      );
    }

    const profile = resolveModel(requestedModel);

    if (!profile) {
      return sendError(
        res,
        400,
        `Unknown or unsupported model "${requestedModel}".`,
        'invalid_request_error'
      );
    }

    const provider = providerDetails(profile.provider);

    // The provider key comes from the client with every request.

    const apiKey = getClientKey(req);

    if (!apiKey) {
      return sendError(res, 401, MISSING_KEY_MESSAGE, 'authentication_error');
    }

    // Never forward an NVIDIA key to Google.
    if (profile.provider === 'gemini' && isNvidiaKey(apiKey)) {
      return sendError(
        res,
        400,
        `"${requestedModel}" is a Gemini model, but the API key sent is an NVIDIA key. Use your Gemini key with this model.`,
        'invalid_request_error'
      );
    }

    // Prepare the provider-specific request.

    let prepared;

    try {
      prepared = prepareProviderRequest(input, profile);
    } catch (error) {
      return sendError(
        res,
        400,
        error.message || 'Invalid request.',
        'invalid_request_error'
      );
    }

    const body = prepared.body;
    const streaming = body.stream === true;

    if (profile.adapter === 'mistral4' && body.max_tokens > 262144) {
      return sendError(
        res,
        400,
        'Mistral Large 4 max_tokens must be between 1 and 262144.',
        'invalid_request_error'
      );
    }

    const logFields = {
      requestId,
      provider: profile.provider,
      model: profile.model
    };

    logEvent('generation_started', {
      ...logFields,
      streaming,
      messages: body.messages.length,
      continuity: RP_CONTINUITY_MODE,
      mistralPrefixTranslated: prepared.prefixTranslated
    });

    // Acquire a provider concurrency slot, then submit once (no retries).

    releaseQueue = await queues[profile.provider].acquire(signal);

    const upstream = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        Accept: streaming ? 'text/event-stream' : 'application/json'
      },
      body: JSON.stringify(body),
      signal,
      redirect: 'error'
    });

    if (signal.aborted || res.destroyed) return;

    // NVIDIA asynchronous request polling.

    let polledData = null;
    let completedByPolling = false;

    if (profile.provider === 'nvidia' && upstream.status === 202) {
      const pendingRead = await readLimited(upstream, 1024 * 1024);

      const nvidiaRequestId = getNvidiaRequestId(
        upstream.headers,
        parseJson(pendingRead.text)
      );

      if (!nvidiaRequestId) {
        return sendError(
          res,
          502,
          'NVIDIA returned 202 Accepted without a request ID.',
          'upstream_error'
        );
      }

      // Polling can take a while. For streaming clients, open the SSE
      // response now and send comment lines so the connection stays alive.
      if (streaming) {
        startSseResponse(res);

        heartbeat = setInterval(() => {
          if (!res.destroyed && !res.writableEnded) {
            res.write(': keep-alive\n\n');
          }
        }, 15000);
        heartbeat.unref?.();
      }

      try {
        polledData = await pollNvidiaResult(nvidiaRequestId, apiKey, signal);
      } finally {
        clearInterval(heartbeat);
        heartbeat = null;
      }

      completedByPolling = true;
    } else if (!upstream.ok) {
      const errorRead = await readLimited(upstream, 1024 * 1024);

      logEvent(
        'generation_failed',
        {
          ...logFields,
          status: upstream.status,
          durationMs: Date.now() - started
        },
        true
      );

      return sendError(
        res,
        upstream.status,
        extractErrorMessage(
          errorRead.text,
          `${provider.name} returned HTTP ${upstream.status}.`
        ),
        'upstream_error'
      );
    }

    if (signal.aborted || res.destroyed) return;

    // Deliver the result.

    const contentType = upstream.headers.get('content-type') || '';

    if (
      streaming &&
      !completedByPolling &&
      contentType.includes('text/event-stream')
    ) {
      await streamResponse(upstream, res, signal);
    } else {
      let data = polledData;

      if (!completedByPolling) {
        const read = await readLimited(upstream, MAX_RESPONSE_BYTES);

        if (read.truncated) {
          return sendError(
            res,
            502,
            'Provider response exceeded the configured size limit.',
            'upstream_error'
          );
        }

        data = parseJson(read.text);
      }

      if (data === null || data === undefined) {
        throw new Error('Provider returned invalid or empty JSON.');
      }

      if (streaming) {
        await sendJsonAsSse(res, data, signal);
      } else {
        sendJson(res, 200, cleanCompletion(data));
      }
    }

    logEvent('generation_finished', {
      ...logFields,
      durationMs: Date.now() - started
    });
  } catch (error) {
    if (clientDisconnected || res.destroyed || res.writableEnded) return;

    let status;
    let type;
    let message;

    if (timedOut) {
      status = 504;
      type = 'upstream_timeout';
      message = 'Provider request timed out.';
    } else if (shuttingDown && signal.aborted) {
      status = 503;
      type = 'server_unavailable';
      message = 'Server is shutting down.';
    } else {
      status = Number.isInteger(error.status) ? error.status : 502;
      type = error.type ||
        (status === 503 ? 'server_overloaded' : 'upstream_error');
      message = error.message || 'Provider request failed.';
    }

    logEvent(
      'proxy_error',
      {
        requestId,
        status,
        code: error.code || null,
        message: String(error.message || 'Request failed.').slice(0, 300)
      },
      true
    );

    if (res.headersSent) endSseWithError(res, message, type);
    else sendError(res, status, message, type);
  } finally {
    clearTimeout(timeout);
    clearInterval(heartbeat);

    res.removeListener('close', onResponseClose);

    releaseQueue?.();

    activeControllers.delete(controller);
  }
}

// =============================================================================
// 20. ROUTER AND ENDPOINTS
// =============================================================================

const server = http.createServer(async (req, res) => {
  const requestId = crypto.randomUUID();
  const started = Date.now();

  res.setHeader('X-Request-ID', requestId);
  res.setHeader('X-Content-Type-Options', 'nosniff');

  let pathname;

  try {
    const url = new URL(req.url, 'http://localhost');
    pathname = url.pathname.replace(/\/+$/, '') || '/';
  } catch {
    return sendError(res, 400, 'Invalid request URL.', 'invalid_request_error');
  }

  res.once('close', () => {
    logEvent('http_request', {
      requestId,
      method: req.method,
      path: pathname,
      status: res.statusCode,
      durationMs: Date.now() - started
    });
  });

  try {
    if (!applyCors(req, res)) return;

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      return res.end();
    }

    if (req.method === 'GET' && pathname === '/') {
      return sendJson(res, 200, {
        name: 'JanitorAI Multi-Provider Proxy',
        status: shuttingDown ? 'shutting_down' : 'online',
        keys: 'Send your NVIDIA or Gemini API key as a Bearer token with each request.',
        default_model: DEFAULT_MODEL || null,
        allow_unknown_nvidia_models: ALLOW_UNKNOWN_NVIDIA_MODELS,
        endpoints: ['/health', '/v1/models', '/v1/chat/completions']
      });
    }

    if (req.method === 'GET' && pathname === '/health') {
      return sendJson(res, shuttingDown ? 503 : 200, {
        status: shuttingDown ? 'shutting_down' : 'ok',
        providers: {
          nvidia: {
            active: queues.nvidia.active,
            queued: queues.nvidia.waiting.length
          },
          gemini: {
            active: queues.gemini.active,
            queued: queues.gemini.waiting.length
          }
        },
        roleplay_continuity: RP_CONTINUITY_MODE,
        mistral_assistant_prefix_mode: MISTRAL_PREFIX_MODE,
        reasoning_filter: STRIP_REASONING,
        uptime_seconds: Math.floor(process.uptime())
      });
    }

    if (req.method === 'GET' && pathname === '/v1/models') {
      const apiKey = getClientKey(req);

      if (!apiKey) {
        return sendError(res, 401, MISSING_KEY_MESSAGE, 'authentication_error');
      }

      try {
        return sendJson(res, 200, await buildModelList(apiKey));
      } catch (error) {
        return sendError(
          res,
          502,
          error.message || 'Model discovery failed.',
          'model_discovery_error'
        );
      }
    }

    if (req.method === 'POST' && pathname === '/v1/chat/completions') {
      return await chatCompletions(req, res, requestId);
    }

    return sendError(res, 404, 'Endpoint not found.', 'not_found');
  } catch (error) {
    logEvent(
      'handler_error',
      {
        requestId,
        message: String(error?.message || error).slice(0, 300)
      },
      true
    );

    if (res.headersSent) res.destroy();
    else sendError(res, 500, 'Internal proxy error.', 'proxy_error');
  }
});

// =============================================================================
// 21. SERVER SETTINGS
// =============================================================================

server.requestTimeout = 0;
server.timeout = 0;
server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;
server.maxHeadersCount = 100;

server.on('error', error => {
  console.error('[SERVER ERROR]', error.message);
  process.exitCode = 1;
});

server.on('clientError', (error, socket) => {
  if (socket.writable) {
    socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n');
  }
});

// =============================================================================
// 22. START SERVER
// =============================================================================

server.listen(PORT, HOST, () => {
  const lines = [
    ['Host', HOST],
    ['Port', PORT],
    ['Provider keys', 'sent by the client with each request (none stored)'],
    ['Default model', DEFAULT_MODEL || '(request must specify model)'],
    ['Unknown NVIDIA IDs', ALLOW_UNKNOWN_NVIDIA_MODELS],
    ['NVIDIA concurrency', NVIDIA_CONCURRENCY],
    ['Gemini concurrency', GEMINI_CONCURRENCY],
    ['Queue limit/provider', REQUEST_QUEUE_LIMIT],
    ['Queue timeout', `${REQUEST_QUEUE_TIMEOUT_MS}ms`],
    ['Request timeout', `${REQUEST_TIMEOUT_MS}ms`],
    ['Default max tokens', DEFAULT_MAX_TOKENS],
    ['Maximum request', `${MAX_BODY_BYTES} bytes`],
    ['Maximum response', `${MAX_RESPONSE_BYTES} bytes`],
    ['Strip reasoning', STRIP_REASONING],
    ['RP continuity', RP_CONTINUITY_MODE],
    ['Mistral prefix mode', MISTRAL_PREFIX_MODE],
    ['Conversation memory', 'stateless; supplied transcript only']
  ];

  console.log('==============================================');
  console.log(' JANITORAI MULTI-PROVIDER ROLEPLAY PROXY v3');
  console.log('==============================================');

  for (const [label, value] of lines) {
    console.log(`${`${label}:`.padEnd(22)}${value}`);
  }

  console.log('==============================================');
});

// =============================================================================
// 23. GRACEFUL SHUTDOWN
// =============================================================================

function shutdown(signalName) {
  if (shuttingDown) return;

  shuttingDown = true;

  console.log(`[SERVER] ${signalName} received; shutting down.`);

  for (const controller of activeControllers) {
    controller.abort(new Error('Server is shutting down.'));
  }

  const forceExit = setTimeout(() => {
    console.error('[SERVER] Forced shutdown after grace period.');
    server.closeAllConnections?.();
    process.exit(1);
  }, 10000);

  server.close(() => {
    clearTimeout(forceExit);
    console.log('[SERVER] Shutdown complete.');
    process.exit(0);
  });

  server.closeIdleConnections?.();
}

process.once('SIGTERM', () => shutdown('SIGTERM'));
process.once('SIGINT', () => shutdown('SIGINT'));
