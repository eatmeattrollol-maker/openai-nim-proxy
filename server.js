'use strict';

/*
 * JANITORAI MULTI-PROVIDER ROLEPLAY PROXY v2
 *
 * Node.js 20+ | Zero external dependencies
 * Optimized for personal long-term uncensored narrative RP on free Render.
 *
 * PROVIDERS
 *   - NVIDIA NIM (primary)
 *   - Google Gemini (OpenAI-compatible)
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
 *   everything else        -> NVIDIA
 *
 * FEATURES
 *   - Model-specific NVIDIA adapters (Mistral Large 4, GLM, MiniMax, Nemotron, Muse Glimmer)
 *   - Strong anti-hallucination + lorebook supremacy continuity system
 *   - World-engine & NPC simulator instructions
 *   - Novel/cinematic prose guidance
 *   - Mistral assistant-prefix continuation
 *   - Streaming and non-streaming
 *   - NVIDIA async request polling
 *   - Reasoning-output filtering
 *   - Per-provider concurrency queues (free-tier friendly defaults)
 *   - Request cancellation and timeouts
 *   - Model discovery and caching
 *   - Optional proxy authentication (strongly recommended)
 *   - CORS restricted to JanitorAI by default
 *
 * MEMORY
 *   Stateless. Continuity depends entirely on the transcript + lorebook
 *   JanitorAI supplies with each request.
 */

const http = require('node:http');
const crypto = require('node:crypto');

const env = process.env;

// =============================================================================
// 1. CONFIGURATION
// =============================================================================

function intEnv(
  name,
  fallback,
  min = 1,
  max = Number.MAX_SAFE_INTEGER
) {
  const value = env[name] === undefined
    ? fallback
    : Number(env[name]);

  if (
    !Number.isSafeInteger(value) ||
    value < min ||
    value > max
  ) {
    throw new Error(
      `${name} must be an integer between ${min} and ${max}.`
    );
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

  return env[name]
    .split(',')
    .map(value => value.trim())
    .filter(Boolean);
}

function baseUrlEnv(name, fallback) {
  const value = (env[name] || fallback).replace(/\/+$/, '');

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

const NVIDIA_API_KEY = env.NVIDIA_API_KEY || '';
const GEMINI_API_KEY = env.GEMINI_API_KEY || '';

const NVIDIA_BASE_URL = baseUrlEnv(
  'NVIDIA_BASE_URL',
  'https://integrate.api.nvidia.com/v1'
);

const GEMINI_BASE_URL = baseUrlEnv(
  'GEMINI_BASE_URL',
  'https://generativelanguage.googleapis.com/v1beta/openai'
);

const DEFAULT_MODEL = (env.DEFAULT_MODEL || '').trim();

const DEFAULT_MAX_TOKENS = intEnv(
  'DEFAULT_MAX_TOKENS',
  16384
);

const MAX_BODY_BYTES = intEnv(
  'MAX_BODY_BYTES',
  20 * 1024 * 1024
);

const MAX_RESPONSE_BYTES = intEnv(
  'MAX_RESPONSE_BYTES',
  32 * 1024 * 1024
);

const MAX_SSE_EVENT_BYTES = intEnv(
  'MAX_SSE_EVENT_BYTES',
  2 * 1024 * 1024
);

const REQUEST_TIMEOUT_MS = intEnv(
  'REQUEST_TIMEOUT_MS',
  600000,
  1000
);

const MODEL_LIST_TIMEOUT_MS = intEnv(
  'MODEL_LIST_TIMEOUT_MS',
  15000,
  1000
);

const MODEL_CACHE_MS = intEnv(
  'MODEL_CACHE_SECONDS',
  60,
  1
) * 1000;

// Free Render defaults: lower concurrency to stay comfortable under 512 MB.
const NVIDIA_CONCURRENCY = intEnv(
  'NVIDIA_MAX_CONCURRENT_REQUESTS',
  1
);

const GEMINI_CONCURRENCY = intEnv(
  'GEMINI_MAX_CONCURRENT_REQUESTS',
  1
);

const REQUEST_QUEUE_LIMIT = intEnv(
  'REQUEST_QUEUE_LIMIT',
  4,
  0
);

const REQUEST_QUEUE_TIMEOUT_MS = intEnv(
  'REQUEST_QUEUE_TIMEOUT_MS',
  30000,
  1000
);

const STRIP_REASONING = boolEnv(
  'STRIP_REASONING',
  true
);

const RP_CONTINUITY_MODE = boolEnv(
  'RP_CONTINUITY_MODE',
  true
);

const ALLOW_UNKNOWN_NVIDIA_MODELS = boolEnv(
  'ALLOW_UNKNOWN_MODELS',
  true
);

const PROXY_API_KEY = env.PROXY_API_KEY || '';

const ALLOWED_ORIGINS = csvEnv(
  'CORS_ORIGINS',
  [
    'https://janitorai.com',
    'https://www.janitorai.com'
  ]
).map(origin => origin.replace(/\/+$/, ''));

const EXTRA_NVIDIA_MODELS = csvEnv(
  'EXTRA_NVIDIA_MODELS'
);

const EXTRA_GEMINI_MODELS = csvEnv(
  'EXTRA_GEMINI_MODELS'
);

const MISTRAL_PREFIX_MODE = (
  env.MISTRAL_ASSISTANT_PREFIX_MODE || 'translate'
).trim().toLowerCase();

if (!['translate', 'preserve', 'off'].includes(MISTRAL_PREFIX_MODE)) {
  throw new Error(
    'MISTRAL_ASSISTANT_PREFIX_MODE must be translate, preserve, or off.'
  );
}

if (!NVIDIA_API_KEY && !GEMINI_API_KEY) {
  throw new Error(
    'Configure NVIDIA_API_KEY, GEMINI_API_KEY, or both.'
  );
}

if (
  PROXY_API_KEY &&
  Buffer.byteLength(PROXY_API_KEY, 'utf8') < 32
) {
  throw new Error(
    'PROXY_API_KEY must contain at least 32 bytes.'
  );
}

// =============================================================================
// 2. ROLEPLAY CONTINUITY SYSTEM
// =============================================================================

const RP_PROMPT_MARKER = 'ROLEPLAY CONTINUITY INSTRUCTIONS';

// Prefer external file so you can iterate the prompt without redeploying code.
// Fallback is a short, high-priority absolute-rules block.
const fs = require('node:fs');
const path = require('node:path');

function loadContinuityPrompt() {
  const candidates = [
    env.RP_CONTINUITY_FILE,
    path.join(__dirname, 'rp-continuity.txt'),
    path.join(process.cwd(), 'rp-continuity.txt')
  ].filter(Boolean);

  for (const file of candidates) {
    try {
      const content = fs.readFileSync(file, 'utf8').trim();
      if (content) {
        console.log(JSON.stringify({
          type: 'continuity_prompt_loaded',
          source: file,
          bytes: Buffer.byteLength(content, 'utf8')
        }));
        return content.includes(RP_PROMPT_MARKER)
          ? content
          : `${RP_PROMPT_MARKER}\n\n${content}`;
      }
    } catch {}
  }

  // Minimal embedded fallback (absolute rules only).
  return `
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

Treat all supplied messages as one continuous timeline. Continue from the actual current endpoint. Characters remember meaningful experiences. Actions have consequences. Relationships evolve. The response should simply continue the story.

ROLEPLAY CONTINUITY INSTRUCTIONS END.
`.trim();
}

const DEFAULT_RP_CONTINUITY_PROMPT = loadContinuityPrompt();

const RP_CONTINUITY_PROMPT = (() => {
  const custom = typeof env.RP_CONTINUITY_PROMPT === 'string'
    ? env.RP_CONTINUITY_PROMPT.trim()
    : '';

  if (!custom) return DEFAULT_RP_CONTINUITY_PROMPT;

  if (custom.includes(RP_PROMPT_MARKER)) return custom;

  return `${RP_PROMPT_MARKER}\n\n${custom}`;
})();

const MISTRAL_CONTINUATION_PROMPT = `
Continue the immediately preceding assistant message as an unfinished response prefix.

The existing assistant text is already written content. Continue directly from its endpoint.

Do not repeat, quote, summarize, paraphrase, rewrite, or restart any of the existing assistant text.

Preserve the existing writing style, narrative perspective, formatting, tone, dialogue flow, and sentence structure.

Use the complete supplied conversation history to preserve established characters, chronology, locations, physical positioning, relationships, character knowledge, ongoing actions, and consequences.

Do not introduce a new opening, unnecessary recap, unrelated scene, or artificial transition.

Do not treat the unfinished assistant prefix as a new user request.

The continuation must read as one seamless assistant response with no indication that it was generated separately.
`;

// =============================================================================
// 3. MODEL PROFILES
// =============================================================================

const MODEL_PROFILES = new Map();

function addProfile(
  id,
  label,
  upstream = id,
  adapter = 'generic',
  effort = null
) {
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

addProfile(
  MISTRAL4,
  'Mistral Large 4 - High Reasoning',
  MISTRAL4,
  'mistral4',
  'high'
);

addProfile(
  `${MISTRAL4}-medium`,
  'Mistral Large 4 - Medium Reasoning',
  MISTRAL4,
  'mistral4',
  'medium'
);

addProfile(
  `${MISTRAL4}-low`,
  'Mistral Large 4 - Low Reasoning',
  MISTRAL4,
  'mistral4',
  'low'
);

addProfile(
  `${MISTRAL4}-no-thinking`,
  'Mistral Large 4 - No Reasoning',
  MISTRAL4,
  'mistral4',
  'none'
);

// GLM

const GLM = 'z-ai/glm-5.3-flash';

addProfile(
  GLM,
  'GLM 5.3 Flash - High Reasoning',
  GLM,
  'glm',
  'high'
);

addProfile(
  `${GLM}-low`,
  'GLM 5.3 Flash - Low Reasoning',
  GLM,
  'glm',
  'low'
);

addProfile(
  `${GLM}-medium`,
  'GLM 5.3 Flash - Medium Reasoning',
  GLM,
  'glm',
  'high'
);

addProfile(
  `${GLM}-high`,
  'GLM 5.3 Flash - High Reasoning',
  GLM,
  'glm',
  'max'
);

// MiniMax

const MINIMAX = 'minimaxai/minimax-m3.1-flash';

addProfile(
  MINIMAX,
  'MiniMax M3.1 Flash - Medium Reasoning',
  MINIMAX,
  'minimax',
  'medium'
);

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

addProfile(
  NEMOTRON_VL,
  'Nemotron 3.5 Super VL - High Reasoning',
  NEMOTRON_VL,
  'nemotron-vl',
  'high'
);

addProfile(
  `${NEMOTRON_VL}-no-thinking`,
  'Nemotron 3.5 Super VL - No Reasoning',
  NEMOTRON_VL,
  'nemotron-vl',
  'none'
);

addProfile(
  `${NEMOTRON_VL}-low`,
  'Nemotron 3.5 Super VL - Low Reasoning',
  NEMOTRON_VL,
  'nemotron-vl',
  'low'
);

addProfile(
  `${NEMOTRON_VL}-high`,
  'Nemotron 3.5 Super VL - High Reasoning',
  NEMOTRON_VL,
  'nemotron-vl',
  'high'
);

// Muse Glimmer (Meta 30B agentic, available on NVIDIA NIM)

const MUSE_GLIMMER = 'meta/muse-glimmer-30b';

addProfile(
  MUSE_GLIMMER,
  'Muse Glimmer 30B',
  MUSE_GLIMMER,
  'generic'
);

addProfile(
  `${MUSE_GLIMMER}-high`,
  'Muse Glimmer 30B - High Reasoning',
  MUSE_GLIMMER,
  'generic'
);

// Other NVIDIA models

addProfile('stepfun-ai/step-5-preview', 'Step 5 Preview');
addProfile('poolside/laguna-xs-2.1', 'Laguna XS 2.1');
addProfile('nvidia/nemotron-3.5-bolt-e4b', 'Nemotron 3.5 Bolt E4B');

// =============================================================================
// 4. MODEL RESOLUTION
// =============================================================================

function resolveModel(requested) {
  if (typeof requested !== 'string' || !requested.trim()) {
    return null;
  }

  const id = requested.trim();

  const known = MODEL_PROFILES.get(id);

  if (known) return known;

  // Explicit Gemini namespace.

  if (id.startsWith('gemini/')) {
    const model = id.slice('gemini/'.length);

    if (!model) return null;

    return {
      id,
      model,
      provider: 'gemini',
      adapter: 'generic',
      label: id
    };
  }

  // Google compatibility namespace.

  if (id.startsWith('google/')) {
    const model = id.slice('google/'.length);

    if (
      model.startsWith('gemini-') ||
      model.startsWith('models/gemini-')
    ) {
      return {
        id,
        model,
        provider: 'gemini',
        adapter: 'generic',
        label: id
      };
    }
  }

  if (ALLOW_UNKNOWN_NVIDIA_MODELS) {
    return {
      id,
      model: id,
      provider: 'nvidia',
      adapter: 'generic',
      label: id
    };
  }

  return null;
}

function providerDetails(provider) {
  if (provider === 'gemini') {
    return {
      name: 'Gemini',
      baseUrl: GEMINI_BASE_URL,
      apiKey: GEMINI_API_KEY
    };
  }

  return {
    name: 'NVIDIA',
    baseUrl: NVIDIA_BASE_URL,
    apiKey: NVIDIA_API_KEY
  };
}

// =============================================================================
// 5. CONCURRENCY QUEUE
// =============================================================================

class Queue {
  constructor(limit, maxQueue, timeoutMs) {
    this.limit = limit;
    this.maxQueue = maxQueue;
    this.timeoutMs = timeoutMs;
    this.active = 0;
    this.waiting = [];
  }

  acquire(signal) {
    if (signal.aborted) {
      return Promise.reject(
        signal.reason || new Error('Request aborted.')
      );
    }

    if (this.active < this.limit) {
      this.active++;
      return Promise.resolve(this.makeRelease());
    }

    if (this.waiting.length >= this.maxQueue) {
      const error = new Error('Provider queue is full.');

      error.status = 503;
      error.type = 'server_overloaded';

      return Promise.reject(error);
    }

    return new Promise((resolve, reject) => {
      const entry = {
        settled: false,
        timer: null,
        onAbort: null,
        grant: null
      };

      const remove = () => {
        const index = this.waiting.indexOf(entry);

        if (index >= 0) {
          this.waiting.splice(index, 1);
        }
      };

      const cleanup = () => {
        clearTimeout(entry.timer);

        if (entry.onAbort) {
          signal.removeEventListener('abort', entry.onAbort);
        }
      };

      const rejectEntry = error => {
        if (entry.settled) return;

        entry.settled = true;

        remove();
        cleanup();

        reject(error);
      };

      entry.onAbort = () => {
        rejectEntry(
          signal.reason || new Error('Request aborted.')
        );
      };

      entry.grant = () => {
        if (entry.settled) return;

        entry.settled = true;
        cleanup();

        resolve(this.makeRelease());
      };

      entry.timer = setTimeout(() => {
        const error = new Error(
          'Provider queue wait timed out.'
        );

        error.status = 503;
        error.type = 'queue_timeout';

        rejectEntry(error);
      }, this.timeoutMs);

      entry.timer.unref?.();

      signal.addEventListener(
        'abort',
        entry.onAbort,
        { once: true }
      );

      this.waiting.push(entry);

      if (signal.aborted) entry.onAbort();
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
    while (
      this.waiting.length &&
      this.active < this.limit
    ) {
      const entry = this.waiting.shift();

      if (!entry || entry.settled) continue;

      this.active++;

      entry.grant();
    }
  }
}

const queues = {
  nvidia: new Queue(
    NVIDIA_CONCURRENCY,
    REQUEST_QUEUE_LIMIT,
    REQUEST_QUEUE_TIMEOUT_MS
  ),

  gemini: new Queue(
    GEMINI_CONCURRENCY,
    REQUEST_QUEUE_LIMIT,
    REQUEST_QUEUE_TIMEOUT_MS
  )
};

// =============================================================================
// 6. BASIC HTTP AND JSON HELPERS
// =============================================================================

function sendJson(res, status, payload) {
  if (res.destroyed || res.writableEnded) return;

  const body = JSON.stringify(payload);

  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store'
  });

  res.end(body);
}

function sendError(
  res,
  status,
  message,
  type = 'proxy_error'
) {
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

function extractErrorMessage(
  data,
  fallback = 'Provider request failed.'
) {
  const parsed = typeof data === 'string'
    ? parseJson(data) || data
    : data;

  if (typeof parsed === 'string') {
    return parsed.slice(0, 3000) || fallback;
  }

  if (parsed && typeof parsed === 'object') {
    const message =
      parsed.error?.message ||
      parsed.message ||
      parsed.detail ||
      parsed.title;

    if (typeof message === 'string') {
      return message.slice(0, 3000);
    }
  }

  return fallback;
}

// =============================================================================
// 7. LIMITED RESPONSE READING
// =============================================================================

async function readLimited(
  response,
  limit = MAX_RESPONSE_BYTES
) {
  if (!response.body) {
    return {
      text: '',
      truncated: false
    };
  }

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

      if (remaining <= 0) {
        truncated = true;
        break;
      }

      if (chunk.length > remaining) {
        chunks.push(chunk.subarray(0, remaining));

        total += remaining;
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

  return {
    text: Buffer.concat(chunks).toString('utf8'),
    truncated
  };
}

// =============================================================================
// 8. REQUEST BODY READING
// =============================================================================

async function nextWithAbort(iterator, signal) {
  if (signal.aborted) {
    throw signal.reason || new Error('Request aborted.');
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      signal.removeEventListener('abort', onAbort);
    };

    const finish = (fn, value) => {
      if (settled) return;

      settled = true;
      cleanup();

      fn(value);
    };

    const onAbort = () => {
      finish(
        reject,
        signal.reason || new Error('Request aborted.')
      );
    };

    signal.addEventListener(
      'abort',
      onAbort,
      { once: true }
    );

    if (signal.aborted) {
      onAbort();
      return;
    }

    Promise.resolve()
      .then(() => iterator.next())
      .then(
        value => finish(resolve, value),
        error => finish(reject, error)
      );
  });
}

async function readRequestBody(req, signal) {
  const declaredLength = Number(
    req.headers['content-length'] || 0
  );

  if (
    Number.isFinite(declaredLength) &&
    declaredLength > MAX_BODY_BYTES
  ) {
    const error = new Error('Request body is too large.');

    error.status = 413;

    throw error;
  }

  const chunks = [];

  let total = 0;
  let tooLarge = false;

  const iterator = req[Symbol.asyncIterator]();

  while (true) {
    const result = await nextWithAbort(iterator, signal);

    if (result.done) break;

    const chunk = Buffer.from(result.value);

    total += chunk.length;

    if (total > MAX_BODY_BYTES) {
      tooLarge = true;
      chunks.length = 0;
    } else if (!tooLarge) {
      chunks.push(chunk);
    }
  }

  if (tooLarge) {
    const error = new Error('Request body is too large.');

    error.status = 413;

    throw error;
  }

  const text = Buffer.concat(chunks).toString('utf8');

  const parsed = parseJson(text);

  if (
    !parsed ||
    typeof parsed !== 'object' ||
    Array.isArray(parsed)
  ) {
    const error = new Error(
      'Request body must be a valid JSON object.'
    );

    error.status = 400;

    throw error;
  }

  return parsed;
}

// =============================================================================
// 9. AUTHENTICATION AND CORS
// =============================================================================

function authenticate(req, res) {
  if (!PROXY_API_KEY) return true;

  const match = String(
    req.headers.authorization || ''
  ).match(/^Bearer\s+(.+)$/i);

  const supplied = Buffer.from(
    match?.[1] || '',
    'utf8'
  );

  const expected = Buffer.from(
    PROXY_API_KEY,
    'utf8'
  );

  const valid =
    supplied.length === expected.length &&
    crypto.timingSafeEqual(supplied, expected);

  if (!valid) {
    sendError(
      res,
      401,
      'Invalid or missing proxy API key.',
      'authentication_error'
    );

    return false;
  }

  return true;
}

function applyCors(req, res) {
  const rawOrigin = req.headers.origin;

  const origin = typeof rawOrigin === 'string'
    ? rawOrigin.replace(/\/+$/, '')
    : '';

  const wildcard = ALLOWED_ORIGINS.includes('*');

  const allowed =
    !origin ||
    wildcard ||
    ALLOWED_ORIGINS.includes(origin);

  if (!allowed) {
    sendError(
      res,
      403,
      'Origin not allowed.',
      'cors_error'
    );

    return false;
  }

  if (origin) {
    res.setHeader(
      'Access-Control-Allow-Origin',
      wildcard ? '*' : rawOrigin
    );

    res.setHeader('Vary', 'Origin');
  }

  res.setHeader(
    'Access-Control-Allow-Methods',
    'GET, POST, OPTIONS'
  );

  res.setHeader(
    'Access-Control-Allow-Headers',
    req.headers['access-control-request-headers'] ||
      'Authorization, Content-Type'
  );

  res.setHeader(
    'Access-Control-Expose-Headers',
    'X-Request-ID'
  );

  res.setHeader(
    'Access-Control-Max-Age',
    '86400'
  );

  return true;
}

// =============================================================================
// 10. REQUEST VALIDATION
// =============================================================================

function validateChatRequest(body) {
  if (
    !Array.isArray(body.messages) ||
    body.messages.length === 0
  ) {
    return 'messages must be a non-empty array.';
  }

  if (body.messages.length > 10000) {
    return 'messages contains too many entries.';
  }

  if (
    body.model !== undefined &&
    (
      typeof body.model !== 'string' ||
      !body.model.trim()
    )
  ) {
    return 'model must be a non-empty string.';
  }

  const validRoles = new Set([
    'system',
    'developer',
    'context',
    'user',
    'assistant',
    'tool'
  ]);

  for (let i = 0; i < body.messages.length; i++) {
    const message = body.messages[i];

    if (
      !message ||
      typeof message !== 'object' ||
      Array.isArray(message)
    ) {
      return `Message ${i + 1} must be an object.`;
    }

    if (
      typeof message.role !== 'string' ||
      !validRoles.has(message.role)
    ) {
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
      (
        typeof message.tool_call_id !== 'string' ||
        !message.tool_call_id
      )
    ) {
      return `Tool message ${i + 1} is missing tool_call_id.`;
    }
  }

  for (const key of [
    'temperature',
    'top_p',
    'frequency_penalty',
    'presence_penalty'
  ]) {
    if (
      body[key] !== undefined &&
      (
        typeof body[key] !== 'number' ||
        !Number.isFinite(body[key])
      )
    ) {
      return `${key} must be a finite number.`;
    }
  }

  if (
    body.temperature !== undefined &&
    (body.temperature < 0 || body.temperature > 2)
  ) {
    return 'temperature must be between 0 and 2.';
  }

  if (
    body.top_p !== undefined &&
    (body.top_p < 0 || body.top_p > 1)
  ) {
    return 'top_p must be between 0 and 1.';
  }

  for (const key of [
    'frequency_penalty',
    'presence_penalty'
  ]) {
    if (
      body[key] !== undefined &&
      (body[key] < -2 || body[key] > 2)
    ) {
      return `${key} must be between -2 and 2.`;
    }
  }

  for (const key of [
    'max_tokens',
    'max_completion_tokens'
  ]) {
    if (
      body[key] !== undefined &&
      (
        !Number.isInteger(body[key]) ||
        body[key] < 1
      )
    ) {
      return `${key} must be a positive integer.`;
    }
  }

  if (
    body.stream !== undefined &&
    typeof body.stream !== 'boolean'
  ) {
    return 'stream must be a boolean.';
  }

  if (
    body.n !== undefined &&
    (
      !Number.isInteger(body.n) ||
      body.n < 1 ||
      body.n > 16
    )
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
    (
      body.stop.length > 16 ||
      body.stop.some(value => typeof value !== 'string')
    )
  ) {
    return 'stop must contain no more than 16 strings.';
  }

  return null;
}

// =============================================================================
// 11. ROLEPLAY CONTINUITY INJECTION
// =============================================================================

function contentContainsMarker(content) {
  if (typeof content === 'string') {
    return content.includes(RP_PROMPT_MARKER);
  }

  if (Array.isArray(content)) {
    return content.some(part => {
      if (typeof part === 'string') {
        return part.includes(RP_PROMPT_MARKER);
      }

      if (!part || typeof part !== 'object') {
        return false;
      }

      return (
        typeof part.text === 'string' &&
        part.text.includes(RP_PROMPT_MARKER)
      );
    });
  }

  return false;
}

function alreadyHasContinuityPrompt(messages) {
  return messages.some(message => {
    if (
      !message ||
      !['system', 'developer'].includes(message.role)
    ) {
      return false;
    }

    return contentContainsMarker(message.content);
  });
}

function addContinuityInstruction(messages) {
  if (
    !RP_CONTINUITY_MODE ||
    !RP_CONTINUITY_PROMPT.trim()
  ) {
    return messages.map(message => ({ ...message }));
  }

  const result = messages.map(message => ({ ...message }));

  if (alreadyHasContinuityPrompt(result)) {
    return result;
  }

  // Keep the existing system/developer instructions intact.
  // Insert the continuity prompt after the initial instruction
  // block and before the conversation messages.

  let insertionIndex = 0;

  const leadingInstructionRoles = new Set([
    'system',
    'developer',
    'context'
  ]);

  while (
    insertionIndex < result.length &&
    leadingInstructionRoles.has(result[insertionIndex].role)
  ) {
    insertionIndex++;
  }

  result.splice(insertionIndex, 0, {
    role: 'system',
    content: RP_CONTINUITY_PROMPT
  });

  return result;
}

// =============================================================================
// 12. MISTRAL MESSAGE NORMALIZATION
// =============================================================================

function normalizeMistralMessages(messages) {
  const supportedRoles = new Set([
    'system',
    'user',
    'assistant',
    'tool'
  ]);

  return messages.map((message, index) => {
    let role = message.role;

    if (
      role === 'developer' ||
      role === 'context'
    ) {
      role = 'system';
    }

    if (!supportedRoles.has(role)) {
      throw new Error(
        `Mistral message ${index + 1} has unsupported role "${role}".`
      );
    }

    const output = { role };

    for (const key of [
      'content',
      'name',
      'tool_call_id',
      'tool_calls',
      'function_call',
      'refusal',
      'reasoning_content'
    ]) {
      if (
        Object.prototype.hasOwnProperty.call(message, key)
      ) {
        output[key] = message[key];
      }
    }

    if (
      role === 'tool' &&
      !output.tool_call_id
    ) {
      throw new Error(
        `Mistral tool message ${index + 1} is missing tool_call_id.`
      );
    }

    return output;
  });
}

// =============================================================================
// 13. MISTRAL ASSISTANT-PREFIX CONTINUATION
// =============================================================================

function translateMistralAssistantPrefix(messages) {
  const result = messages.slice();

  const last = result[result.length - 1];

  if (
    MISTRAL_PREFIX_MODE !== 'translate' ||
    !last ||
    last.role !== 'assistant'
  ) {
    return {
      messages: result,
      translated: false
    };
  }

  if (
    Array.isArray(last.tool_calls) &&
    last.tool_calls.length > 0
  ) {
    throw new Error(
      'The final Mistral message contains tool_calls without a tool result; it cannot be translated as a narrative prefix.'
    );
  }

  result.push({
    role: 'user',
    content: MISTRAL_CONTINUATION_PROMPT
  });

  return {
    messages: result,
    translated: true
  };
}

// =============================================================================
// 14. PROVIDER REQUEST ADAPTERS
// =============================================================================

const MISTRAL_ALLOWED_FIELDS = new Set([
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
]);

const ADAPTER_ALLOWED_FIELDS = new Set([
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
  'parallel_tool_calls',
  'response_format',
  'stream_options',
  'chat_template_kwargs',
  'reasoning_effort',
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
    if (allowed.has(key)) {
      output[key] = value;
    }
  }

  return output;
}

function prepareProviderRequest(input, profile) {
  const body = {
    ...input,
    model: profile.model
  };

  if (
    body.max_tokens === undefined &&
    body.max_completion_tokens !== undefined
  ) {
    body.max_tokens = body.max_completion_tokens;
  }

  delete body.max_completion_tokens;

  if (body.max_tokens === undefined) {
    body.max_tokens = DEFAULT_MAX_TOKENS;
  }

  // Gemini

  if (profile.provider !== 'nvidia') {
    body.messages = addContinuityInstruction(input.messages);

    return {
      body,
      prefixTranslated: false
    };
  }

  // Mistral Large 4

  if (profile.adapter === 'mistral4') {
    body.reasoning_effort = profile.effort || 'high';

    delete body.chat_template_kwargs;

    let messages = normalizeMistralMessages(body.messages);

    messages = addContinuityInstruction(messages);

    const translated = translateMistralAssistantPrefix(messages);

    body.messages = translated.messages;

    // NVIDIA recommends not overriding both parameters together
    // for this Mistral profile.

    if (
      body.temperature !== undefined &&
      body.top_p !== undefined
    ) {
      delete body.top_p;
    }

    return {
      body: copyAllowed(body, MISTRAL_ALLOWED_FIELDS),
      prefixTranslated: translated.translated
    };
  }

  // Other NVIDIA adapters

  let template =
    body.chat_template_kwargs &&
    typeof body.chat_template_kwargs === 'object' &&
    !Array.isArray(body.chat_template_kwargs)
      ? { ...body.chat_template_kwargs }
      : {};

  body.messages = addContinuityInstruction(body.messages);

  switch (profile.adapter) {
    case 'glm':
      body.reasoning_effort = profile.effort || 'high';

      template.clear_thinking = true;

      body.chat_template_kwargs = template;

      break;

    case 'minimax':
      template.reasoning_effort = profile.effort || 'medium';

      body.chat_template_kwargs = template;

      delete body.reasoning_effort;

      break;

    case 'nemotron-vl':
      body.reasoning_effort = profile.effort || 'high';

      delete template.enable_thinking;

      body.chat_template_kwargs = template;

      break;

    case 'step':
    case 'generic':
    default:
      break;
  }

  if (
    body.chat_template_kwargs &&
    Object.keys(body.chat_template_kwargs).length === 0
  ) {
    delete body.chat_template_kwargs;
  }

  // Generic NVIDIA models retain JanitorAI's other parameters.

  if (profile.adapter === 'generic') {
    return {
      body,
      prefixTranslated: false
    };
  }

  return {
    body: copyAllowed(body, ADAPTER_ALLOWED_FIELDS),
    prefixTranslated: false
  };
}

// =============================================================================
// 15. REASONING OUTPUT FILTER
// =============================================================================

const REASONING_KEYS = new Set([
  'reasoning',
  'reasoning_content',
  'internal_reasoning',
  'thinking'
]);

function stripReasoning(value) {
  if (Array.isArray(value)) {
    return value.map(stripReasoning);
  }

  if (!value || typeof value !== 'object') {
    return value;
  }

  const output = {};

  for (const [key, child] of Object.entries(value)) {
    if (REASONING_KEYS.has(key.toLowerCase())) {
      continue;
    }

    output[key] = stripReasoning(child);
  }

  return output;
}

// =============================================================================
// 16. STREAMING HELPERS
// =============================================================================

function isDoneEvent(eventText) {
  return /(?:^|\n)data:\s*\[DONE\]\s*(?:\n|$)/.test(eventText);
}

function transformSseEvent(eventText) {
  if (!eventText) return '';

  const lines = eventText.split(/\r?\n/);

  const dataLines = [];
  const otherLines = [];

  for (const line of lines) {
    if (line.startsWith('data:')) {
      dataLines.push(
        line.slice(5).replace(/^ /, '')
      );
    } else {
      otherLines.push(line);
    }
  }

  if (!dataLines.length) {
    return `${eventText}\n\n`;
  }

  const data = dataLines.join('\n');

  const prefix = otherLines.length
    ? `${otherLines.join('\n')}\n`
    : '';

  if (data.trim() === '[DONE]') {
    return `${prefix}data: [DONE]\n\n`;
  }

  if (!STRIP_REASONING) {
    return `${prefix}data: ${data}\n\n`;
  }

  try {
    const filtered = stripReasoning(JSON.parse(data));

    return `${prefix}data: ${JSON.stringify(filtered)}\n\n`;
  } catch {
    return `${prefix}data: ${data}\n\n`;
  }
}

function waitForDrain(res, signal) {
  return new Promise(resolve => {
    let settled = false;

    const cleanup = () => {
      res.removeListener('drain', onDrain);
      res.removeListener('close', onClose);

      signal.removeEventListener('abort', onAbort);
    };

    const finish = value => {
      if (settled) return;

      settled = true;

      cleanup();

      resolve(value);
    };

    const onDrain = () => finish(true);
    const onClose = () => finish(false);
    const onAbort = () => finish(false);

    res.once('drain', onDrain);
    res.once('close', onClose);

    signal.addEventListener(
      'abort',
      onAbort,
      { once: true }
    );

    if (res.destroyed || signal.aborted) {
      finish(false);
    }
  });
}

async function writeChunk(res, chunk, signal) {
  if (
    res.destroyed ||
    res.writableEnded ||
    signal.aborted
  ) {
    return false;
  }

  if (res.write(chunk)) {
    return true;
  }

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

// =============================================================================
// 17. SSE STREAM RESPONSE
// =============================================================================

async function streamResponse(upstream, res, signal) {
  startSseResponse(res);

  if (!upstream.body) {
    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }

    return;
  }

  // Preserve exact upstream bytes when reasoning filtering
  // is disabled.

  if (!STRIP_REASONING) {
    for await (const chunk of upstream.body) {
      const ok = await writeChunk(
        res,
        Buffer.from(chunk),
        signal
      );

      if (!ok) return;
    }

    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }

    return;
  }

  const decoder = new TextDecoder();

  let buffer = '';
  let receivedDone = false;

  for await (const chunk of upstream.body) {
    if (signal.aborted || res.destroyed) return;

    buffer += decoder.decode(chunk, {
      stream: true
    });

    if (
      Buffer.byteLength(buffer, 'utf8') >
      MAX_SSE_EVENT_BYTES
    ) {
      throw new Error(
        'Provider SSE event exceeded the configured size limit.'
      );
    }

    while (true) {
      const match = /\r?\n\r?\n/.exec(buffer);

      if (!match) break;

      const eventText = buffer.slice(0, match.index);

      const separator = match[0];

      buffer = buffer.slice(
        match.index + separator.length
      );

      const output = transformSseEvent(eventText);

      if (isDoneEvent(output)) {
        receivedDone = true;
      }

      const ok = await writeChunk(
        res,
        output,
        signal
      );

      if (!ok) return;

      if (receivedDone) break;
    }

    if (receivedDone) break;
  }

  if (signal.aborted || res.destroyed) return;

  buffer += decoder.decode();

  if (!receivedDone && buffer.trim()) {
    const output = transformSseEvent(
      buffer.replace(/(?:\r?\n)+$/, '')
    );

    const ok = await writeChunk(
      res,
      output,
      signal
    );

    if (!ok) return;

    receivedDone = isDoneEvent(output);
  }

  if (!receivedDone) {
    const ok = await writeChunk(
      res,
      'data: [DONE]\n\n',
      signal
    );

    if (!ok) return;
  }

  if (!res.destroyed && !res.writableEnded) {
    res.end();
  }
}

// =============================================================================
// 18. JSON-TO-SSE CONVERSION
// =============================================================================

async function sendJsonAsSse(res, data, signal) {
  const result = STRIP_REASONING
    ? stripReasoning(data)
    : data;

  startSseResponse(res);

  if (!result || !Array.isArray(result.choices)) {
    const ok = await writeChunk(
      res,
      `data: ${JSON.stringify(result)}\n\n`,
      signal
    );

    if (!ok) return;

    if (!(await writeChunk(
      res,
      'data: [DONE]\n\n',
      signal
    ))) {
      return;
    }

    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }

    return;
  }

  const id = result.id ||
    `chatcmpl-${crypto.randomUUID()}`;

  const created = result.created ||
    Math.floor(Date.now() / 1000);

  const model = result.model ||
    'janitorai-proxy';

  for (const choice of result.choices) {
    const message = {
      ...(choice.message || {})
    };

    delete message.role;

    const index = Number.isInteger(choice.index)
      ? choice.index
      : 0;

    const firstChunk = {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{
        index,
        delta: {
          role: 'assistant',
          ...message
        },
        finish_reason: null
      }]
    };

    if (!(await writeChunk(
      res,
      `data: ${JSON.stringify(firstChunk)}\n\n`,
      signal
    ))) {
      return;
    }

    const finalChunk = {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{
        index,
        delta: {},
        finish_reason: choice.finish_reason || 'stop'
      }]
    };

    if (!(await writeChunk(
      res,
      `data: ${JSON.stringify(finalChunk)}\n\n`,
      signal
    ))) {
      return;
    }
  }

  if (result.usage) {
    const usageChunk = {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [],
      usage: result.usage
    };

    if (!(await writeChunk(
      res,
      `data: ${JSON.stringify(usageChunk)}\n\n`,
      signal
    ))) {
      return;
    }
  }

  if (!(await writeChunk(
    res,
    'data: [DONE]\n\n',
    signal
  ))) {
    return;
  }

  if (!res.destroyed && !res.writableEnded) {
    res.end();
  }
}

// =============================================================================
// 19. NVIDIA ASYNCHRONOUS REQUEST POLLING
// =============================================================================

function getRequestId(headers, data) {
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

function abortableDelay(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(
        signal.reason || new Error('Request aborted.')
      );

      return;
    }

    let settled = false;

    const cleanup = () => {
      clearTimeout(timer);

      signal.removeEventListener('abort', onAbort);
    };

    const finish = (fn, value) => {
      if (settled) return;

      settled = true;

      cleanup();

      fn(value);
    };

    const timer = setTimeout(() => {
      finish(resolve);
    }, ms);

    timer.unref?.();

    const onAbort = () => {
      finish(
        reject,
        signal.reason || new Error('Request aborted.')
      );
    };

    signal.addEventListener(
      'abort',
      onAbort,
      { once: true }
    );

    if (signal.aborted) onAbort();
  });
}

async function pollNvidiaResult(requestId, signal) {
  const url = `${NVIDIA_BASE_URL}/status/${encodeURIComponent(requestId)}`;

  let attempt = 0;

  while (true) {
    if (signal.aborted) {
      throw signal.reason || new Error('Request aborted.');
    }

    attempt++;

    const delay = Math.min(
      1000 + attempt * 250,
      3000
    );

    await abortableDelay(delay, signal);

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${NVIDIA_API_KEY}`,
        Accept: 'application/json'
      },
      signal,
      redirect: 'error'
    });

    if (response.status === 200) {
      const read = await readLimited(
        response,
        MAX_RESPONSE_BYTES
      );

      if (read.truncated) {
        throw new Error(
          'NVIDIA status response exceeded the size limit.'
        );
      }

      const parsed = parseJson(read.text);

      if (parsed === null) {
        throw new Error(
          'NVIDIA status endpoint returned invalid JSON.'
        );
      }

      return parsed;
    }

    if (response.status === 202) {
      try {
        await response.body?.cancel();
      } catch {}

      continue;
    }

    const read = await readLimited(
      response,
      1024 * 1024
    );

    throw new Error(
      extractErrorMessage(
        read.text,
        `NVIDIA status endpoint returned HTTP ${response.status}.`
      )
    );
  }
}

// =============================================================================
// 20. MODEL DISCOVERY AND CACHING
// =============================================================================

let modelCache = null;
let modelCacheAt = 0;
let modelFetchPromise = null;

async function fetchProviderModels(providerName) {
  const provider = providerDetails(providerName);

  if (!provider.apiKey) return [];

  const controller = new AbortController();

  const timer = setTimeout(
    () => controller.abort(
      new Error('Model discovery timed out.')
    ),
    MODEL_LIST_TIMEOUT_MS
  );

  timer.unref?.();

  try {
    const response = await fetch(
      `${provider.baseUrl}/models`,
      {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
          Accept: 'application/json'
        },
        signal: controller.signal,
        redirect: 'error'
      }
    );

    const read = await readLimited(
      response,
      5 * 1024 * 1024
    );

    if (!response.ok) {
      throw new Error(
        `${provider.name} model listing returned HTTP ${response.status}.`
      );
    }

    if (read.truncated) {
      throw new Error(
        `${provider.name} model list exceeded the size limit.`
      );
    }

    const parsed = parseJson(read.text);

    if (!Array.isArray(parsed?.data)) {
      throw new Error(
        `${provider.name} returned an unexpected model-list response.`
      );
    }

    return parsed.data
      .filter(item =>
        item &&
        typeof item.id === 'string' &&
        item.id.trim()
      )
      .map(item => {
        const id = item.id.trim();

        if (providerName === 'gemini') {
          return {
            ...item,
            id: id.startsWith('gemini/')
              ? id
              : `gemini/${id}`,
            owned_by: 'google'
          };
        }

        return {
          ...item,
          id,
          owned_by: item.owned_by || 'nvidia'
        };
      });
  } finally {
    clearTimeout(timer);
  }
}

async function buildModelList() {
  const providers = [];

  if (NVIDIA_API_KEY) providers.push('nvidia');
  if (GEMINI_API_KEY) providers.push('gemini');

  const results = await Promise.allSettled(
    providers.map(fetchProviderModels)
  );

  const models = [];

  let successfulProviderCount = 0;

  for (let i = 0; i < results.length; i++) {
    const result = results[i];

    if (result.status === 'fulfilled') {
      successfulProviderCount++;

      models.push(...result.value);
    } else {
      console.error(JSON.stringify({
        type: 'model_discovery_error',
        provider: providers[i],
        message:
          result.reason?.message ||
          'Model listing failed.'
      }));
    }
  }

  for (const id of EXTRA_NVIDIA_MODELS) {
    models.push({
      id,
      object: 'model',
      owned_by: 'nvidia'
    });
  }

  for (const rawId of EXTRA_GEMINI_MODELS) {
    const id = rawId.startsWith('gemini/')
      ? rawId
      : `gemini/${rawId}`;

    models.push({
      id,
      object: 'model',
      owned_by: 'google'
    });
  }

  const created = Math.floor(Date.now() / 1000);

  if (NVIDIA_API_KEY) {
    for (const profile of MODEL_PROFILES.values()) {
      models.push({
        id: profile.id,
        object: 'model',
        created,
        owned_by: 'nvidia-nim-proxy',
        name: profile.label
      });
    }
  }

  const unique = new Map();

  for (const model of models) {
    unique.set(model.id, model);
  }

  if (
    successfulProviderCount === 0 &&
    unique.size === 0
  ) {
    throw new Error(
      'Could not retrieve models from any configured provider.'
    );
  }

  return {
    object: 'list',
    data: [...unique.values()]
  };
}

async function getModelList(forceRefresh = false) {
  if (
    !forceRefresh &&
    modelCache &&
    Date.now() - modelCacheAt < MODEL_CACHE_MS
  ) {
    return modelCache;
  }

  if (modelFetchPromise) {
    return modelFetchPromise;
  }

  modelFetchPromise = buildModelList();

  try {
    const result = await modelFetchPromise;

    modelCache = result;
    modelCacheAt = Date.now();

    return result;
  } catch (error) {
    // Preserve an earlier working model catalog if a refresh fails.

    if (modelCache) return modelCache;

    throw error;
  } finally {
    modelFetchPromise = null;
  }
}

// =============================================================================
// 21. CHAT COMPLETIONS
// =============================================================================

const activeControllers = new Set();

let shuttingDown = false;

async function chatCompletions(req, res, requestId) {
  const started = Date.now();

  const controller = new AbortController();

  activeControllers.add(controller);

  let releaseQueue = null;

  let timedOut = false;
  let clientDisconnected = false;

  const timeout = setTimeout(() => {
    timedOut = true;

    controller.abort(
      new Error('Provider request timed out.')
    );
  }, REQUEST_TIMEOUT_MS);

  timeout.unref?.();

  const onRequestAborted = () => {
    clientDisconnected = true;

    controller.abort(
      new Error('Client disconnected.')
    );
  };

  const onResponseClose = () => {
    if (!res.writableEnded) {
      clientDisconnected = true;

      controller.abort(
        new Error('Client disconnected.')
      );
    }
  };

  req.once('aborted', onRequestAborted);
  res.once('close', onResponseClose);

  try {
    // Read and validate incoming request.

    const input = await readRequestBody(
      req,
      controller.signal
    );

    const validationError = validateChatRequest(input);

    if (validationError) {
      return sendError(
        res,
        400,
        validationError,
        'invalid_request_error'
      );
    }

    // Resolve requested model.

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

    if (!provider.apiKey) {
      return sendError(
        res,
        503,
        `${provider.name} is not configured. Add its API key to the server environment.`,
        'provider_not_configured'
      );
    }

    // Prepare provider-specific request.

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

    if (
      profile.adapter === 'mistral4' &&
      (
        !Number.isInteger(body.max_tokens) ||
        body.max_tokens < 1 ||
        body.max_tokens > 262144
      )
    ) {
      return sendError(
        res,
        400,
        'Mistral Large 4 max_tokens must be between 1 and 262144.',
        'invalid_request_error'
      );
    }

    if (shuttingDown) {
      return sendError(
        res,
        503,
        'Server is shutting down.',
        'server_unavailable'
      );
    }

    console.log(JSON.stringify({
      type: 'generation_started',
      requestId,
      provider: profile.provider,
      model: profile.model,
      streaming,
      continuity: RP_CONTINUITY_MODE,
      mistralPrefixTranslated: prepared.prefixTranslated
    }));

    // Acquire provider concurrency slot.

    releaseQueue = await queues[profile.provider].acquire(
      controller.signal
    );

    if (controller.signal.aborted) return;

    // Submit request.

    const upstream = await fetch(
      `${provider.baseUrl}/chat/completions`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
          'Content-Type': 'application/json',
          Accept: streaming
            ? 'text/event-stream'
            : 'application/json'
        },
        body: JSON.stringify(body),
        signal: controller.signal,
        redirect: 'error'
      }
    );

    if (controller.signal.aborted || res.destroyed) {
      return;
    }

    // NVIDIA asynchronous request polling.

    let finalData = null;
    let completedByPolling = false;

    if (
      profile.provider === 'nvidia' &&
      upstream.status === 202
    ) {
      const pendingRead = await readLimited(
        upstream,
        1024 * 1024
      );

      if (pendingRead.truncated) {
        return sendError(
          res,
          502,
          'NVIDIA pending response exceeded the size limit.',
          'upstream_error'
        );
      }

      const pending = parseJson(pendingRead.text);

      const nvidiaRequestId = getRequestId(
        upstream.headers,
        pending
      );

      if (!nvidiaRequestId) {
        return sendError(
          res,
          502,
          'NVIDIA returned 202 Accepted without a request ID.',
          'upstream_error'
        );
      }

      finalData = await pollNvidiaResult(
        nvidiaRequestId,
        controller.signal
      );

      completedByPolling = true;
    } else if (!upstream.ok) {
      const errorRead = await readLimited(
        upstream,
        1024 * 1024
      );

      const message = extractErrorMessage(
        errorRead.text,
        `${provider.name} returned HTTP ${upstream.status}.`
      );

      console.error(JSON.stringify({
        type: 'generation_failed',
        requestId,
        provider: profile.provider,
        model: profile.model,
        status: upstream.status,
        durationMs: Date.now() - started
      }));

      return sendError(
        res,
        upstream.status,
        message,
        'upstream_error'
      );
    }

    if (controller.signal.aborted || res.destroyed) {
      return;
    }

    // -------------------------------------------------------------------------
    // STREAMING RESPONSE
    // -------------------------------------------------------------------------

    if (streaming) {
      if (completedByPolling) {
        await sendJsonAsSse(
          res,
          finalData,
          controller.signal
        );
      } else {
        const contentType =
          upstream.headers.get('content-type') || '';

        if (contentType.includes('text/event-stream')) {
          await streamResponse(
            upstream,
            res,
            controller.signal
          );
        } else {
          const read = await readLimited(
            upstream,
            MAX_RESPONSE_BYTES
          );

          if (read.truncated) {
            return sendError(
              res,
              502,
              'Provider response exceeded the configured size limit.',
              'upstream_error'
            );
          }

          const data = parseJson(read.text);

          if (data === null) {
            return sendError(
              res,
              502,
              'Provider returned invalid JSON for a streaming request.',
              'upstream_error'
            );
          }

          await sendJsonAsSse(
            res,
            data,
            controller.signal
          );
        }
      }

      console.log(JSON.stringify({
        type: 'generation_finished',
        requestId,
        provider: profile.provider,
        model: profile.model,
        durationMs: Date.now() - started
      }));

      return;
    }

    // -------------------------------------------------------------------------
    // NON-STREAMING RESPONSE
    // -------------------------------------------------------------------------

    let data = finalData;

    if (!completedByPolling) {
      const read = await readLimited(
        upstream,
        MAX_RESPONSE_BYTES
      );

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

    if (
      data === null ||
      data === undefined
    ) {
      return sendError(
        res,
        502,
        'Provider returned invalid or empty JSON.',
        'upstream_error'
      );
    }

    if (STRIP_REASONING) {
      data = stripReasoning(data);
    }

    sendJson(res, 200, data);

    console.log(JSON.stringify({
      type: 'generation_finished',
      requestId,
      provider: profile.provider,
      model: profile.model,
      durationMs: Date.now() - started
    }));

  } catch (error) {
    if (
      clientDisconnected ||
      res.destroyed ||
      res.writableEnded
    ) {
      return;
    }

    // Timeout handling.

    if (timedOut) {
      if (res.headersSent) {
        try {
          await writeChunk(
            res,
            `data: ${JSON.stringify({
              error: {
                message: 'Provider request timed out.',
                type: 'upstream_timeout'
              }
            })}\n\n`,
            new AbortController().signal
          );

          if (!res.writableEnded) {
            res.end();
          }
        } catch {}
      } else {
        sendError(
          res,
          504,
          'Provider request timed out.',
          'upstream_timeout'
        );
      }

      return;
    }

    if (controller.signal.aborted) {
      return;
    }

    const status = Number.isInteger(error.status)
      ? error.status
      : 502;

    const type = error.type ||
      (status === 503
        ? 'server_overloaded'
        : 'upstream_error');

    console.error(JSON.stringify({
      type: 'proxy_error',
      requestId,
      status,
      code: error.code || null,
      message: String(
        error.message || 'Request failed.'
      ).slice(0, 300)
    }));

    if (res.headersSent) {
      if (!res.destroyed && !res.writableEnded) {
        res.destroy();
      }
    } else {
      sendError(
        res,
        status,
        error.message || 'Provider request failed.',
        type
      );
    }

  } finally {
    clearTimeout(timeout);

    req.removeListener(
      'aborted',
      onRequestAborted
    );

    res.removeListener(
      'close',
      onResponseClose
    );

    releaseQueue?.();

    activeControllers.delete(controller);
  }
}

// =============================================================================
// 22. ROUTER AND ENDPOINTS
// =============================================================================

const server = http.createServer(async (req, res) => {
  const requestId = crypto.randomUUID();

  const started = Date.now();

  res.setHeader('X-Request-ID', requestId);
  res.setHeader('X-Content-Type-Options', 'nosniff');

  let path;
  let searchParams;

  try {
    const url = new URL(
      req.url,
      'http://localhost'
    );

    path = url.pathname;
    searchParams = url.searchParams;
  } catch {
    return sendError(
      res,
      400,
      'Invalid request URL.',
      'invalid_request_error'
    );
  }

  let logged = false;

  const log = () => {
    if (logged) return;

    logged = true;

    console.log(JSON.stringify({
      type: 'http_request',
      requestId,
      method: req.method,
      path,
      status: res.statusCode,
      durationMs: Date.now() - started
    }));
  };

  res.once('finish', log);
  res.once('close', log);

  // CORS

  if (!applyCors(req, res)) return;

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // Root endpoint

  if (req.method === 'GET' && path === '/') {
    return sendJson(res, 200, {
      name: 'JanitorAI Multi-Provider Proxy',

      status: shuttingDown
        ? 'shutting_down'
        : 'online',

      providers: {
        nvidia: Boolean(NVIDIA_API_KEY),
        gemini: Boolean(GEMINI_API_KEY)
      },

      default_model: DEFAULT_MODEL || null,

      allow_unknown_nvidia_models:
        ALLOW_UNKNOWN_NVIDIA_MODELS,

      endpoints: [
        '/health',
        '/v1/models',
        '/v1/chat/completions'
      ]
    });
  }

  // Health endpoint

  if (req.method === 'GET' && path === '/health') {
    return sendJson(
      res,
      shuttingDown ? 503 : 200,
      {
        status: shuttingDown
          ? 'shutting_down'
          : 'ok',

        providers: {
          nvidia: {
            configured: Boolean(NVIDIA_API_KEY),
            active: queues.nvidia.active,
            queued: queues.nvidia.waiting.length
          },

          gemini: {
            configured: Boolean(GEMINI_API_KEY),
            active: queues.gemini.active,
            queued: queues.gemini.waiting.length
          }
        },

        roleplay_continuity: RP_CONTINUITY_MODE,

        mistral_assistant_prefix_mode:
          MISTRAL_PREFIX_MODE,

        reasoning_filter: STRIP_REASONING,

        uptime_seconds: Math.floor(process.uptime())
      }
    );
  }

  // Authenticate API routes.

  if (
    path.startsWith('/v1/') &&
    !authenticate(req, res)
  ) {
    return;
  }

  // Model discovery

  if (
    req.method === 'GET' &&
    path === '/v1/models'
  ) {
    try {
      const models = await getModelList(
        searchParams.get('refresh') === 'true'
      );

      return sendJson(res, 200, models);
    } catch (error) {
      return sendError(
        res,
        502,
        error.message || 'Model discovery failed.',
        'model_discovery_error'
      );
    }
  }

  // Chat completions

  if (
    req.method === 'POST' &&
    path === '/v1/chat/completions'
  ) {
    return chatCompletions(
      req,
      res,
      requestId
    );
  }

  return sendError(
    res,
    404,
    'Endpoint not found.',
    'not_found'
  );
});

// =============================================================================
// 23. SERVER SETTINGS
// =============================================================================

server.requestTimeout = 0;
server.timeout = 0;

server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;

server.maxHeadersCount = 100;

server.on('error', error => {
  console.error(
    '[SERVER ERROR]',
    error.message
  );

  process.exitCode = 1;
});

server.on('clientError', (error, socket) => {
  console.error(
    '[CLIENT ERROR]',
    error.message
  );

  if (socket.writable) {
    socket.end(
      'HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n'
    );
  }
});

// =============================================================================
// 24. START SERVER
// =============================================================================

server.listen(PORT, HOST, () => {
  console.log('==============================================');
  console.log(' JANITORAI MULTI-PROVIDER ROLEPLAY PROXY');
  console.log('==============================================');

  console.log(`Host:                 ${HOST}`);
  console.log(`Port:                 ${PORT}`);

  console.log(`NVIDIA configured:    ${Boolean(NVIDIA_API_KEY)}`);
  console.log(`Gemini configured:    ${Boolean(GEMINI_API_KEY)}`);

  console.log(
    `Default model:        ${DEFAULT_MODEL || '(request must specify model)'}`
  );

  console.log(
    `Unknown NVIDIA IDs:   ${ALLOW_UNKNOWN_NVIDIA_MODELS}`
  );

  console.log(`NVIDIA concurrency:   ${NVIDIA_CONCURRENCY}`);
  console.log(`Gemini concurrency:   ${GEMINI_CONCURRENCY}`);

  console.log(
    `Queue limit/provider: ${REQUEST_QUEUE_LIMIT}`
  );

  console.log(
    `Queue timeout:        ${REQUEST_QUEUE_TIMEOUT_MS}ms`
  );

  console.log(
    `Request timeout:      ${REQUEST_TIMEOUT_MS}ms`
  );

  console.log(
    `Default max tokens:   ${DEFAULT_MAX_TOKENS}`
  );

  console.log(
    `Maximum request:      ${MAX_BODY_BYTES} bytes`
  );

  console.log(
    `Maximum response:     ${MAX_RESPONSE_BYTES} bytes`
  );

  console.log(
    `Strip reasoning:      ${STRIP_REASONING}`
  );

  console.log(
    `RP continuity:        ${RP_CONTINUITY_MODE}`
  );

  console.log(
    `Mistral prefix mode:  ${MISTRAL_PREFIX_MODE}`
  );

  console.log(
    `Proxy authentication: ${Boolean(PROXY_API_KEY)}`
  );

  console.log(
    'Conversation memory:  stateless; supplied transcript only'
  );

  if (!PROXY_API_KEY) {
    console.warn(
      '[SECURITY WARNING] PROXY_API_KEY is unset; /v1 endpoints are not authenticated.'
    );
  }

  console.log('==============================================');
});

// =============================================================================
// 25. GRACEFUL SHUTDOWN
// =============================================================================

function shutdown(signal) {
  if (shuttingDown) return;

  shuttingDown = true;

  console.log(
    `[SERVER] ${signal} received; shutting down.`
  );

  // Cancel all active provider requests.

  for (const controller of activeControllers) {
    controller.abort(
      new Error('Server is shutting down.')
    );
  }

  // Stop accepting new connections.

  const forceExit = setTimeout(() => {
    console.error(
      '[SERVER] Forced shutdown after grace period.'
    );

    server.closeAllConnections?.();

    process.exit(1);
  }, 10000);

  server.close(() => {
    clearTimeout(forceExit);

    console.log(
      '[SERVER] Shutdown complete.'
    );

    process.exit(0);
  });

  server.closeIdleConnections?.();
}

process.once(
  'SIGTERM',
  () => shutdown('SIGTERM')
);

process.once(
  'SIGINT',
  () => shutdown('SIGINT')
);
