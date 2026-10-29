'use strict';

const express = require('express');
const axios = require('axios');
const crypto = require('node:crypto');
const { StringDecoder } = require('node:string_decoder');

const app = express();

// ============================================================
// CONFIGURATION
// ============================================================

const PORT = Number(process.env.PORT || 3000);

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY;

const NVIDIA_BASE_URL = (
  process.env.NVIDIA_BASE_URL ||
  'https://integrate.api.nvidia.com/v1'
).replace(/\/+$/, '');

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  'mistralai/mistral-large-4-1t-a52b';

const REQUEST_TIMEOUT_MS = Number(
  process.env.REQUEST_TIMEOUT_MS || 600000
);

const MAX_BODY_BYTES = process.env.MAX_BODY_BYTES || '10mb';

const MAX_CONCURRENT_REQUESTS = Number(
  process.env.MAX_CONCURRENT_REQUESTS || 1
);

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 8192
);

const STRIP_REASONING =
  String(process.env.STRIP_REASONING || 'true').toLowerCase() === 'true';

const ALLOW_UNKNOWN_MODELS =
  String(process.env.ALLOW_UNKNOWN_MODELS || 'false').toLowerCase() === 'true';

const PROXY_API_KEY = process.env.PROXY_API_KEY || '';

const ALLOWED_ORIGINS = (
  process.env.CORS_ORIGINS || 'https://janitorai.com'
)
  .split(',')
  .map((origin) => origin.trim().replace(/\/+$/, ''))
  .filter(Boolean);

if (!NVIDIA_API_KEY) {
  console.error('FATAL: NVIDIA_API_KEY is missing.');
  process.exit(1);
}

if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) {
  throw new Error('PORT must be a valid TCP port.');
}

if (!Number.isFinite(REQUEST_TIMEOUT_MS) || REQUEST_TIMEOUT_MS < 1000) {
  throw new Error('REQUEST_TIMEOUT_MS must be at least 1000.');
}

if (
  !Number.isInteger(MAX_CONCURRENT_REQUESTS) ||
  MAX_CONCURRENT_REQUESTS < 1
) {
  throw new Error('MAX_CONCURRENT_REQUESTS must be at least 1.');
}

if (!Number.isInteger(DEFAULT_MAX_TOKENS) || DEFAULT_MAX_TOKENS < 1) {
  throw new Error('DEFAULT_MAX_TOKENS must be a positive integer.');
}

if (PROXY_API_KEY && PROXY_API_KEY.length < 32) {
  throw new Error('PROXY_API_KEY must be at least 32 characters.');
}

if (!NVIDIA_BASE_URL.startsWith('https://')) {
  throw new Error('NVIDIA_BASE_URL must use HTTPS.');
}

app.disable('x-powered-by');
app.set('trust proxy', 1);

// ============================================================
// MODEL PROFILES
//
// Only hosted NVIDIA Free Endpoint models are registered here.
// A JanitorAI alias is resolved to the actual upstream model ID.
//
// Different models expose different reasoning controls.
// Profiles are only created where the documented API supports
// the relevant setting; unsupported levels are not fabricated.
// ============================================================

const MODEL_PROFILES = Object.create(null);

function addModel(id, label, upstream, settings = {}) {
  MODEL_PROFILES[id] = {
    id,
    label,
    upstream,
    adapter: 'generic',
    ...settings
  };
}

// ------------------------------------------------------------
// MISTRAL LARGE 4
//
// Current NVIDIA API reference documents:
//   reasoning_effort: none | high
//
// It does not document low/medium/high for this endpoint.
// ------------------------------------------------------------

const MISTRAL4 = 'mistralai/mistral-large-4-1t-a52b';

addModel(
  MISTRAL4,
  'Mistral Large 4 - High Reasoning',
  MISTRAL4,
  {
    adapter: 'mistral4',
    effort: 'high'
  }
);

addModel(
  `${MISTRAL4}-thinking`,
  'Mistral Large 4 - Thinking',
  MISTRAL4,
  {
    adapter: 'mistral4',
    effort: 'high'
  }
);

addModel(
  `${MISTRAL4}-no-thinking`,
  'Mistral Large 4 - No Thinking',
  MISTRAL4,
  {
    adapter: 'mistral4',
    effort: 'none'
  }
);

// ------------------------------------------------------------
// MISTRAL NEMOTRON
//
// Hosted Free Endpoint. No special reasoning adapter is added
// without a documented endpoint-specific reasoning control.
// ------------------------------------------------------------

addModel(
  'mistralai/mistral-nemotron',
  'Mistral Nemotron',
  'mistralai/mistral-nemotron'
);

// ------------------------------------------------------------
// NEMOTRON 3 ULTRA
//
// reasoning_effort: none | medium | high
// reasoning_budget: integer from -1 to 32768
// ------------------------------------------------------------

const NEMOTRON_ULTRA = 'nvidia/nemotron-3-ultra-550b-a55b';

addModel(
  NEMOTRON_ULTRA,
  'Nemotron 3 Ultra - High Reasoning',
  NEMOTRON_ULTRA,
  {
    adapter: 'nemotron-ultra',
    effort: 'high',
    reasoningBudget: 4096
  }
);

addModel(
  `${NEMOTRON_ULTRA}-no-thinking`,
  'Nemotron 3 Ultra - No Thinking',
  NEMOTRON_ULTRA,
  {
    adapter: 'nemotron-ultra',
    effort: 'none'
  }
);

addModel(
  `${NEMOTRON_ULTRA}-medium`,
  'Nemotron 3 Ultra - Medium Reasoning',
  NEMOTRON_ULTRA,
  {
    adapter: 'nemotron-ultra',
    effort: 'medium',
    reasoningBudget: 2048
  }
);

addModel(
  `${NEMOTRON_ULTRA}-high`,
  'Nemotron 3 Ultra - High Reasoning',
  NEMOTRON_ULTRA,
  {
    adapter: 'nemotron-ultra',
    effort: 'high',
    reasoningBudget: 4096
  }
);

// ------------------------------------------------------------
// NEMOTRON 3.5 SUPER VL
//
// reasoning_effort: none | low | high
// ------------------------------------------------------------

const NEMOTRON_VL = 'nvidia/nemotron-3.5-super-vl-120b-a12b';

addModel(
  NEMOTRON_VL,
  'Nemotron 3.5 Super VL - High Reasoning',
  NEMOTRON_VL,
  {
    adapter: 'nemotron-vl',
    effort: 'high'
  }
);

addModel(
  `${NEMOTRON_VL}-no-thinking`,
  'Nemotron 3.5 Super VL - No Thinking',
  NEMOTRON_VL,
  {
    adapter: 'nemotron-vl',
    effort: 'none'
  }
);

addModel(
  `${NEMOTRON_VL}-low`,
  'Nemotron 3.5 Super VL - Low Reasoning',
  NEMOTRON_VL,
  {
    adapter: 'nemotron-vl',
    effort: 'low'
  }
);

addModel(
  `${NEMOTRON_VL}-high`,
  'Nemotron 3.5 Super VL - High Reasoning',
  NEMOTRON_VL,
  {
    adapter: 'nemotron-vl',
    effort: 'high'
  }
);

// ------------------------------------------------------------
// GLM 5.3 FLASH
//
// reasoning_effort: low | high | max
// "Medium" is mapped to the documented API level "high".
// "High" is mapped to "max".
// There is no no-thinking alias: low does not mean no thinking.
// ------------------------------------------------------------

const GLM = 'z-ai/glm-5.3-flash';

addModel(
  GLM,
  'GLM 5.3 Flash - High Reasoning',
  GLM,
  {
    adapter: 'glm',
    effort: 'high'
  }
);

addModel(
  `${GLM}-low`,
  'GLM 5.3 Flash - Low Reasoning',
  GLM,
  {
    adapter: 'glm',
    effort: 'low'
  }
);

addModel(
  `${GLM}-medium`,
  'GLM 5.3 Flash - Medium (API High)',
  GLM,
  {
    adapter: 'glm',
    effort: 'high'
  }
);

addModel(
  `${GLM}-high`,
  'GLM 5.3 Flash - High (API Max)',
  GLM,
  {
    adapter: 'glm',
    effort: 'max'
  }
);

// ------------------------------------------------------------
// DEEPSEEK V4.1 FLASH
//
// NVIDIA documents continuously controlled numeric reasoning
// effort from 1 through 100.
// ------------------------------------------------------------

const DEEPSEEK = 'deepseek-ai/deepseek-v4.1-flash';

addModel(
  DEEPSEEK,
  'DeepSeek V4.1 Flash - Medium Reasoning',
  DEEPSEEK,
  {
    adapter: 'deepseek',
    effort: 50
  }
);

addModel(
  `${DEEPSEEK}-low`,
  'DeepSeek V4.1 Flash - Low Reasoning',
  DEEPSEEK,
  {
    adapter: 'deepseek',
    effort: 25
  }
);

addModel(
  `${DEEPSEEK}-medium`,
  'DeepSeek V4.1 Flash - Medium Reasoning',
  DEEPSEEK,
  {
    adapter: 'deepseek',
    effort: 50
  }
);

addModel(
  `${DEEPSEEK}-high`,
  'DeepSeek V4.1 Flash - High Reasoning',
  DEEPSEEK,
  {
    adapter: 'deepseek',
    effort: 100
  }
);

// ------------------------------------------------------------
// STEP 5 PREVIEW
//
// NVIDIA documents that reasoning remains enabled and that
// enable_thinking=false does not disable it for this checkpoint.
// No fake reasoning levels are added.
// ------------------------------------------------------------

addModel(
  'stepfun-ai/step-5-preview',
  'Step 5 Preview',
  'stepfun-ai/step-5-preview',
  {
    adapter: 'step'
  }
);

// ------------------------------------------------------------
// MINIMAX M3.1 FLASH
//
// NVIDIA's model documentation describes the levels:
// low | medium | high | xhigh | max
//
// The reasoning setting is passed through chat_template_kwargs.
// ------------------------------------------------------------

const MINIMAX = 'minimaxai/minimax-m3.1-flash';

addModel(
  MINIMAX,
  'MiniMax M3.1 Flash - Medium Reasoning',
  MINIMAX,
  {
    adapter: 'minimax',
    effort: 'medium'
  }
);

for (const effort of ['low', 'medium', 'high', 'xhigh', 'max']) {
  addModel(
    `${MINIMAX}-${effort}`,
    `MiniMax M3.1 Flash - ${effort.toUpperCase()} Reasoning`,
    MINIMAX,
    {
      adapter: 'minimax',
      effort
    }
  );
}

// ------------------------------------------------------------
// POOLSIDE LAGUNA XS 2.1
//
// Hosted Free Endpoint. Primarily an agentic coding model.
// ------------------------------------------------------------

addModel(
  'poolside/laguna-xs-2.1',
  'Laguna XS 2.1',
  'poolside/laguna-xs-2.1'
);

// ------------------------------------------------------------
// GPT OSS 20B
// ------------------------------------------------------------

addModel(
  'openai/gpt-oss-20b',
  'GPT OSS 20B',
  'openai/gpt-oss-20b'
);

// ------------------------------------------------------------
// GEMMA 4 31B IT
//
// NVIDIA supports enable_thinking through chat_template_kwargs.
// ------------------------------------------------------------

const GEMMA = 'google/gemma-4-31b-it';

addModel(
  GEMMA,
  'Gemma 4 31B IT - Thinking',
  GEMMA,
  {
    adapter: 'gemma',
    thinking: true
  }
);

addModel(
  `${GEMMA}-thinking`,
  'Gemma 4 31B IT - Thinking',
  GEMMA,
  {
    adapter: 'gemma',
    thinking: true
  }
);

addModel(
  `${GEMMA}-no-thinking`,
  'Gemma 4 31B IT - No Thinking',
  GEMMA,
  {
    adapter: 'gemma',
    thinking: false
  }
);

// ------------------------------------------------------------
// MUSE GLIMMER 30B
//
// reasoning_effort: none | minimal | low | medium | high | max
// ------------------------------------------------------------

const MUSE = 'meta/muse-glimmer-30b';

addModel(
  MUSE,
  'Muse Glimmer 30B - Medium Reasoning',
  MUSE,
  {
    adapter: 'muse',
    effort: 'medium'
  }
);

addModel(
  `${MUSE}-no-thinking`,
  'Muse Glimmer 30B - No Thinking',
  MUSE,
  {
    adapter: 'muse',
    effort: 'none'
  }
);

for (const effort of ['low', 'medium', 'high', 'max']) {
  addModel(
    `${MUSE}-${effort}`,
    `Muse Glimmer 30B - ${effort.toUpperCase()} Reasoning`,
    MUSE,
    {
      adapter: 'muse',
      effort
    }
  );
}

// Ensure a typo in DEFAULT_MODEL cannot silently create a broken server.
if (!Object.prototype.hasOwnProperty.call(MODEL_PROFILES, DEFAULT_MODEL)) {
  throw new Error(
    `DEFAULT_MODEL "${DEFAULT_MODEL}" is not in the registered model profiles.`
  );
}

// ============================================================
// REQUEST LOGGING
// Never log prompts, message content, or API keys.
// ============================================================

app.use((req, res, next) => {
  const started = Date.now();

  res.once('finish', () => {
    console.log(
      `${req.method} ${req.path} -> ${res.statusCode} ` +
      `(${Date.now() - started}ms)`
    );
  });

  next();
});

// ============================================================
// CORS
// ============================================================

app.use((req, res, next) => {
  const rawOrigin = req.get('origin');
  const origin = rawOrigin
    ? rawOrigin.replace(/\/+$/, '')
    : '';

  const allowed =
    !origin ||
    ALLOWED_ORIGINS.includes('*') ||
    ALLOWED_ORIGINS.includes(origin);

  if (rawOrigin && allowed) {
    res.setHeader('Access-Control-Allow-Origin', rawOrigin);
    res.setHeader('Vary', 'Origin');

    res.setHeader(
      'Access-Control-Allow-Methods',
      'GET, POST, OPTIONS'
    );

    res.setHeader(
      'Access-Control-Allow-Headers',
      req.get('access-control-request-headers') ||
        'Authorization, Content-Type'
    );

    res.setHeader('Access-Control-Max-Age', '86400');
  }

  if (req.method === 'OPTIONS') {
    return res.sendStatus(allowed ? 204 : 403);
  }

  if (!allowed) {
    return res.status(403).json({
      error: {
        message: 'Origin not allowed.',
        type: 'cors_error'
      }
    });
  }

  next();
});

// ============================================================
// JSON PARSING
// ============================================================

app.use(express.json({
  limit: MAX_BODY_BYTES,
  strict: true
}));

// ============================================================
// OPTIONAL PROXY AUTHENTICATION
//
// For a public Render URL, set PROXY_API_KEY to a random,
// private string of at least 32 characters.
// Use that key in JanitorAI as the API key for this proxy.
// ============================================================

if (PROXY_API_KEY) {
  const expectedKey = Buffer.from(PROXY_API_KEY);

  app.use('/v1', (req, res, next) => {
    const match = (req.get('authorization') || '')
      .match(/^Bearer\s+(.+)$/i);

    const suppliedKey = Buffer.from(match ? match[1] : '');

    const valid =
      suppliedKey.length === expectedKey.length &&
      crypto.timingSafeEqual(suppliedKey, expectedKey);

    if (!valid) {
      return res.status(401).json({
        error: {
          message: 'Invalid or missing proxy API key.',
          type: 'authentication_error'
        }
      });
    }

    next();
  });
} else {
  console.warn(
    'WARNING: PROXY_API_KEY is unset; /v1 endpoints are publicly accessible.'
  );
}

// ============================================================
// CONCURRENCY CONTROL
//
// One generation at a time by default. Only chat completions
// use a slot; health and model-list requests remain available.
// No queue, Redis, or other external service.
// ============================================================

let activeRequests = 0;

function concurrencyMiddleware(req, res, next) {
  if (activeRequests >= MAX_CONCURRENT_REQUESTS) {
    res.setHeader('Retry-After', '3');

    return res.status(503).json({
      error: {
        message: 'A generation is already running. Try again shortly.',
        type: 'server_overloaded'
      }
    });
  }

  activeRequests += 1;
  let released = false;

  const release = () => {
    if (released) return;
    released = true;
    activeRequests = Math.max(0, activeRequests - 1);
  };

  res.once('finish', release);
  res.once('close', release);

  next();
}

// ============================================================
// ERROR HELPERS
// ============================================================

function jsonError(res, status, message, type = 'proxy_error') {
  if (res.headersSent) {
    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }
    return;
  }

  return res.status(status).json({
    error: {
      message: String(message || 'Unknown error').slice(0, 4000),
      type
    }
  });
}

function upstreamErrorMessage(
  data,
  fallback = 'Unknown NVIDIA error.'
) {
  let parsed = data;

  if (typeof parsed === 'string') {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      return parsed.slice(0, 3000) || fallback;
    }
  }

  if (parsed && typeof parsed === 'object') {
    const message =
      parsed.error?.message ||
      parsed.message ||
      parsed.detail;

    if (typeof message === 'string') {
      return message.slice(0, 3000);
    }
  }

  return fallback;
}

function parseJsonOrText(text) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function extractRequestId(headers, payload) {
  const headerId =
    headers?.['nvcf-reqid'] ||
    headers?.['nvcf-request-id'] ||
    headers?.['x-request-id'] ||
    headers?.['request-id'];

  if (headerId) return String(headerId);

  if (payload && typeof payload === 'object') {
    const bodyId =
      payload.requestId ||
      payload.request_id ||
      payload.id;

    if (bodyId) return String(bodyId);
  }

  return null;
}

async function readStreamText(stream, maxBytes = 1024 * 1024) {
  const chunks = [];
  let total = 0;

  for await (const chunk of stream) {
    const part = Buffer.from(chunk);
    const remaining = maxBytes - total;

    if (remaining <= 0) break;

    const accepted = part.subarray(0, remaining);
    chunks.push(accepted);
    total += accepted.length;

    if (total >= maxBytes) break;
  }

  if (!stream.destroyed && !stream.readableEnded) {
    stream.destroy();
  }

  return Buffer.concat(chunks).toString('utf8');
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      return reject(
        signal.reason || new Error('Request was cancelled.')
      );
    }

    const timer = setTimeout(cleanupResolve, ms);

    function cleanup() {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    }

    function cleanupResolve() {
      cleanup();
      resolve();
    }

    function onAbort() {
      cleanup();
      reject(signal.reason || new Error('Request was cancelled.'));
    }

    signal.addEventListener('abort', onAbort, { once: true });
  });
}

// ============================================================
// NVIDIA ASYNC RESULT POLLING
//
// Some NVIDIA endpoints return 202 Accepted and a request ID.
// Poll /v1/status/{requestId} until the result is ready.
// The generation timeout and AbortController cover polling too.
// ============================================================

async function pollNvidiaResult(requestId, signal) {
  const statusUrl =
    `${NVIDIA_BASE_URL}/status/${encodeURIComponent(requestId)}`;

  let pollCount = 0;

  while (true) {
    if (signal.aborted) {
      throw signal.reason || new Error('Request was cancelled.');
    }

    pollCount += 1;

    // Avoid hammering the status API while inference is pending.
    await sleep(Math.min(1000 + pollCount * 250, 3000), signal);

    const response = await axios.get(statusUrl, {
      headers: {
        Authorization: `Bearer ${NVIDIA_API_KEY}`,
        Accept: 'application/json'
      },
      responseType: 'json',
      timeout: 0,
      signal,
      validateStatus: () => true,
      maxRedirects: 0
    });

    if (response.status === 200) {
      return response.data;
    }

    if (response.status === 202) {
      continue;
    }

    const error = new Error(
      upstreamErrorMessage(
        response.data,
        `NVIDIA status polling returned HTTP ${response.status}.`
      )
    );

    error.response = {
      status: response.status,
      data: response.data
    };

    throw error;
  }
}

// ============================================================
// MODEL RESOLUTION
// ============================================================

function getProfile(requestedModel) {
  if (
    typeof requestedModel === 'string' &&
    Object.prototype.hasOwnProperty.call(
      MODEL_PROFILES,
      requestedModel
    )
  ) {
    return MODEL_PROFILES[requestedModel];
  }

  if (
    ALLOW_UNKNOWN_MODELS &&
    typeof requestedModel === 'string' &&
    requestedModel.trim()
  ) {
    const model = requestedModel.trim();

    return {
      id: model,
      label: model,
      upstream: model,
      adapter: 'generic'
    };
  }

  return null;
}

// ============================================================
// MODEL-SPECIFIC REQUEST ADAPTERS
// ============================================================

function applyProfile(requestBody, profile) {
  const body = { ...requestBody };

  body.model = profile.upstream;

  const template = {
    ...(body.chat_template_kwargs || {})
  };

  switch (profile.adapter) {
    case 'mistral4': {
      body.reasoning_effort = profile.effort;

      // This recipe does not honor enable_thinking.
      delete template.enable_thinking;
      break;
    }

    case 'nemotron-vl': {
      body.reasoning_effort = profile.effort;

      // NVIDIA recommends reasoning_effort for this endpoint.
      delete template.enable_thinking;
      break;
    }

    case 'glm': {
      body.reasoning_effort = profile.effort;

      // Keep thinking traces separate from the chat answer.
      template.clear_thinking = true;
      break;
    }

    case 'minimax': {
      // Important: use the top-level field expected by the
      // hosted NVIDIA endpoint, not chat_template_kwargs.
      body.reasoning_effort = profile.effort;
      delete template.reasoning_effort;
      break;
    }

    case 'step': {
      // Do not pretend enable_thinking=false disables reasoning
      // for this checkpoint.
      delete template.enable_thinking;
      break;
    }

    default: {
      // Leave settings untouched for profiles without an
      // endpoint-specific adapter.
      break;
    }
  }

  if (Object.keys(template).length > 0) {
    body.chat_template_kwargs = template;
  } else {
    delete body.chat_template_kwargs;
  }

  return body;
}

// ============================================================
// REQUEST VALIDATION
// Does not rewrite or remove any conversation messages.
// ============================================================

function validateChatRequest(body) {
  if (
    !body ||
    typeof body !== 'object' ||
    Array.isArray(body)
  ) {
    return 'Request body must be a JSON object.';
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

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return 'messages must be a non-empty array.';
  }

  for (const message of body.messages) {
    if (
      !message ||
      typeof message !== 'object' ||
      Array.isArray(message) ||
      typeof message.role !== 'string'
    ) {
      return 'Each message must be an object with a string role.';
    }
  }

  for (const key of ['temperature', 'top_p']) {
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

  for (const key of ['max_tokens', 'max_completion_tokens']) {
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

  return null;
}

// ============================================================
// REASONING FIELD FILTER
//
// Removes separate reasoning fields from responses while leaving
// normal message content, tool calls, and request history alone.
// ============================================================

const REASONING_KEYS = new Set([
  'reasoning',
  'reasoning_content',
  'thinking',
  'internal_reasoning'
]);

function stripReasoningFields(value) {
  if (Array.isArray(value)) {
    return value.map(stripReasoningFields);
  }

  if (!value || typeof value !== 'object') {
    return value;
  }

  const output = {};

  for (const [key, child] of Object.entries(value)) {
    if (
      STRIP_REASONING &&
      REASONING_KEYS.has(key.toLowerCase())
    ) {
      continue;
    }

    output[key] = stripReasoningFields(child);
  }

  return output;
}

// ============================================================
// UPSTREAM HEADERS
// ============================================================

function upstreamHeaders(streaming) {
  return {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    'Content-Type': 'application/json',
    Accept: streaming
      ? 'text/event-stream'
      : 'application/json'
  };
}

// ============================================================
// SSE PARSER
// Parses complete SSE events rather than modifying arbitrary
// network chunks. UTF-8 data is decoded safely by StringDecoder.
// ============================================================

function extractSseEvents(buffer) {
  const events = [];
  let remaining = buffer;
  let match;

  while ((match = /\r?\n\r?\n/.exec(remaining)) !== null) {
    events.push(remaining.slice(0, match.index));
    remaining = remaining.slice(match.index + match[0].length);
  }

  return { events, remaining };
}

function transformSseEvent(eventText) {
  if (!eventText) return '';

  const lines = eventText.split(/\r?\n/);
  const dataLines = [];
  const otherLines = [];

  for (const line of lines) {
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''));
    } else {
      otherLines.push(line);
    }
  }

  if (dataLines.length === 0) {
    return `${eventText}\n\n`;
  }

  const data = dataLines.join('\n');
  const prefix = otherLines.length
    ? `${otherLines.join('\n')}\n`
    : '';

  if (data.trim() === '[DONE]') {
    return `${prefix}data: [DONE]\n\n`;
  }

  let outputData = data;

  if (STRIP_REASONING) {
    try {
      outputData = JSON.stringify(
        stripReasoningFields(JSON.parse(data))
      );
    } catch {
      // Preserve an event that cannot safely be parsed.
      outputData = data;
    }
  }

  return `${prefix}data: ${outputData}\n\n`;
}

function isDoneEvent(output) {
  return /(?:^|\n)data:\s*\[DONE\]\s*(?:\n|$)/.test(output);
}

// ============================================================
// SSE BACKPRESSURE
// ============================================================

function writeChunk(res, chunk) {
  if (res.destroyed || res.writableEnded) {
    return Promise.resolve(false);
  }

  if (res.write(chunk)) {
    return Promise.resolve(true);
  }

  return new Promise((resolve) => {
    const cleanup = () => {
      res.removeListener('drain', onDrain);
      res.removeListener('close', onClose);
    };

    const onDrain = () => {
      cleanup();
      resolve(!res.destroyed && !res.writableEnded);
    };

    const onClose = () => {
      cleanup();
      resolve(false);
    };

    res.once('drain', onDrain);
    res.once('close', onClose);

    if (res.destroyed || res.writableEnded) {
      onClose();
    }
  });
}

function startSseResponse(res) {
  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();
}

// ============================================================
// ADAPT A POLLED COMPLETION TO SSE
//
// If NVIDIA answers the original streaming POST with 202, the
// status endpoint eventually returns a completed JSON response.
// Convert that response to OpenAI-compatible SSE for JanitorAI.
// ============================================================

async function sendCompletionAsSse(res, originalData) {
  const data = STRIP_REASONING
    ? stripReasoningFields(originalData)
    : originalData;

  startSseResponse(res);

  if (!data || !Array.isArray(data.choices)) {
    await writeChunk(res, `data: ${JSON.stringify(data)}\n\n`);
    await writeChunk(res, 'data: [DONE]\n\n');

    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }

    return;
  }

  const id = data.id || `chatcmpl-${crypto.randomUUID()}`;
  const created = data.created || Math.floor(Date.now() / 1000);
  const model = data.model || 'nvidia-nim-proxy';

  for (const choice of data.choices) {
    const message = choice.message || {};
    const index = Number.isInteger(choice.index) ? choice.index : 0;

    const delta = { ...message };

    // A streamed delta uses role, not the full message object.
    delete delta.role;

    const contentChunk = {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [
        {
          index,
          delta: {
            role: 'assistant',
            ...delta
          },
          finish_reason: null
        }
      ]
    };

    await writeChunk(res, `data: ${JSON.stringify(contentChunk)}\n\n`);

    const finishChunk = {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [
        {
          index,
          delta: {},
          finish_reason: choice.finish_reason || 'stop'
        }
      ]
    };

    await writeChunk(res, `data: ${JSON.stringify(finishChunk)}\n\n`);
  }

  if (data.usage) {
    const usageChunk = {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [],
      usage: data.usage
    };

    await writeChunk(res, `data: ${JSON.stringify(usageChunk)}\n\n`);
  }

  await writeChunk(res, 'data: [DONE]\n\n');

  if (!res.destroyed && !res.writableEnded) {
    res.end();
  }
}

// ============================================================
// HEALTH ENDPOINTS
// ============================================================

app.get('/', (req, res) => {
  res.json({
    name: 'NVIDIA NIM JanitorAI Proxy',
    status: 'ok',
    defaultModel: DEFAULT_MODEL,
    modelProfiles: Object.keys(MODEL_PROFILES).length,
    allowUnknownModels: ALLOW_UNKNOWN_MODELS,
    endpoints: [
      '/health',
      '/v1/models',
      '/v1/chat/completions'
    ]
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    activeRequests,
    maxConcurrentRequests: MAX_CONCURRENT_REQUESTS,
    modelProfiles: Object.keys(MODEL_PROFILES).length,
    allowUnknownModels: ALLOW_UNKNOWN_MODELS
  });
});

// ============================================================
// OPENAI-COMPATIBLE MODEL LIST
// ============================================================

app.get('/v1/models', (req, res) => {
  const created = Math.floor(Date.now() / 1000);

  const data = Object.values(MODEL_PROFILES).map((profile) => ({
    id: profile.id,
    object: 'model',
    created,
    owned_by: 'nvidia-nim-proxy',
    name: profile.label
  }));

  res.json({
    object: 'list',
    data
  });
});

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
  '/v1/chat/completions',
  concurrencyMiddleware,
  async (req, res) => {
    let requestBody = req.body;

    if (
      requestBody &&
      typeof requestBody === 'object' &&
      !Array.isArray(requestBody) &&
      !requestBody.model
    ) {
      requestBody = {
        ...requestBody,
        model: DEFAULT_MODEL
      };
    }

    const validationError = validateChatRequest(requestBody);

    if (validationError) {
      return jsonError(
        res,
        400,
        validationError,
        'invalid_request_error'
      );
    }

    const requestedModel = requestBody.model;
    const profile = getProfile(requestedModel);

    if (!profile) {
      return jsonError(
        res,
        400,
        `Unknown model "${requestedModel}".`,
        'invalid_request_error'
      );
    }

    // Crucial: do not remove, reorder, or truncate assistant
    // prefixes or previous conversation messages.
    const body = applyProfile(requestBody, profile);

    // Translate OpenAI's newer token-limit field to NVIDIA's
    // documented max_tokens field without overwriting max_tokens.
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

    const streaming = body.stream === true;
    const controller = new AbortController();

    let upstreamStream = null;
    let clientDisconnected = false;
    let timedOut = false;
    let timeout;

    const cancelUpstream = () => {
      if (clientDisconnected) return;

      clientDisconnected = true;
      controller.abort();

      if (upstreamStream && !upstreamStream.destroyed) {
        upstreamStream.destroy();
      }
    };

    const onRequestAborted = () => cancelUpstream();

    const onResponseClose = () => {
      if (!res.writableEnded) {
        cancelUpstream();
      }
    };

    req.once('aborted', onRequestAborted);
    res.once('close', onResponseClose);

    timeout = setTimeout(() => {
      if (clientDisconnected) return;

      timedOut = true;

      const error = Object.assign(
        new Error('NVIDIA request timed out.'),
        { code: 'UPSTREAM_TIMEOUT' }
      );

      console.warn('NVIDIA request timed out:', {
        model: requestedModel,
        timeoutMs: REQUEST_TIMEOUT_MS
      });

      controller.abort(error);

      if (upstreamStream && !upstreamStream.destroyed) {
        upstreamStream.destroy(error);
      }
    }, REQUEST_TIMEOUT_MS);

    try {
      let upstream = await axios.post(
        `${NVIDIA_BASE_URL}/chat/completions`,
        body,
        {
          headers: upstreamHeaders(streaming),
          responseType: streaming ? 'stream' : 'json',
          timeout: 0,
          signal: controller.signal,
          validateStatus: () => true,
          maxRedirects: 0,
          maxContentLength: 100 * 1024 * 1024,
          maxBodyLength: 100 * 1024 * 1024
        }
      );

      if (clientDisconnected || res.destroyed) {
        if (
          streaming &&
          upstream.data &&
          !upstream.data.destroyed
        ) {
          upstream.data.destroy();
        }

        return;
      }

      let completedByPolling = false;

      // ------------------------------------------------------
      // ASYNC ACCEPTED RESPONSE
      // ------------------------------------------------------

      if (upstream.status === 202) {
        let pendingPayload = upstream.data;

        if (streaming && upstream.data) {
          upstreamStream = upstream.data;

          const pendingText = await readStreamText(
            upstreamStream,
            1024 * 1024
          );

          upstreamStream = null;
          pendingPayload = parseJsonOrText(pendingText);
        }

        const requestId = extractRequestId(
          upstream.headers,
          pendingPayload
        );

        if (!requestId) {
          return jsonError(
            res,
            502,
            'NVIDIA returned 202 Accepted without a usable request ID.',
            'upstream_error'
          );
        }

        console.log('NVIDIA accepted asynchronous request:', {
          model: requestedModel
        });

        const completedData = await pollNvidiaResult(
          requestId,
          controller.signal
        );

        upstream = {
          ...upstream,
          status: 200,
          data: completedData
        };

        completedByPolling = true;
      }

      if (clientDisconnected || res.destroyed) {
        return;
      }

      // ------------------------------------------------------
      // UPSTREAM ERRORS
      // ------------------------------------------------------

      if (upstream.status < 200 || upstream.status >= 300) {
        let errorData = upstream.data;

        if (streaming && upstream.data) {
          upstreamStream = upstream.data;

          errorData = await readStreamText(upstreamStream, 64 * 1024);
          upstreamStream = null;
          errorData = parseJsonOrText(errorData);
        }

        if (clientDisconnected || res.destroyed) {
          return;
        }

        const message = upstreamErrorMessage(
          errorData,
          `NVIDIA returned HTTP ${upstream.status}.`
        );

        console.error('NVIDIA API error:', {
          status: upstream.status,
          model: requestedModel,
          upstreamModel: profile.upstream,
          streaming,
          message
        });

        return jsonError(
          res,
          upstream.status,
          message,
          'upstream_error'
        );
      }

      // ------------------------------------------------------
      // COMPLETED RESPONSE AFTER POLLING
      // ------------------------------------------------------

      if (streaming && completedByPolling) {
        await sendCompletionAsSse(res, upstream.data);
        return;
      }

      // ------------------------------------------------------
      // NORMAL STREAMING RESPONSE
      // ------------------------------------------------------

      if (streaming) {
        upstreamStream = upstream.data;
        startSseResponse(res);

        const decoder = new StringDecoder('utf8');

        let buffer = '';
        let receivedDone = false;

        try {
          for await (const chunk of upstreamStream) {
            if (clientDisconnected || res.destroyed) {
              return;
            }

            buffer += decoder.write(chunk);

            const parsed = extractSseEvents(buffer);
            buffer = parsed.remaining;

            for (const eventText of parsed.events) {
              if (clientDisconnected || res.destroyed) {
                return;
              }

              const output = transformSseEvent(eventText);

              if (!output) continue;

              if (isDoneEvent(output)) {
                receivedDone = true;
              }

              const writable = await writeChunk(res, output);

              if (!writable) return;

              if (receivedDone) break;
            }

            if (receivedDone) break;
          }

          if (clientDisconnected || res.destroyed) {
            return;
          }

          if (!receivedDone) {
            buffer += decoder.end();

            if (buffer.trim()) {
              const finalText = buffer.replace(
                /(?:\r?\n)+$/,
                ''
              );

              const output = transformSseEvent(finalText);

              if (output) {
                const writable = await writeChunk(res, output);
                if (!writable) return;

                if (isDoneEvent(output)) {
                  receivedDone = true;
                }
              }
            }
          }

          if (clientDisconnected || res.destroyed) {
            return;
          }

          if (!receivedDone) {
            const writable = await writeChunk(
              res,
              'data: [DONE]\n\n'
            );

            if (!writable) return;
          }

          if (!res.destroyed && !res.writableEnded) {
            res.end();
          }
        } catch (error) {
          if (
            clientDisconnected ||
            res.destroyed ||
            error?.code === 'ERR_CANCELED' ||
            error?.code === 'ABORT_ERR'
          ) {
            return;
          }

          console.error('NVIDIA streaming failed:', {
            code: error?.code || 'UNKNOWN',
            message: error?.message || 'Unknown stream error',
            model: requestedModel,
            timedOut
          });

          if (!res.destroyed && !res.writableEnded) {
            const message = timedOut
              ? 'NVIDIA generation timed out.'
              : 'NVIDIA streaming connection failed.';

            const errorEvent = `data: ${JSON.stringify({
              error: {
                message,
                type: timedOut
                  ? 'upstream_timeout'
                  : 'upstream_error'
              }
            })}\n\n`;

            await writeChunk(res, errorEvent);

            if (!res.destroyed && !res.writableEnded) {
              res.end();
            }
          }
        }

        return;
      }

      // ------------------------------------------------------
      // NORMAL NON-STREAMING RESPONSE
      // ------------------------------------------------------

      const responseData = STRIP_REASONING
        ? stripReasoningFields(upstream.data)
        : upstream.data;

      return res.status(upstream.status).json(responseData);
    } catch (error) {
      if (clientDisconnected || res.destroyed) {
        return;
      }

      const isTimeout =
        timedOut ||
        error?.code === 'UPSTREAM_TIMEOUT' ||
        error?.cause?.code === 'UPSTREAM_TIMEOUT' ||
        controller.signal.reason?.code === 'UPSTREAM_TIMEOUT';

      if (isTimeout) {
        return jsonError(
          res,
          504,
          'NVIDIA request timed out.',
          'upstream_timeout'
        );
      }

      if (
        error?.code === 'ERR_CANCELED' ||
        controller.signal.aborted
      ) {
        return;
      }

      const status = error?.response?.status || 502;

      const message = upstreamErrorMessage(
        error?.response?.data,
        error?.message || 'Could not reach NVIDIA.'
      );

      console.error('Proxy request failed:', {
        status,
        code: error?.code || 'UNKNOWN',
        message,
        model: requestedModel,
        upstreamModel: profile.upstream
      });

      return jsonError(
        res,
        status,
        message,
        'upstream_error'
      );
    } finally {
      clearTimeout(timeout);

      req.removeListener('aborted', onRequestAborted);
      res.removeListener('close', onResponseClose);

      if (
        upstreamStream &&
        !upstreamStream.destroyed &&
        !upstreamStream.readableEnded
      ) {
        upstreamStream.destroy();
      }
    }
  }
);

// ============================================================
// ERROR HANDLING
// ============================================================

app.use((err, req, res, next) => {
  if (res.headersSent) {
    return next(err);
  }

  if (err?.type === 'entity.too.large') {
    return jsonError(
      res,
      413,
      `Request body exceeds the configured limit (${MAX_BODY_BYTES}).`,
      'invalid_request_error'
    );
  }

  if (
    err instanceof SyntaxError &&
    Object.prototype.hasOwnProperty.call(err, 'body')
  ) {
    return jsonError(
      res,
      400,
      'Malformed JSON request body.',
      'invalid_request_error'
    );
  }

  console.error(
    'Unhandled server error:',
    err?.message || 'Unknown error'
  );

  return jsonError(
    res,
    500,
    'Internal server error.',
    'server_error'
  );
});

// ============================================================
// START SERVER
// ============================================================

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log('========================================');
  console.log('NVIDIA NIM JanitorAI Proxy');
  console.log('========================================');
  console.log(`Port: ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`NVIDIA endpoint: ${NVIDIA_BASE_URL}`);
  console.log(`Model profiles: ${Object.keys(MODEL_PROFILES).length}`);
  console.log(`Reasoning stripping: ${STRIP_REASONING}`);
  console.log(`Unknown models: ${ALLOW_UNKNOWN_MODELS}`);
  console.log(`Max concurrent generations: ${MAX_CONCURRENT_REQUESTS}`);
  console.log(`Default max tokens: ${DEFAULT_MAX_TOKENS}`);
  console.log(`Timeout: ${REQUEST_TIMEOUT_MS}ms`);
  console.log(`Proxy authentication: ${PROXY_API_KEY ? 'enabled' : 'DISABLED'}`);
  console.log('========================================');
});

// Long generations are controlled by REQUEST_TIMEOUT_MS.
server.requestTimeout = 0;
server.timeout = 0;
server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;

// ============================================================
// GRACEFUL SHUTDOWN
// ============================================================

function shutdown(signal) {
  console.log(`${signal} received. Closing server.`);

  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });

  const forceExit = setTimeout(() => {
    console.error('Graceful shutdown timed out.');
    process.exit(1);
  }, 10000);

  forceExit.unref();
}

process.once('SIGTERM', () => shutdown('SIGTERM'));
process.once('SIGINT', () => shutdown('SIGINT'));
