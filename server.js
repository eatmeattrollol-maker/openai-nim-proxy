
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

const MAX_BODY_BYTES = process.env.MAX_BODY_BYTES || '5mb';

const MAX_CONCURRENT_REQUESTS = Number(
  process.env.MAX_CONCURRENT_REQUESTS || 1
);

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 8192
);

const STRIP_REASONING =
  (process.env.STRIP_REASONING || 'true').toLowerCase() === 'true';

const ALLOW_UNKNOWN_MODELS =
  (process.env.ALLOW_UNKNOWN_MODELS || 'false').toLowerCase() === 'true';

const PROXY_API_KEY = process.env.PROXY_API_KEY || '';

const ALLOWED_ORIGINS = (
  process.env.CORS_ORIGINS || 'https://janitorai.com'
)
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

if (!NVIDIA_API_KEY) {
  console.error('FATAL: NVIDIA_API_KEY is missing.');
  process.exit(1);
}

if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) {
  throw new Error('PORT must be a valid TCP port.');
}

if (
  !Number.isFinite(REQUEST_TIMEOUT_MS) ||
  REQUEST_TIMEOUT_MS < 1000
) {
  throw new Error('REQUEST_TIMEOUT_MS must be at least 1000.');
}

if (
  !Number.isInteger(MAX_CONCURRENT_REQUESTS) ||
  MAX_CONCURRENT_REQUESTS < 1
) {
  throw new Error('MAX_CONCURRENT_REQUESTS must be at least 1.');
}

if (
  !Number.isInteger(DEFAULT_MAX_TOKENS) ||
  DEFAULT_MAX_TOKENS < 1
) {
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
// The left-hand ID is what JanitorAI selects.
// "upstream" is the actual model ID sent to NVIDIA.
//
// Reasoning aliases are intentional. They let you switch
// reasoning levels in JanitorAI without editing this server.
//
// Every model has its own reasoning adapter because NVIDIA
// models do not all accept the same reasoning parameters.
// ============================================================

const MODEL_PROFILES = {};

function addModel(id, label, upstream, settings = {}) {
  MODEL_PROFILES[id] = {
    id,
    label,
    upstream,
    ...settings
  };
}

// ------------------------------------------------------------
// KIMI K3
// NVIDIA supports: low, high, max
// We map Janitor's medium/high profiles to higher effort.
// ------------------------------------------------------------

addModel(
  'moonshotai/kimi-k3',
  'Kimi K3',
  'moonshotai/kimi-k3',
  { adapter: 'kimi' }
);

addModel(
  'moonshotai/kimi-k3-low',
  'Kimi K3 - Low Reasoning',
  'moonshotai/kimi-k3',
  { adapter: 'kimi', effort: 'low' }
);

addModel(
  'moonshotai/kimi-k3-medium',
  'Kimi K3 - Medium Reasoning',
  'moonshotai/kimi-k3',
  { adapter: 'kimi', effort: 'high' }
);

addModel(
  'moonshotai/kimi-k3-high',
  'Kimi K3 - High Reasoning',
  'moonshotai/kimi-k3',
  { adapter: 'kimi', effort: 'max' }
);

// ------------------------------------------------------------
// MISTRAL LARGE 4
// NVIDIA supports: low, medium, high
// This is a particularly useful general-purpose RP option.
// ------------------------------------------------------------

addModel(
  'mistralai/mistral-large-4-1t-a52b',
  'Mistral Large 4',
  'mistralai/mistral-large-4-1t-a52b',
  { adapter: 'mistral-large-4' }
);

for (const effort of ['low', 'medium', 'high']) {
  addModel(
    `mistralai/mistral-large-4-1t-a52b-${effort}`,
    `Mistral Large 4 - ${effort[0].toUpperCase()}${effort.slice(1)} Reasoning`,
    'mistralai/mistral-large-4-1t-a52b',
    {
      adapter: 'mistral-large-4',
      effort
    }
  );
}

// ------------------------------------------------------------
// MISTRAL NEMOTRON
// No invented reasoning controls. Preserve client parameters.
// ------------------------------------------------------------

addModel(
  'mistralai/mistral-nemotron',
  'Mistral Nemotron',
  'mistralai/mistral-nemotron',
  { adapter: 'generic' }
);

// ------------------------------------------------------------
// NEMOTRON 3 ULTRA
// NVIDIA supports: none, medium, high
// Also supports reasoning_budget.
// ------------------------------------------------------------

const ULTRA_ID = 'nvidia/nemotron-3-ultra-550b-a55b';

addModel(
  ULTRA_ID,
  'Nemotron 3 Ultra - Default',
  ULTRA_ID,
  { adapter: 'nemotron-ultra' }
);

addModel(
  `${ULTRA_ID}-no-thinking`,
  'Nemotron 3 Ultra - No Thinking',
  ULTRA_ID,
  {
    adapter: 'nemotron-ultra',
    effort: 'none'
  }
);

addModel(
  `${ULTRA_ID}-thinking`,
  'Nemotron 3 Ultra - Thinking',
  ULTRA_ID,
  {
    adapter: 'nemotron-ultra',
    effort: 'medium',
    reasoningBudget: 4096
  }
);

addModel(
  `${ULTRA_ID}-deep-thinking`,
  'Nemotron 3 Ultra - Deep Thinking',
  ULTRA_ID,
  {
    adapter: 'nemotron-ultra',
    effort: 'high',
    reasoningBudget: 8192
  }
);

for (const [suffix, effort, budget] of [
  ['low', 'medium', 2048],
  ['medium', 'medium', 4096],
  ['high', 'high', 8192]
]) {
  addModel(
    `${ULTRA_ID}-${suffix}`,
    `Nemotron 3 Ultra - ${suffix[0].toUpperCase()}${suffix.slice(1)} Reasoning`,
    ULTRA_ID,
    {
      adapter: 'nemotron-ultra',
      effort,
      reasoningBudget: budget
    }
  );
}

// ------------------------------------------------------------
// NEMOTRON 3.5 SUPER VL
// NVIDIA supports: none, low, high
// ------------------------------------------------------------

const NEMOTRON_VL_ID =
  'nvidia/nemotron-3.5-super-vl-120b-a12b';

addModel(
  NEMOTRON_VL_ID,
  'Nemotron 3.5 Super VL',
  NEMOTRON_VL_ID,
  { adapter: 'nemotron-vl' }
);

addModel(
  `${NEMOTRON_VL_ID}-no-thinking`,
  'Nemotron 3.5 Super VL - No Thinking',
  NEMOTRON_VL_ID,
  {
    adapter: 'nemotron-vl',
    effort: 'none'
  }
);

for (const [suffix, effort] of [
  ['low', 'low'],
  ['medium', 'high'],
  ['high', 'high']
]) {
  addModel(
    `${NEMOTRON_VL_ID}-${suffix}`,
    `Nemotron 3.5 Super VL - ${suffix[0].toUpperCase()}${suffix.slice(1)} Reasoning`,
    NEMOTRON_VL_ID,
    {
      adapter: 'nemotron-vl',
      effort
    }
  );
}

// ------------------------------------------------------------
// GLM 5.3 FLASH
// NVIDIA documents low, high, max reasoning effort.
// clear_thinking is explicitly enabled for chat-style use.
// ------------------------------------------------------------

const GLM_ID = 'z-ai/glm-5.3-flash';

addModel(
  GLM_ID,
  'GLM 5.3 Flash',
  GLM_ID,
  { adapter: 'glm' }
);

for (const [suffix, effort] of [
  ['low', 'low'],
  ['medium', 'high'],
  ['high', 'max']
]) {
  addModel(
    `${GLM_ID}-${suffix}`,
    `GLM 5.3 Flash - ${suffix[0].toUpperCase()}${suffix.slice(1)} Reasoning`,
    GLM_ID,
    {
      adapter: 'glm',
      effort
    }
  );
}

addModel(
  `${GLM_ID}-thinking`,
  'GLM 5.3 Flash - Thinking',
  GLM_ID,
  {
    adapter: 'glm',
    effort: 'max'
  }
);

addModel(
  `${GLM_ID}-no-thinking`,
  'GLM 5.3 Flash - No Thinking',
  GLM_ID,
  {
    adapter: 'glm',
    effort: 'low'
  }
);

// ------------------------------------------------------------
// DEEPSEEK V4.1 FLASH
// NVIDIA's model supports continuously controlled reasoning.
// Numeric levels are mapped to its 1-100 reasoning scale.
// ------------------------------------------------------------

const DEEPSEEK_ID = 'deepseek-ai/deepseek-v4.1-flash';

addModel(
  DEEPSEEK_ID,
  'DeepSeek V4.1 Flash',
  DEEPSEEK_ID,
  { adapter: 'deepseek' }
);

for (const [suffix, effort] of [
  ['low', 25],
  ['medium', 50],
  ['high', 100]
]) {
  addModel(
    `${DEEPSEEK_ID}-${suffix}`,
    `DeepSeek V4.1 Flash - ${suffix[0].toUpperCase()}${suffix.slice(1)} Reasoning`,
    DEEPSEEK_ID,
    {
      adapter: 'deepseek',
      effort
    }
  );
}

// ------------------------------------------------------------
// STEPFUN STEP 5 PREVIEW
// Reasoning is enabled by the checkpoint.
// NVIDIA says setting enable_thinking=false does not disable it.
// Do not invent low/medium/high controls for this model.
// ------------------------------------------------------------

addModel(
  'stepfun-ai/step-5-preview',
  'Step 5 Preview',
  'stepfun-ai/step-5-preview',
  { adapter: 'step' }
);

// ------------------------------------------------------------
// MINIMAX M3.1 FLASH
// NVIDIA supports low, medium, high, xhigh, max.
// ------------------------------------------------------------

const MINIMAX_ID = 'minimaxai/minimax-m3.1-flash';

addModel(
  MINIMAX_ID,
  'MiniMax M3.1 Flash',
  MINIMAX_ID,
  { adapter: 'minimax' }
);

for (const effort of ['low', 'medium', 'high', 'xhigh', 'max']) {
  addModel(
    `${MINIMAX_ID}-${effort}`,
    `MiniMax M3.1 Flash - ${effort[0].toUpperCase()}${effort.slice(1)} Reasoning`,
    MINIMAX_ID,
    {
      adapter: 'minimax',
      effort
    }
  );
}

// ------------------------------------------------------------
// POOLSIDE LAGUNA XS 2.1
// Correct NVIDIA model ID spelling.
// It is primarily a coding/agentic model, not an RP-first pick.
// ------------------------------------------------------------

addModel(
  'poolside/laguna-xs-2.1',
  'Laguna XS 2.1',
  'poolside/laguna-xs-2.1',
  { adapter: 'generic' }
);

// ------------------------------------------------------------
// GPT OSS 20B
// ------------------------------------------------------------

addModel(
  'openai/gpt-oss-20b',
  'GPT OSS 20B',
  'openai/gpt-oss-20b',
  { adapter: 'generic' }
);

// ------------------------------------------------------------
// GEMMA 4 31B IT
// Supports thinking configuration through chat_template_kwargs.
// ------------------------------------------------------------

const GEMMA_ID = 'google/gemma-4-31b-it';

addModel(
  GEMMA_ID,
  'Gemma 4 31B IT',
  GEMMA_ID,
  { adapter: 'gemma' }
);

addModel(
  `${GEMMA_ID}-thinking`,
  'Gemma 4 31B IT - Thinking',
  GEMMA_ID,
  {
    adapter: 'gemma',
    thinking: true
  }
);

addModel(
  `${GEMMA_ID}-no-thinking`,
  'Gemma 4 31B IT - No Thinking',
  GEMMA_ID,
  {
    adapter: 'gemma',
    thinking: false
  }
);

// ------------------------------------------------------------
// META MUSE GLIMMER 30B
// NVIDIA supports none, minimal, low, medium, high, max.
// ------------------------------------------------------------

const MUSE_ID = 'meta/muse-glimmer-30b';

addModel(
  MUSE_ID,
  'Muse Glimmer 30B',
  MUSE_ID,
  { adapter: 'muse' }
);

for (const effort of ['low', 'medium', 'high', 'max']) {
  addModel(
    `${MUSE_ID}-${effort}`,
    `Muse Glimmer 30B - ${effort[0].toUpperCase()}${effort.slice(1)} Reasoning`,
    MUSE_ID,
    {
      adapter: 'muse',
      effort
    }
  );
}

// ============================================================
// REQUEST LOGGING
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
  const origin = req.get('origin');

  const allowed =
    !origin ||
    ALLOWED_ORIGINS.includes('*') ||
    ALLOWED_ORIGINS.includes(origin);

  if (origin && allowed) {
    res.setHeader('Access-Control-Allow-Origin', origin);
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
}

// ============================================================
// CONCURRENCY CONTROL
//
// Only chat completions consume a generation slot.
// Health and model-list requests remain available.
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

function jsonError(
  res,
  status,
  message,
  type = 'proxy_error'
) {
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
  if (data && typeof data === 'object') {
    const message =
      data.error?.message ||
      data.message ||
      data.detail;

    if (typeof message === 'string') {
      return message.slice(0, 3000);
    }
  }

  if (typeof data === 'string') {
    return data.slice(0, 3000);
  }

  return fallback;
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
// APPLY MODEL-SPECIFIC REASONING SETTINGS
//
// This is the important part of the profile system.
// JanitorAI selects an alias; this function translates it.
// ============================================================

function applyProfile(requestBody, profile) {
  const body = { ...requestBody };

  body.model = profile.upstream;

  const template = {
    ...(body.chat_template_kwargs || {})
  };

  switch (profile.adapter) {
    case 'kimi': {
      // Kimi's NVIDIA endpoint does not expose the usual
      // top_p / presence_penalty / frequency_penalty controls.
      delete body.top_p;
      delete body.presence_penalty;
      delete body.frequency_penalty;

      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      break;
    }

    case 'mistral-large-4': {
      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      // This recipe does not honor enable_thinking.
      delete template.enable_thinking;

      if (Object.keys(template).length > 0) {
        body.chat_template_kwargs = template;
      } else {
        delete body.chat_template_kwargs;
      }

      break;
    }

    case 'nemotron-ultra': {
      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      if (profile.reasoningBudget !== undefined) {
        body.reasoning_budget = profile.reasoningBudget;
      }

      // Remove the old, incorrect parameter name.
      delete body.thinking_token_budget;

      delete template.enable_thinking;

      if (Object.keys(template).length > 0) {
        body.chat_template_kwargs = template;
      } else {
        delete body.chat_template_kwargs;
      }

      break;
    }

    case 'nemotron-vl': {
      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      delete template.enable_thinking;

      if (Object.keys(template).length > 0) {
        body.chat_template_kwargs = template;
      } else {
        delete body.chat_template_kwargs;
      }

      break;
    }

    case 'glm': {
      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      // Explicitly request normal chat output rather than
      // exposing the model's thinking trace as the answer.
      template.clear_thinking = true;

      body.chat_template_kwargs = template;
      break;
    }

    case 'deepseek': {
      if (profile.effort !== undefined) {
        body.reasoning_effort = profile.effort;
      }

      break;
    }

    case 'minimax': {
      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      break;
    }

    case 'gemma': {
      if (profile.thinking !== undefined) {
        template.enable_thinking = profile.thinking;
      }

      body.chat_template_kwargs = template;
      break;
    }

    case 'muse': {
      if (profile.effort) {
        body.reasoning_effort = profile.effort;
      }

      break;
    }

    case 'step': {
      // Step 5 Preview's NVIDIA recipe keeps reasoning enabled.
      // Do not force enable_thinking=false.
      break;
    }

    default: {
      // Generic models keep the client's supplied settings.
      break;
    }
  }

  return body;
}

// ============================================================
// REQUEST VALIDATION
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

  if (
    !Array.isArray(body.messages) ||
    body.messages.length === 0
  ) {
    return 'messages must be a non-empty array.';
  }

  for (const message of body.messages) {
    if (
      !message ||
      typeof message !== 'object' ||
      Array.isArray(message) ||
      typeof message.role !== 'string'
    ) {
      return 'Each message must be an object with a role.';
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
// REASONING-FIELD FILTER
//
// Removes separate reasoning fields while preserving normal
// assistant content, tool calls, and the conversation itself.
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
//
// Parses full SSE events instead of modifying arbitrary network
// chunks. This avoids corrupting JSON when chunks are split.
// ============================================================

function extractSseEvents(buffer) {
  const events = [];
  let remaining = buffer;
  let match;

  while (
    (match = /\r?\n\r?\n/.exec(remaining)) !== null
  ) {
    events.push(remaining.slice(0, match.index));

    remaining = remaining.slice(
      match.index + match[0].length
    );
  }

  return {
    events,
    remaining
  };
}

function transformSseEvent(eventText) {
  if (!eventText) {
    return '';
  }

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
      // Never corrupt an event we cannot parse.
      outputData = data;
    }
  }

  return `${prefix}data: ${outputData}\n\n`;
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
//
// JanitorAI sees aliases as separate selectable models.
// The proxy resolves each alias to its real NVIDIA model.
// ============================================================

app.get('/v1/models', (req, res) => {
  const created = Math.floor(Date.now() / 1000);

  const data = Object.entries(MODEL_PROFILES).map(
    ([id, profile]) => ({
      id,
      object: 'model',
      created,
      owned_by: 'nvidia-nim-proxy',
      name: profile.label
    })
  );

  if (!data.some((model) => model.id === DEFAULT_MODEL)) {
    data.unshift({
      id: DEFAULT_MODEL,
      object: 'model',
      created,
      owned_by: 'nvidia-nim-proxy',
      name: DEFAULT_MODEL
    });
  }

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

    // Apply the selected profile without rewriting messages.
    const body = applyProfile(requestBody, profile);

    // NVIDIA's chat API uses max_tokens. Translate the newer
    // OpenAI max_completion_tokens field when necessary.
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

      if (
        upstreamStream &&
        !upstreamStream.destroyed
      ) {
        upstreamStream.destroy();
      }
    };

    const onRequestAborted = () => {
      cancelUpstream();
    };

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

      console.warn('NVIDIA request timed out:', {
        model: requestedModel,
        timeoutMs: REQUEST_TIMEOUT_MS
      });

      const error = Object.assign(
        new Error('NVIDIA request timed out.'),
        { code: 'UPSTREAM_TIMEOUT' }
      );

      controller.abort(error);

      if (
        upstreamStream &&
        !upstreamStream.destroyed
      ) {
        upstreamStream.destroy(error);
      }
    }, REQUEST_TIMEOUT_MS);

    try {
      const upstream = await axios.post(
        `${NVIDIA_BASE_URL}/chat/completions`,
        body,
        {
          headers: upstreamHeaders(streaming),
          responseType: streaming ? 'stream' : 'json',

          // Our own timer is the overall deadline.
          // Do not use Axios inactivity timeout for long streams.
          timeout: 0,

          signal: controller.signal,
          validateStatus: () => true,
          maxRedirects: 0,

          // Avoid unexpectedly buffering large JSON responses.
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

      // ------------------------------------------------------
      // UPSTREAM ERRORS
      // ------------------------------------------------------

      if (
        upstream.status < 200 ||
        upstream.status >= 300
      ) {
        let errorData = upstream.data;

        if (streaming && upstream.data) {
          upstreamStream = upstream.data;

          const chunks = [];
          let totalBytes = 0;
          const maxErrorBytes = 64 * 1024;

          for await (const chunk of upstreamStream) {
            const part = Buffer.from(chunk);
            const remaining = maxErrorBytes - totalBytes;

            if (remaining <= 0) break;

            const accepted = part.subarray(0, remaining);

            chunks.push(accepted);
            totalBytes += accepted.length;

            if (totalBytes >= maxErrorBytes) break;
          }

          errorData = Buffer.concat(chunks).toString('utf8');
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
      // STREAMING
      // ------------------------------------------------------

      if (streaming) {
        upstreamStream = upstream.data;

        res.status(200);

        res.setHeader(
          'Content-Type',
          'text/event-stream; charset=utf-8'
        );

        res.setHeader(
          'Cache-Control',
          'no-cache, no-transform'
        );

        res.setHeader('X-Accel-Buffering', 'no');
        res.setHeader('Connection', 'keep-alive');

        res.flushHeaders?.();

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

              if (
                /(?:^|\n)data:\s*\[DONE\]\s*(?:\n|$)/.test(output)
              ) {
                receivedDone = true;
              }

              const writable = await writeChunk(res, output);

              if (!writable) return;

              if (receivedDone) break;
            }

            if (receivedDone) {
              break;
            }
          }

          if (clientDisconnected || res.destroyed) {
            return;
          }

          // Only flush the remaining decoder data on a clean EOF.
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

                if (
                  /(?:^|\n)data:\s*\[DONE\]\s*(?:\n|$)/.test(output)
                ) {
                  receivedDone = true;
                }
              }
            }
          }

          if (clientDisconnected || res.destroyed) {
            return;
          }

          // A clean upstream EOF without [DONE] is still a
          // completed HTTP stream. Supply the OpenAI terminator.
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

            const errorEvent =
              `data: ${JSON.stringify({
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
      // NON-STREAMING
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
  console.log(`Proxy authentication: ${PROXY_API_KEY ? 'enabled' : 'disabled'}`);
  console.log('========================================');
});

// Let the upstream generation timer control long requests.
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
