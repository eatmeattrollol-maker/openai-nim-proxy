
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
  'mistralai/mistral-large-4-1t-a52b-thinking';

const REQUEST_TIMEOUT_MS = Number(
  process.env.REQUEST_TIMEOUT_MS || 600000
);

const MAX_BODY_BYTES =
  process.env.MAX_BODY_BYTES || '20mb';

const MAX_CONCURRENT_REQUESTS = Number(
  process.env.MAX_CONCURRENT_REQUESTS || 1
);

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 16384
);

const STRIP_REASONING =
  String(process.env.STRIP_REASONING || 'true').toLowerCase() === 'true';

const RP_CONTINUITY_MODE =
  String(process.env.RP_CONTINUITY_MODE || 'true').toLowerCase() === 'true';

const ALLOW_UNKNOWN_MODELS =
  String(process.env.ALLOW_UNKNOWN_MODELS || 'false').toLowerCase() === 'true';

const PROXY_API_KEY = process.env.PROXY_API_KEY || '';

const ALLOWED_ORIGINS = (
  process.env.CORS_ORIGINS || 'https://janitorai.com'
)
  .split(',')
  .map(value => value.trim().replace(/\/+$/, ''))
  .filter(Boolean);

// ============================================================
// ROLEPLAY CONTINUITY
//
// Instruction-only. No persistent memory is created.
// No chat is identified by character name.
// No retry creates or saves conversation history.
// ============================================================

const RP_CONTINUITY_PROMPT =
  process.env.RP_CONTINUITY_PROMPT || [
    'ROLEPLAY CONTINUITY INSTRUCTIONS:',
    'Treat the supplied conversation as one chronological story, not a list of independent prompts.',
    'All earlier user messages and assistant replies are established history and events that have already happened.',
    'Earlier user messages that already have assistant replies are not new requests. Do not answer them again or replay them.',
    'Continue from the current end of the supplied transcript.',
    'Maintain established facts, chronology, character identities, personalities, knowledge, motives, relationships, locations, injuries, possessions, promises, and consequences.',
    'Characters must retain only information they have actually learned.',
    'Do not reset scenes, repeat completed events, resurrect resolved conflicts, or contradict established events.',
    'Treat character actions and dialogue from earlier assistant responses as events that have already happened.',
    'Respond to the newest user contribution in the context of the entire supplied history.',
    'Allow NPCs and the surrounding world to behave naturally and independently rather than existing only to answer the player.',
    'Respect the existing character definitions, scenario instructions, and roleplay boundaries.',
    'Do not take control of the user character or invent their thoughts, dialogue, decisions, or actions unless the existing instructions explicitly permit it.'
  ].join('\n');

// ============================================================
// STARTUP VALIDATION
// ============================================================

if (!NVIDIA_API_KEY) {
  throw new Error('NVIDIA_API_KEY is missing.');
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

if (
  PROXY_API_KEY &&
  Buffer.byteLength(PROXY_API_KEY, 'utf8') < 32
) {
  throw new Error('PROXY_API_KEY must contain at least 32 bytes.');
}

if (!NVIDIA_BASE_URL.startsWith('https://')) {
  throw new Error('NVIDIA_BASE_URL must use HTTPS.');
}

app.disable('x-powered-by');
app.set('trust proxy', 1);

// ============================================================
// MODEL PROFILES
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
// High reasoning and no-reasoning profiles.
// Automatic fallback is available for HTTP 400.
// ------------------------------------------------------------

const MISTRAL4 =
  'mistralai/mistral-large-4-1t-a52b';

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
  'Mistral Large 4 - High Reasoning',
  MISTRAL4,
  {
    adapter: 'mistral4',
    effort: 'high'
  }
);

addModel(
  `${MISTRAL4}-no-thinking`,
  'Mistral Large 4 - No Reasoning',
  MISTRAL4,
  {
    adapter: 'mistral4',
    effort: 'none'
  }
);

addModel(
  `${MISTRAL4}-high`,
  'Mistral Large 4 - High Reasoning',
  MISTRAL4,
  {
    adapter: 'mistral4',
    effort: 'high'
  }
);

// ------------------------------------------------------------
// GLM 5.3 FLASH
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
// MINIMAX M3.1 FLASH
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

for (const effort of [
  'low',
  'medium',
  'high',
  'xhigh',
  'max'
]) {
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
// NEMOTRON 3.5 SUPER VL
// ------------------------------------------------------------

const NEMOTRON_VL =
  'nvidia/nemotron-3.5-super-vl-120b-a12b';

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
// STEP 5 PREVIEW
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
// LAGUNA XS 2.1
// ------------------------------------------------------------

addModel(
  'poolside/laguna-xs-2.1',
  'Laguna XS 2.1',
  'poolside/laguna-xs-2.1'
);

// ------------------------------------------------------------
// NEMOTRON 3.5 BOLT E4B
// ------------------------------------------------------------

addModel(
  'nvidia/nemotron-3.5-bolt-e4b',
  'Nemotron 3.5 Bolt E4B',
  'nvidia/nemotron-3.5-bolt-e4b'
);

// ============================================================
// DEFAULT MODEL VALIDATION
// ============================================================

if (
  !Object.prototype.hasOwnProperty.call(
    MODEL_PROFILES,
    DEFAULT_MODEL
  )
) {
  throw new Error(
    `DEFAULT_MODEL "${DEFAULT_MODEL}" is not registered.`
  );
}

// ============================================================
// REQUEST LOGGING
// Never log prompts, message content, or API keys.
// ============================================================

app.use((req, res, next) => {
  const started = Date.now();

  req.proxyRequestId = crypto.randomUUID();

  res.setHeader(
    'X-Proxy-Request-Id',
    req.proxyRequestId
  );

  res.once('finish', () => {
    console.log(
      `${req.method} ${req.path} -> ${res.statusCode} ` +
      `(${Date.now() - started}ms) [${req.proxyRequestId}]`
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
        'Authorization, Content-Type, X-Conversation-Id, X-Chat-Id'
    );

    res.setHeader(
      'Access-Control-Expose-Headers',
      'X-Proxy-Request-Id'
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
// JSON BODY PARSER
// ============================================================

app.use(express.json({
  limit: MAX_BODY_BYTES,
  strict: true
}));

// ============================================================
// OPTIONAL PROXY AUTHENTICATION
// ============================================================

if (PROXY_API_KEY) {
  const expectedKey = Buffer.from(PROXY_API_KEY, 'utf8');

  app.use('/v1', (req, res, next) => {
    const match = (req.get('authorization') || '')
      .match(/^Bearer\s+(.+)$/i);

    const suppliedKey = Buffer.from(
      match ? match[1] : '',
      'utf8'
    );

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
    'WARNING: PROXY_API_KEY is unset; public /v1 endpoints have no proxy authentication.'
  );
}

// ============================================================
// CONCURRENCY
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

    activeRequests = Math.max(
      0,
      activeRequests - 1
    );
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

function parseJsonOrText(value) {
  if (typeof value !== 'string') return value;

  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function upstreamErrorMessage(
  data,
  fallback = 'Unknown NVIDIA error.'
) {
  const parsed = parseJsonOrText(data);

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

function safeErrorForLog(data) {
  let value = data;

  if (Buffer.isBuffer(value)) {
    value = value.toString('utf8');
  }

  if (typeof value === 'string') {
    return value.slice(0, 4000);
  }

  try {
    return JSON.stringify(value).slice(0, 4000);
  } catch {
    return '[unprintable upstream error]';
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

  for (const [index, message] of body.messages.entries()) {
    if (
      !message ||
      typeof message !== 'object' ||
      Array.isArray(message) ||
      typeof message.role !== 'string'
    ) {
      return `Message ${index + 1} must be an object with a string role.`;
    }

    if (
      message.content !== undefined &&
      message.content !== null &&
      typeof message.content !== 'string' &&
      !Array.isArray(message.content)
    ) {
      return `Message ${index + 1} content must be a string, null, or content-part array.`;
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
    (
      body.temperature < 0 ||
      body.temperature > 2
    )
  ) {
    return 'temperature must be between 0 and 2.';
  }

  if (
    body.top_p !== undefined &&
    (
      body.top_p < 0 ||
      body.top_p > 1
    )
  ) {
    return 'top_p must be between 0 and 1.';
  }

  for (const key of [
    'frequency_penalty',
    'presence_penalty'
  ]) {
    if (
      body[key] !== undefined &&
      (
        body[key] < -2 ||
        body[key] > 2
      )
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

  return null;
}

// ============================================================
// MISTRAL MESSAGE NORMALIZATION
// ============================================================

const MISTRAL_ROLES = new Set([
  'system',
  'user',
  'assistant',
  'tool'
]);

function normalizeMistralMessages(messages) {
  const normalized = [];

  messages.forEach((message, index) => {
    let role = message.role;

    if (
      role === 'developer' ||
      role === 'context'
    ) {
      role = 'system';
    }

    if (!MISTRAL_ROLES.has(role)) {
      throw new Error(
        `Mistral message ${index + 1} has unsupported role "${role}".`
      );
    }

    const content = message.content;

    // Remove only whitespace-only user/assistant messages.
    // Do not discard actual narrative turns.
    if (
      (role === 'user' || role === 'assistant') &&
      typeof content === 'string' &&
      content.trim().length === 0
    ) {
      return;
    }

    const output = { role };

    if (
      Object.prototype.hasOwnProperty.call(message, 'content')
    ) {
      output.content = content;
    }

    if (
      typeof message.name === 'string' &&
      message.name.length > 0
    ) {
      output.name = message.name;
    }

    if (role === 'assistant') {
      if (
        typeof message.reasoning_content === 'string' &&
        message.reasoning_content.length > 0
      ) {
        output.reasoning_content = message.reasoning_content;
      }

      if (Array.isArray(message.tool_calls)) {
        output.tool_calls = message.tool_calls;
      }
    }

    if (role === 'tool') {
      if (
        typeof message.tool_call_id !== 'string' ||
        !message.tool_call_id
      ) {
        throw new Error(
          `Mistral tool message ${index + 1} is missing tool_call_id.`
        );
      }

      output.tool_call_id = message.tool_call_id;
    }

    normalized.push(output);
  });

  if (normalized.length === 0) {
    throw new Error(
      'Mistral request contains no usable messages.'
    );
  }

  return normalized;
}

// ============================================================
// CONTINUITY INSTRUCTION INJECTION
//
// The instruction is appended to the existing first system
// message instead of creating another system message.
// ============================================================

function addRpContinuityInstruction(messages) {
  if (
    !RP_CONTINUITY_MODE ||
    !RP_CONTINUITY_PROMPT.trim()
  ) {
    return messages;
  }

  const marker = 'ROLEPLAY CONTINUITY INSTRUCTIONS:';
  const result = messages.slice();

  const firstSystemIndex = result.findIndex(
    message => message.role === 'system'
  );

  if (firstSystemIndex === -1) {
    result.unshift({
      role: 'system',
      content: RP_CONTINUITY_PROMPT
    });

    return result;
  }

  const firstSystem = {
    ...result[firstSystemIndex]
  };

  const currentContent = firstSystem.content;

  if (
    typeof currentContent === 'string' &&
    currentContent.includes(marker)
  ) {
    return result;
  }

  if (
    typeof currentContent === 'string' ||
    currentContent == null
  ) {
    firstSystem.content = [
      currentContent || '',
      RP_CONTINUITY_PROMPT
    ]
      .filter(Boolean)
      .join('\n\n');
  } else if (Array.isArray(currentContent)) {
    firstSystem.content = [
      ...currentContent,
      {
        type: 'text',
        text: RP_CONTINUITY_PROMPT
      }
    ];
  } else {
    return result;
  }

  result[firstSystemIndex] = firstSystem;

  return result;
}

// ============================================================
// SAFE MESSAGE DIAGNOSTICS
// Never log prompts or message contents.
// ============================================================

function describeContent(content) {
  if (typeof content === 'string') {
    return {
      type: 'string',
      characters: content.length
    };
  }

  if (content === null) {
    return {
      type: 'null'
    };
  }

  if (Array.isArray(content)) {
    return {
      type: 'array',
      parts: content.length,
      partTypes: content.slice(0, 20).map(part => {
        if (!part || typeof part !== 'object') {
          return typeof part;
        }

        return part.type || 'unknown';
      })
    };
  }

  return {
    type: typeof content
  };
}

function summarizeMessages(messages) {
  return messages.slice(-40).map((message, offset) => ({
    index: Math.max(0, messages.length - 40) + offset,
    role: message.role,
    content: describeContent(message.content),
    hasReasoningContent:
      typeof message.reasoning_content === 'string' &&
      message.reasoning_content.length > 0,
    toolCallCount: Array.isArray(message.tool_calls)
      ? message.tool_calls.length
      : 0
  }));
}

function countWhitespaceOnlyMessages(messages) {
  return messages.filter(message =>
    message &&
    (message.role === 'user' || message.role === 'assistant') &&
    typeof message.content === 'string' &&
    message.content.length > 0 &&
    message.content.trim().length === 0
  ).length;
}

function countNullReasoningFields(messages) {
  return messages.filter(message =>
    message &&
    message.role === 'assistant' &&
    Object.prototype.hasOwnProperty.call(
      message,
      'reasoning_content'
    ) &&
    message.reasoning_content === null
  ).length;
}

// ============================================================
// REQUEST ALLOWLISTS
// ============================================================

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

const COMMON_ALLOWED_FIELDS = new Set([
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
  'user'
]);

function copyAllowedFields(source, allowedFields) {
  const output = {};

  for (const [key, value] of Object.entries(source)) {
    if (allowedFields.has(key)) {
      output[key] = value;
    }
  }

  return output;
}

// ============================================================
// MISTRAL FALLBACK REQUEST
//
// First: core parameters + high reasoning.
// Second: core parameters + no explicit reasoning,
// with reasoning_effort=none.
//
// Both attempts retain the same conversation history.
// ============================================================

function buildMistralFallbackBody(
  body,
  reasoningEffort
) {
  const fallback = {
    model: body.model,
    messages: body.messages,
    max_tokens: body.max_tokens,
    stream: body.stream === true
  };

  if (reasoningEffort) {
    fallback.reasoning_effort = reasoningEffort;
  }

  return fallback;
}

function findOneCharacterMessages(messages) {
  return messages
    .map((message, index) => ({
      message,
      index
    }))
    .filter(item =>
      typeof item.message.content === 'string' &&
      item.message.content.length === 1
    )
    .map(item => ({
      index: item.index,
      role: item.message.role,
      isWhitespace: /^\s$/.test(item.message.content),
      codePoint: item.message.content.codePointAt(0)
    }));
}

// ============================================================
// MODEL-SPECIFIC REQUEST ADAPTERS
// ============================================================

function applyProfile(requestBody, profile) {
  const body = {
    ...requestBody,
    model: profile.upstream
  };

  const template =
    body.chat_template_kwargs &&
    typeof body.chat_template_kwargs === 'object' &&
    !Array.isArray(body.chat_template_kwargs)
      ? { ...body.chat_template_kwargs }
      : {};

  switch (profile.adapter) {
    case 'mistral4': {
      body.reasoning_effort =
        profile.effort || 'high';

      delete body.chat_template_kwargs;

      const filtered = copyAllowedFields(
        body,
        MISTRAL_ALLOWED_FIELDS
      );

      filtered.messages = addRpContinuityInstruction(
        normalizeMistralMessages(body.messages)
      );

      // Avoid sending both temperature and top_p.
      // Preserve JanitorAI temperature if supplied.
      if (
        filtered.temperature !== undefined &&
        filtered.top_p !== undefined
      ) {
        delete filtered.top_p;
      }

      return filtered;
    }

    case 'glm': {
      body.reasoning_effort = profile.effort;
      template.clear_thinking = true;
      body.chat_template_kwargs = template;
      break;
    }

    case 'minimax': {
      template.reasoning_effort = profile.effort;
      body.chat_template_kwargs = template;
      delete body.reasoning_effort;
      break;
    }

    case 'nemotron-vl': {
      body.reasoning_effort = profile.effort;
      delete template.enable_thinking;
      body.chat_template_kwargs = template;
      break;
    }

    case 'step':
      break;

    default:
      break;
  }

  if (
    body.chat_template_kwargs &&
    Object.keys(body.chat_template_kwargs).length === 0
  ) {
    delete body.chat_template_kwargs;
  }

  return copyAllowedFields(
    body,
    COMMON_ALLOWED_FIELDS
  );
}

// ============================================================
// REASONING OUTPUT FILTER
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
    if (REASONING_KEYS.has(key.toLowerCase())) {
      continue;
    }

    output[key] = stripReasoningFields(child);
  }

  return output;
}

// ============================================================
// UPSTREAM HELPERS
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

async function postNvidiaChat(
  body,
  streaming,
  signal
) {
  return axios.post(
    `${NVIDIA_BASE_URL}/chat/completions`,
    body,
    {
      headers: upstreamHeaders(streaming),

      responseType: streaming
        ? 'stream'
        : 'json',

      timeout: 0,
      signal,
      validateStatus: () => true,
      maxRedirects: 0,
      maxContentLength: 100 * 1024 * 1024,
      maxBodyLength: 100 * 1024 * 1024
    }
  );
}

async function readStreamText(
  stream,
  maxBytes = 65536
) {
  const chunks = [];
  let total = 0;

  try {
    for await (const chunk of stream) {
      const part = Buffer.from(chunk);
      const remaining = maxBytes - total;

      if (remaining <= 0) break;

      const accepted = part.subarray(0, remaining);

      chunks.push(accepted);
      total += accepted.length;

      if (total >= maxBytes) break;
    }
  } finally {
    if (!stream.destroyed && !stream.readableEnded) {
      stream.destroy();
    }
  }

  return Buffer.concat(chunks).toString('utf8');
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

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      return reject(
        signal.reason || new Error('Request aborted.')
      );
    }

    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);

    function cleanup() {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    }

    function onAbort() {
      cleanup();
      reject(
        signal.reason || new Error('Request aborted.')
      );
    }

    signal.addEventListener('abort', onAbort, {
      once: true
    });
  });
}

async function pollNvidiaResult(requestId, signal) {
  const url =
    `${NVIDIA_BASE_URL}/status/${encodeURIComponent(requestId)}`;

  let count = 0;

  while (true) {
    if (signal.aborted) {
      throw signal.reason || new Error('Request aborted.');
    }

    count += 1;

    await sleep(
      Math.min(1000 + count * 250, 3000),
      signal
    );

    const response = await axios.get(url, {
      headers: {
        Authorization: `Bearer ${NVIDIA_API_KEY}`,
        Accept: 'application/json'
      },
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
        `NVIDIA status endpoint returned HTTP ${response.status}.`
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
// SERVER-SENT EVENTS
// ============================================================

function extractSseEvents(buffer) {
  const events = [];
  let remaining = buffer;
  let match;

  while (
    (match = /\r?\n\r?\n/.exec(remaining)) !== null
  ) {
    events.push(
      remaining.slice(0, match.index)
    );

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
      outputData = data;
    }
  }

  return `${prefix}data: ${outputData}\n\n`;
}

function isDoneEvent(eventText) {
  return /(?:^|\n)data:\s*\[DONE\]\s*(?:\n|$)/.test(eventText);
}

function writeChunk(res, chunk) {
  if (res.destroyed || res.writableEnded) {
    return Promise.resolve(false);
  }

  if (res.write(chunk)) {
    return Promise.resolve(true);
  }

  return new Promise(resolve => {
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

  res.setHeader(
    'Content-Type',
    'text/event-stream; charset=utf-8'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache, no-transform'
  );

  res.setHeader('X-Accel-Buffering', 'no');

  res.flushHeaders?.();
}

async function sendCompletionAsSse(res, data) {
  const result = STRIP_REASONING
    ? stripReasoningFields(data)
    : data;

  startSseResponse(res);

  if (
    !result ||
    !Array.isArray(result.choices)
  ) {
    await writeChunk(
      res,
      `data: ${JSON.stringify(result)}\n\n`
    );

    await writeChunk(
      res,
      'data: [DONE]\n\n'
    );

    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }

    return;
  }

  const id =
    result.id || `chatcmpl-${crypto.randomUUID()}`;

  const created =
    result.created || Math.floor(Date.now() / 1000);

  const model =
    result.model || 'nvidia-nim-proxy';

  for (const choice of result.choices) {
    const message = choice.message || {};

    const index = Number.isInteger(choice.index)
      ? choice.index
      : 0;

    const delta = { ...message };

    delete delta.role;

    await writeChunk(
      res,
      `data: ${JSON.stringify({
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{
          index,
          delta: {
            role: 'assistant',
            ...delta
          },
          finish_reason: null
        }]
      })}\n\n`
    );

    await writeChunk(
      res,
      `data: ${JSON.stringify({
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{
          index,
          delta: {},
          finish_reason: choice.finish_reason || 'stop'
        }]
      })}\n\n`
    );
  }

  if (result.usage) {
    await writeChunk(
      res,
      `data: ${JSON.stringify({
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [],
        usage: result.usage
      })}\n\n`
    );
  }

  await writeChunk(
    res,
    'data: [DONE]\n\n'
  );

  if (!res.destroyed && !res.writableEnded) {
    res.end();
  }
}

// ============================================================
// STATUS ENDPOINTS
// ============================================================

app.get('/', (req, res) => {
  res.json({
    name: 'NVIDIA NIM JanitorAI RP Proxy',
    status: 'ok',
    defaultModel: DEFAULT_MODEL,
    modelProfiles: Object.keys(MODEL_PROFILES).length,
    rpContinuityMode: RP_CONTINUITY_MODE,
    conversationMemory: 'stateless',
    conversationIsolation: 'request-scoped; no cross-chat memory is stored',
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
    rpContinuityMode: RP_CONTINUITY_MODE,
    conversationMemory: 'stateless'
  });
});

app.get('/v1/models', (req, res) => {
  const created = Math.floor(Date.now() / 1000);

  res.json({
    object: 'list',
    data: Object.values(MODEL_PROFILES).map(profile => ({
      id: profile.id,
      object: 'model',
      created,
      owned_by: 'nvidia-nim-proxy',
      name: profile.label
    }))
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

    const validationError =
      validateChatRequest(requestBody);

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

    let body;

    try {
      body = applyProfile(requestBody, profile);
    } catch (error) {
      return jsonError(
        res,
        400,
        error.message || 'Invalid request.',
        'invalid_request_error'
      );
    }

    if (
      body.max_tokens === undefined &&
      requestBody.max_completion_tokens !== undefined
    ) {
      body.max_tokens =
        requestBody.max_completion_tokens;
    }

    if (body.max_tokens === undefined) {
      body.max_tokens = DEFAULT_MAX_TOKENS;
    }

    const streaming = body.stream === true;

    if (
      profile.adapter === 'mistral4' &&
      (
        !Number.isInteger(body.max_tokens) ||
        body.max_tokens < 1 ||
        body.max_tokens > 262144
      )
    ) {
      return jsonError(
        res,
        400,
        'Mistral Large 4 max_tokens must be between 1 and 262144.',
        'invalid_request_error'
      );
    }

    const originalMistralMessages =
      profile.adapter === 'mistral4'
        ? requestBody.messages
        : [];

    const controller = new AbortController();

    let upstreamStream = null;
    let clientDisconnected = false;
    let timedOut = false;

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

    const timeout = setTimeout(() => {
      if (clientDisconnected) return;

      timedOut = true;

      const error = Object.assign(
        new Error('NVIDIA request timed out.'),
        {
          code: 'UPSTREAM_TIMEOUT'
        }
      );

      controller.abort(error);

      if (
        upstreamStream &&
        !upstreamStream.destroyed
      ) {
        upstreamStream.destroy(error);
      }
    }, REQUEST_TIMEOUT_MS);

    // --------------------------------------------------------
    // MISTRAL FALLBACK DIAGNOSTIC STATE
    // --------------------------------------------------------

    let mistralFallbackAttempted = false;
    let mistralFallbackStage = 0;
    let mistralFirstError = null;
    let mistralSecondError = null;

    try {
      // ------------------------------------------------------
      // INITIAL UPSTREAM REQUEST
      // ------------------------------------------------------

      let upstream = await postNvidiaChat(
        body,
        streaming,
        controller.signal
      );

      // ------------------------------------------------------
      // AUTOMATIC MISTRAL HTTP 400 FALLBACK
      //
      // Retry 1:
      // Core fields, high reasoning.
      //
      // Retry 2:
      // Core fields, none reasoning.
      //
      // Original history is retained in both requests.
      // ------------------------------------------------------

      if (
        profile.adapter === 'mistral4' &&
        upstream.status === 400 &&
        !clientDisconnected
      ) {
        let firstErrorData = upstream.data;

        if (streaming && upstream.data) {
          firstErrorData = parseJsonOrText(
            await readStreamText(
              upstream.data,
              65536
            )
          );
        }

        mistralFirstError =
          safeErrorForLog(firstErrorData);

        mistralFallbackAttempted = true;
        mistralFallbackStage = 1;

        let fallbackBody =
          buildMistralFallbackBody(body, 'high');

        console.warn(
          'Mistral HTTP 400. Retrying with core fields and high reasoning.',
          JSON.stringify({
            requestId: req.proxyRequestId,
            preservedMessageCount: fallbackBody.messages.length,
            removedOptionalFields: Object.keys(body).filter(
              key =>
                !Object.prototype.hasOwnProperty.call(
                  fallbackBody,
                  key
                )
            )
          })
        );

        upstream = await postNvidiaChat(
          fallbackBody,
          streaming,
          controller.signal
        );

        if (
          upstream.status === 400 &&
          !clientDisconnected
        ) {
          let secondErrorData = upstream.data;

          if (streaming && upstream.data) {
            secondErrorData = parseJsonOrText(
              await readStreamText(
                upstream.data,
                65536
              )
            );
          }

          mistralSecondError =
            safeErrorForLog(secondErrorData);

          mistralFallbackStage = 2;

          fallbackBody =
            buildMistralFallbackBody(body, 'none');

          console.warn(
            'Mistral core request still returned HTTP 400. Retrying with reasoning_effort=none.',
            JSON.stringify({
              requestId: req.proxyRequestId,
              preservedMessageCount: fallbackBody.messages.length
            })
          );

          upstream = await postNvidiaChat(
            fallbackBody,
            streaming,
            controller.signal
          );
        }

        // Keep the final attempted request for diagnostics.
        body = fallbackBody;
      }

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
      // NVIDIA ASYNC 202 RESPONSES
      // ------------------------------------------------------

      let completedByPolling = false;

      if (upstream.status === 202) {
        let pendingData = upstream.data;

        if (streaming && upstream.data) {
          upstreamStream = upstream.data;

          const text = await readStreamText(
            upstreamStream,
            1024 * 1024
          );

          upstreamStream = null;

          pendingData = parseJsonOrText(text);
        }

        const requestId = extractRequestId(
          upstream.headers,
          pendingData
        );

        if (!requestId) {
          return jsonError(
            res,
            502,
            'NVIDIA returned 202 Accepted without a request ID.',
            'upstream_error'
          );
        }

        const completed = await pollNvidiaResult(
          requestId,
          controller.signal
        );

        upstream = {
          ...upstream,
          status: 200,
          data: completed
        };

        completedByPolling = true;
      }

      if (clientDisconnected || res.destroyed) {
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

          const text = await readStreamText(
            upstreamStream,
            65536
          );

          upstreamStream = null;

          errorData = parseJsonOrText(text);
        }

        if (clientDisconnected || res.destroyed) {
          return;
        }

        const diagnostic = {
          requestId: req.proxyRequestId,
          status: upstream.status,
          requestedModel,
          upstreamModel: profile.upstream,
          streaming,
          adapter: profile.adapter,
          upstreamError: safeErrorForLog(errorData)
        };

        if (profile.adapter === 'mistral4') {
          diagnostic.reasoningEffort =
            body.reasoning_effort;

          diagnostic.maxTokens =
            body.max_tokens;

          diagnostic.messageStructure =
            summarizeMessages(body.messages);

          diagnostic.inputMessageCount =
            originalMistralMessages.length;

          diagnostic.outputMessageCount =
            body.messages.length;

          diagnostic.systemMessageCount =
            body.messages.filter(
              message => message.role === 'system'
            ).length;

          diagnostic.lastThreeRoles =
            body.messages.slice(-3).map(
              message => message.role
            );

          diagnostic.endsWithAssistant =
            body.messages.at(-1)?.role === 'assistant';

          diagnostic.temperature = body.temperature;
          diagnostic.topP = body.top_p;

          diagnostic.fallbackAttempted =
            mistralFallbackAttempted;

          diagnostic.fallbackStage =
            mistralFallbackStage;

          diagnostic.firstAttemptError =
            mistralFirstError;

          diagnostic.secondAttemptError =
            mistralSecondError;

          diagnostic.whitespaceOnlyMessagesInOriginal =
            countWhitespaceOnlyMessages(
              originalMistralMessages
            );

          diagnostic.nullReasoningFieldsInOriginal =
            countNullReasoningFields(
              originalMistralMessages
            );

          diagnostic.requestBodyBytes =
            Buffer.byteLength(
              JSON.stringify(body),
              'utf8'
            );

          diagnostic.oneCharacterMessages =
            findOneCharacterMessages(
              body.messages
            );
        }

        console.error(
          'NVIDIA API error:',
          JSON.stringify(diagnostic)
        );

        return jsonError(
          res,
          upstream.status,
          upstreamErrorMessage(
            errorData,
            `NVIDIA returned HTTP ${upstream.status}.`
          ),
          'upstream_error'
        );
      }

      // ------------------------------------------------------
      // FALLBACK SUCCESS LOG
      // ------------------------------------------------------

      if (
        profile.adapter === 'mistral4' &&
        mistralFallbackAttempted
      ) {
        console.warn(
          'Mistral core-request fallback succeeded.',
          JSON.stringify({
            requestId: req.proxyRequestId,
            model: profile.upstream,
            status: upstream.status,
            successfulFallbackStage:
              mistralFallbackStage
          })
        );
      }

      // ------------------------------------------------------
      // STREAMING RESPONSE FROM POLLING
      // ------------------------------------------------------

      if (streaming && completedByPolling) {
        await sendCompletionAsSse(
          res,
          upstream.data
        );

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

              const output =
                transformSseEvent(eventText);

              if (!output) continue;

              if (isDoneEvent(output)) {
                receivedDone = true;
              }

              const writable = await writeChunk(
                res,
                output
              );

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

              const output =
                transformSseEvent(finalText);

              if (output) {
                const writable = await writeChunk(
                  res,
                  output
                );

                if (!writable) return;

                receivedDone = isDoneEvent(output);
              }
            }
          }

          if (!receivedDone) {
            await writeChunk(
              res,
              'data: [DONE]\n\n'
            );
          }

          if (!res.destroyed && !res.writableEnded) {
            res.end();
          }
        } catch (error) {
          if (clientDisconnected || res.destroyed) {
            return;
          }

          console.error(
            'NVIDIA stream failed:',
            JSON.stringify({
              requestId: req.proxyRequestId,
              requestedModel,
              code: error?.code || 'UNKNOWN',
              message: error?.message || 'Unknown stream error',
              timedOut
            })
          );

          if (!res.destroyed && !res.writableEnded) {
            const text = timedOut
              ? 'NVIDIA generation timed out.'
              : 'NVIDIA streaming connection failed.';

            await writeChunk(
              res,
              `data: ${JSON.stringify({
                error: {
                  message: text,
                  type: timedOut
                    ? 'upstream_timeout'
                    : 'upstream_error'
                }
              })}\n\n`
            );

            if (!res.destroyed && !res.writableEnded) {
              res.end();
            }
          }
        }

        return;
      }

      // ------------------------------------------------------
      // NON-STREAMING RESPONSE
      // ------------------------------------------------------

      const responseData = STRIP_REASONING
        ? stripReasoningFields(upstream.data)
        : upstream.data;

      return res.status(upstream.status).json(
        responseData
      );
    } catch (error) {
      if (clientDisconnected || res.destroyed) {
        return;
      }

      const isTimeout =
        timedOut ||
        error?.code === 'UPSTREAM_TIMEOUT' ||
        controller.signal.reason?.code === 'UPSTREAM_TIMEOUT';

      if (isTimeout) {
        return jsonError(
          res,
          504,
          'NVIDIA request timed out.',
          'upstream_timeout'
        );
      }

      if (controller.signal.aborted) {
        return;
      }

      const status =
        error?.response?.status || 502;

      const message = upstreamErrorMessage(
        error?.response?.data,
        error?.message || 'Could not reach NVIDIA.'
      );

      console.error(
        'Proxy request failed:',
        JSON.stringify({
          requestId: req.proxyRequestId,
          status,
          requestedModel,
          upstreamModel: profile.upstream,
          code: error?.code || 'UNKNOWN',
          message
        })
      );

      return jsonError(
        res,
        status,
        message,
        'upstream_error'
      );
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
// FINAL ERROR HANDLER
// ============================================================

app.use((err, req, res, next) => {
  if (res.headersSent) {
    return next(err);
  }

  if (err?.type === 'entity.too.large') {
    return jsonError(
      res,
      413,
      `Request body exceeds configured limit (${MAX_BODY_BYTES}).`,
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

const server = app.listen(
  PORT,
  '0.0.0.0',
  () => {
    console.log('========================================');
    console.log('NVIDIA NIM JanitorAI RP Proxy');
    console.log('========================================');
    console.log(`Port: ${PORT}`);
    console.log(`Default model: ${DEFAULT_MODEL}`);
    console.log(`NVIDIA endpoint: ${NVIDIA_BASE_URL}`);
    console.log(
      `Model profiles: ${Object.keys(MODEL_PROFILES).length}`
    );
    console.log(
      `RP continuity instruction: ${RP_CONTINUITY_MODE}`
    );
    console.log(
      'Conversation memory: stateless (no cross-chat persistence)'
    );
    console.log(`Reasoning stripping: ${STRIP_REASONING}`);
    console.log(
      `Concurrent generations: ${MAX_CONCURRENT_REQUESTS}`
    );
    console.log(`Default max tokens: ${DEFAULT_MAX_TOKENS}`);
    console.log(`Timeout: ${REQUEST_TIMEOUT_MS}ms`);
    console.log(
      `Proxy authentication: ${PROXY_API_KEY ? 'enabled' : 'disabled'}`
    );
    console.log('========================================');
  }
);

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
