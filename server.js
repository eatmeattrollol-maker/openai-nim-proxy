
'use strict';

const express = require('express');
const axios = require('axios');
const crypto = require('node:crypto');
const { StringDecoder } = require('node:string_decoder');

const app = express();

// ============================================================
// Configuration
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

const STRIP_REASONING =
  (process.env.STRIP_REASONING || 'true').toLowerCase() === 'true';

const ALLOW_UNKNOWN_MODELS =
  (process.env.ALLOW_UNKNOWN_MODELS || 'true').toLowerCase() === 'true';

const PROXY_API_KEY = process.env.PROXY_API_KEY || '';

const ALLOWED_ORIGINS = (
  process.env.CORS_ORIGINS || 'https://janitorai.com'
)
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

if (!NVIDIA_API_KEY) {
  console.error('FATAL: NVIDIA_API_KEY is missing in Render Environment.');
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

if (PROXY_API_KEY && PROXY_API_KEY.length < 32) {
  throw new Error('PROXY_API_KEY must be at least 32 characters.');
}

app.disable('x-powered-by');
app.set('trust proxy', 1);

// ============================================================
// Model profiles
// ============================================================

const MODEL_PROFILES = {
  'moonshotai/kimi-k3': {
    upstream: 'moonshotai/kimi-k3',
    label: 'Kimi K3',
    adapter: 'generic'
  },

  'mistralai/mistral-large-4-1t-a52b': {
    upstream: 'mistralai/mistral-large-4-1t-a52b',
    label: 'Mistral Large 4',
    adapter: 'mistral-large-4'
  },

  'mistralai/mistral-large-4-1t-a52b-low': {
    upstream: 'mistralai/mistral-large-4-1t-a52b',
    label: 'Mistral Large 4 - Low Reasoning',
    adapter: 'mistral-large-4',
    reasoningEffort: 'low'
  },

  'mistralai/mistral-large-4-1t-a52b-medium': {
    upstream: 'mistralai/mistral-large-4-1t-a52b',
    label: 'Mistral Large 4 - Medium Reasoning',
    adapter: 'mistral-large-4',
    reasoningEffort: 'medium'
  },

  'mistralai/mistral-large-4-1t-a52b-high': {
    upstream: 'mistralai/mistral-large-4-1t-a52b',
    label: 'Mistral Large 4 - High Reasoning',
    adapter: 'mistral-large-4',
    reasoningEffort: 'high'
  },

  'mistralai/mistral-nemotron': {
    upstream: 'mistralai/mistral-nemotron',
    label: 'Mistral Nemotron',
    adapter: 'generic'
  },

  'nvidia/nemotron-3-ultra-550b-a55b': {
    upstream: 'nvidia/nemotron-3-ultra-550b-a55b',
    label: 'Nemotron 3 Ultra - Default',
    adapter: 'nemotron-ultra',
    thinking: true,
    thinkingBudget: 4096
  },

  'nvidia/nemotron-3-ultra-550b-a55b-no-thinking': {
    upstream: 'nvidia/nemotron-3-ultra-550b-a55b',
    label: 'Nemotron 3 Ultra - No Thinking',
    adapter: 'nemotron-ultra',
    thinking: false
  },

  'nvidia/nemotron-3-ultra-550b-a55b-thinking': {
    upstream: 'nvidia/nemotron-3-ultra-550b-a55b',
    label: 'Nemotron 3 Ultra - Thinking',
    adapter: 'nemotron-ultra',
    thinking: true,
    thinkingBudget: 4096
  },

  'nvidia/nemotron-3-ultra-550b-a55b-deep-thinking': {
    upstream: 'nvidia/nemotron-3-ultra-550b-a55b',
    label: 'Nemotron 3 Ultra - Deep Thinking',
    adapter: 'nemotron-ultra',
    thinking: true,
    thinkingBudget: 8192
  },

  'nvidia/nemotron-3.5-super-vl-120b-a12b': {
    upstream: 'nvidia/nemotron-3.5-super-vl-120b-a12b',
    label: 'Nemotron 3.5 Super VL',
    adapter: 'nemotron-vl',
    thinking: true
  },

  'nvidia/nemotron-3.5-super-vl-120b-a12b-no-thinking': {
    upstream: 'nvidia/nemotron-3.5-super-vl-120b-a12b',
    label: 'Nemotron 3.5 Super VL - No Thinking',
    adapter: 'nemotron-vl',
    thinking: false
  },

  'z-ai/glm-5.3-flash': {
    upstream: 'z-ai/glm-5.3-flash',
    label: 'GLM 5.3 Flash',
    adapter: 'glm-flash'
  },

  'z-ai/glm-5.3-flash-thinking': {
    upstream: 'z-ai/glm-5.3-flash',
    label: 'GLM 5.3 Flash - Thinking',
    adapter: 'glm-flash',
    thinking: true
  },

  'z-ai/glm-5.3-flash-no-thinking': {
    upstream: 'z-ai/glm-5.3-flash',
    label: 'GLM 5.3 Flash - No Thinking',
    adapter: 'glm-flash',
    thinking: false
  },

  'deepseek-ai/deepseek-v4.1-flash': {
    upstream: 'deepseek-ai/deepseek-v4.1-flash',
    label: 'DeepSeek V4.1 Flash',
    adapter: 'deepseek'
  },

  'stepfun-ai/step-5-preview': {
    upstream: 'stepfun-ai/step-5-preview',
    label: 'Step 5 Preview',
    adapter: 'step'
  },

  'minimaxai/minimax-m3.1-flash': {
    upstream: 'minimaxai/minimax-m3.1-flash',
    label: 'MiniMax M3.1 Flash',
    adapter: 'minimax-flash'
  },

  'minimaxai/minimax-m3.1-flash-low': {
    upstream: 'minimaxai/minimax-m3.1-flash',
    label: 'MiniMax M3.1 Flash - Low Reasoning',
    adapter: 'minimax-flash',
    reasoningEffort: 'low'
  },

  'minimaxai/minimax-m3.1-flash-medium': {
    upstream: 'minimaxai/minimax-m3.1-flash',
    label: 'MiniMax M3.1 Flash - Medium Reasoning',
    adapter: 'minimax-flash',
    reasoningEffort: 'medium'
  },

  'minimaxai/minimax-m3.1-flash-high': {
    upstream: 'minimaxai/minimax-m3.1-flash',
    label: 'MiniMax M3.1 Flash - High Reasoning',
    adapter: 'minimax-flash',
    reasoningEffort: 'high'
  },

  'poolside/laguna-xs-2.1': {
    upstream: 'poolside/laguna-xs-2.1',
    label: 'Laguna XS 2.1',
    adapter: 'generic'
  },

  'openai/gpt-oss-20b': {
    upstream: 'openai/gpt-oss-20b',
    label: 'GPT OSS 20B',
    adapter: 'generic'
  },

  'google/gemma-4-31b-it': {
    upstream: 'google/gemma-4-31b-it',
    label: 'Gemma 4 31B IT',
    adapter: 'generic'
  },

  'meta/muse-glimmer-30b': {
    upstream: 'meta/muse-glimmer-30b',
    label: 'Muse Glimmer 30B',
    adapter: 'muse-glimmer'
  }
};
 
// ============================================================
// Request logging
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
// JSON parsing
// ============================================================

app.use(express.json({
  limit: MAX_BODY_BYTES,
  strict: true
}));

// ============================================================
// Optional proxy authentication
// ============================================================

// Leave PROXY_API_KEY unset if your client cannot supply a
// separate proxy key. If set, JanitorAI must send it as:
// Authorization: Bearer YOUR_PROXY_API_KEY
//
// NVIDIA_API_KEY remains private on the server.

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
// Concurrency control
// ============================================================

// One user, one active generation by default.
// This prevents overlapping requests from consuming the
// free Render instance's resources.

let activeRequests = 0;

app.use('/v1', (req, res, next) => {
  if (activeRequests >= MAX_CONCURRENT_REQUESTS) {
    res.setHeader('Retry-After', '3');

    return res.status(503).json({
      error: {
        message: 'Proxy is busy. Try again shortly.',
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
});

// ============================================================
// Helper: standard JSON error
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

// ============================================================
// Helper: resolve model profiles
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

  // Unknown models are enabled by default.
  // NVIDIA still must recognize and provide access to the model.
  if (
    ALLOW_UNKNOWN_MODELS &&
    typeof requestedModel === 'string' &&
    requestedModel.trim()
  ) {
    return {
      upstream: requestedModel.trim(),
      label: requestedModel.trim(),
      adapter: 'generic'
    };
  }

  return null;
}

// ============================================================
// Helper: apply model-specific options
// ============================================================

function applyProfile(requestBody, profile) {
  const body = { ...requestBody };

  body.model = profile.upstream;

  const template = {
    ...(body.chat_template_kwargs || {})
  };

  switch (profile.adapter) {
    case 'nemotron-ultra': {
      template.enable_thinking = profile.thinking !== false;

      if (profile.thinking && profile.thinkingBudget) {
        body.thinking_token_budget = profile.thinkingBudget;
      } else {
        delete body.thinking_token_budget;
      }

      body.chat_template_kwargs = template;
      break;
    }

    case 'nemotron-vl': {
      template.enable_thinking = profile.thinking !== false;
      body.chat_template_kwargs = template;
      break;
    }

    case 'glm-flash': {
      template.clear_thinking = profile.thinking !== true;

      if (profile.reasoningEffort) {
        body.reasoning_effort = profile.reasoningEffort;
      }

      body.chat_template_kwargs = template;
      break;
    }

    case 'minimax-flash': {
      if (profile.reasoningEffort) {
        template.reasoning_effort = profile.reasoningEffort;
      }

      body.chat_template_kwargs = template;
      delete body.reasoning_effort;
      break;
    }

    case 'mistral-large-4':
    case 'muse-glimmer': {
      if (profile.reasoningEffort) {
        body.reasoning_effort = profile.reasoningEffort;
      }

      break;
    }

    default:
      // Generic profiles preserve client generation settings.
      break;
  }

  return body;
}

// ============================================================
// Helper: validate chat requests
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
// Helper: upstream request headers
// ============================================================

function upstreamHeaders(streaming = false) {
  return {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    'Content-Type': 'application/json',
    Accept: streaming
      ? 'text/event-stream'
      : 'application/json'
  };
}

// ============================================================
// Helper: extract upstream errors
// ============================================================

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
// Helper: strip reasoning fields from JSON responses
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
// Health endpoint
// ============================================================

app.get('/', (req, res) => {
  res.json({
    name: 'NVIDIA NIM JanitorAI Proxy',
    status: 'ok',
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
    allowUnknownModels: ALLOW_UNKNOWN_MODELS
  });
});

// ============================================================
// Models endpoint
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
// SSE transformation
// ============================================================

// Processes complete Server-Sent Events, not arbitrary chunks.
// This avoids corrupting JSON when NVIDIA splits a response
// across multiple network packets.

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

  // Preserve events that do not contain a data field.
  if (dataLines.length === 0) {
    return `${eventText}\n\n`;
  }

  const data = dataLines.join('\n');
  const prefix = otherLines.length
    ? `${otherLines.join('\n')}\n`
    : '';

  // The OpenAI-compatible stream terminator must be preserved.
  if (data.trim() === '[DONE]') {
    return `${prefix}data: [DONE]\n\n`;
  }

  let outputData = data;

  if (STRIP_REASONING) {
    try {
      const parsed = JSON.parse(data);
      outputData = JSON.stringify(stripReasoningFields(parsed));
    } catch {
      // Preserve unparseable data instead of corrupting it.
      outputData = data;
    }
  }

  return `${prefix}data: ${outputData}\n\n`;
}

// ============================================================
// SSE backpressure
// ============================================================

// Wait until the response socket can accept more data.
// This prevents buffering an entire long generation in RAM.

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
// SSE stream parser
// ============================================================

// Handles CRLF and LF separators and preserves incomplete events
// until the rest of the event arrives.

function extractSseEvents(buffer) {
  const events = [];
  let remaining = buffer;
  let match;

  while ((match = /\r?\n\r?\n/.exec(remaining)) !== null) {
    const eventText = remaining.slice(0, match.index);

    remaining = remaining.slice(
      match.index + match[0].length
    );

    events.push(eventText);
  }

  return {
    events,
    remaining
  };
}

 
// ============================================================
// Chat completions
// ============================================================

app.post('/v1/chat/completions', async (req, res) => {
  const requestBody = req.body;

  // Use the default model only when the client omits one.
  if (
    requestBody &&
    typeof requestBody === 'object' &&
    !Array.isArray(requestBody) &&
    !requestBody.model
  ) {
    requestBody.model = DEFAULT_MODEL;
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

  const body = applyProfile(requestBody, profile);

  // Preserve the client's token limits.
  if (
    body.max_tokens === undefined &&
    body.max_completion_tokens === undefined
  ) {
    body.max_tokens = 8192;
  }

  const streaming = body.stream === true;
  const controller = new AbortController();

  let clientDisconnected = false;
  let upstreamStream = null;
  let timeout;

  const cancelUpstream = () => {
    if (clientDisconnected) return;

    clientDisconnected = true;
    controller.abort();

    if (upstreamStream && !upstreamStream.destroyed) {
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

    console.warn('NVIDIA request timed out:', {
      model: requestedModel,
      timeoutMs: REQUEST_TIMEOUT_MS
    });

    controller.abort(
      Object.assign(new Error('UPSTREAM_TIMEOUT'), {
        code: 'UPSTREAM_TIMEOUT'
      })
    );

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy(
        Object.assign(new Error('UPSTREAM_TIMEOUT'), {
          code: 'UPSTREAM_TIMEOUT'
        })
      );
    }
  }, REQUEST_TIMEOUT_MS);

  try {
    const upstream = await axios.post(
      `${NVIDIA_BASE_URL}/chat/completions`,
      body,
      {
        headers: upstreamHeaders(streaming),
        responseType: streaming ? 'stream' : 'json',

        // Use our own overall deadline above.
        // Axios's timeout is disabled to avoid interrupting
        // a valid long-running generation due to inactivity.
        timeout: 0,

        signal: controller.signal,
        validateStatus: () => true,
        maxRedirects: 0
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

    // --------------------------------------------------------
    // Upstream error responses
    // --------------------------------------------------------

    if (upstream.status < 200 || upstream.status >= 300) {
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
        `NVIDIA returned HTTP ${upstream.status}`
      );

      console.error('NVIDIA API error:', {
  status: upstream.status,
  model: requestedModel,
  requestDetails: {
    stream: body.stream === true,
    messageCount: body.messages?.length,
    messageRoles: body.messages?.map((m) => m.role),
    messageContentTypes: body.messages?.map((m) =>
      Array.isArray(m.content) ? 'array' : typeof m.content
    ),
    keys: Object.keys(body),
    max_tokens: body.max_tokens,
    max_completion_tokens: body.max_completion_tokens,
    temperature: body.temperature,
    top_p: body.top_p,
    reasoning_effort: body.reasoning_effort,
    chat_template_kwargs: body.chat_template_kwargs
  },
  message: message
});

      return jsonError(
        res,
        upstream.status,
        message,
        'upstream_error'
      );
    }

    // --------------------------------------------------------
    // Streaming responses
    // --------------------------------------------------------

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
          }

          // Once NVIDIA signals completion, finish the response.
          if (receivedDone) {
            if (
              upstreamStream &&
              !upstreamStream.destroyed
            ) {
              upstreamStream.destroy();
            }

            break;
          }
        }

        if (clientDisconnected || res.destroyed) {
          return;
        }

        // Flush any remaining decoded bytes.
        buffer += decoder.end();

        if (buffer.trim()) {
          const finalText = buffer.replace(/(?:\r?\n)+$/, '');
          const output = transformSseEvent(finalText);

          if (output) {
            const writable = await writeChunk(res, output);

            if (!writable) return;
          }
        }

        // Some compatible upstreams close the stream without
        // emitting [DONE]. Terminate our HTTP response cleanly.
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
          model: requestedModel
        });

        if (!res.destroyed && !res.writableEnded) {
          if (res.headersSent) {
            const errorEvent =
              `data: ${JSON.stringify({
                error: {
                  message: 'NVIDIA streaming connection failed.',
                  type: 'upstream_error'
                }
              })}\n\n`;

            await writeChunk(res, errorEvent);

            if (!res.destroyed && !res.writableEnded) {
              res.end();
            }
          } else {
            jsonError(
              res,
              502,
              'NVIDIA streaming connection failed.',
              'upstream_error'
            );
          }
        }
      }

      return;
    }

    // --------------------------------------------------------
    // Non-streaming responses
    // --------------------------------------------------------

    const responseData = STRIP_REASONING
      ? stripReasoningFields(upstream.data)
      : upstream.data;

    return res.status(upstream.status).json(responseData);
  } catch (error) {
    if (clientDisconnected || res.destroyed) {
      return;
    }

    const isTimeout =
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
      model: requestedModel
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
});

 
// ============================================================
// Error handling
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
// Start server
// ============================================================

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`NVIDIA NIM proxy listening on port ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`NVIDIA base URL: ${NVIDIA_BASE_URL}`);
  console.log(`Reasoning stripping: ${STRIP_REASONING}`);
  console.log(`Unknown models enabled: ${ALLOW_UNKNOWN_MODELS}`);
  console.log(`Maximum concurrent requests: ${MAX_CONCURRENT_REQUESTS}`);
  console.log(
    `Proxy authentication: ${PROXY_API_KEY ? 'enabled' : 'disabled'}`
  );
});

// Do not let Node's incoming HTTP request timeout terminate
// long-running model generations. REQUEST_TIMEOUT_MS controls
// the upstream generation deadline instead.
server.requestTimeout = 0;
server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;

// ============================================================
// Graceful shutdown
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
