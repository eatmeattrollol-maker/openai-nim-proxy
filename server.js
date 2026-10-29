
'use strict';

const express = require('express');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');

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
  process.env.DEFAULT_MODEL || 'moonshotai/kimi-k3';

const REQUEST_TIMEOUT_MS = Number(
  process.env.REQUEST_TIMEOUT_MS || 600000
);

const MAX_BODY_BYTES = process.env.MAX_BODY_BYTES || '5mb';

const MAX_CONCURRENT_REQUESTS = Number(
  process.env.MAX_CONCURRENT_REQUESTS || 4
);

const STRIP_REASONING =
  (process.env.STRIP_REASONING || 'true').toLowerCase() === 'true';

const ALLOW_UNKNOWN_MODELS =
  (process.env.ALLOW_UNKNOWN_MODELS || 'false').toLowerCase() === 'true';

// This proxy uses NVIDIA_API_KEY for outbound NVIDIA requests.
// Incoming JanitorAI API keys are not validated by this server.

if (!NVIDIA_API_KEY) {
  console.error(
    'FATAL: NVIDIA_API_KEY is missing. Check Render Environment.'
  );
  process.exit(1);
}

if (!Number.isFinite(PORT) || PORT < 1 || PORT > 65535) {
  throw new Error('PORT must be a valid TCP port.');
}

if (
  !Number.isFinite(MAX_CONCURRENT_REQUESTS) ||
  MAX_CONCURRENT_REQUESTS < 1
) {
  throw new Error('MAX_CONCURRENT_REQUESTS must be at least 1.');
}

if (
  !Number.isFinite(REQUEST_TIMEOUT_MS) ||
  REQUEST_TIMEOUT_MS < 1000
) {
  throw new Error('REQUEST_TIMEOUT_MS must be at least 1000.');
}

app.disable('x-powered-by');
app.set('trust proxy', 1);

// ============================================================
// Request logging
// ============================================================

// Logs request method, path, status, and duration.
// Does not log API keys, prompts, or response contents.

app.use((req, res, next) => {
  const started = Date.now();

  res.on('finish', () => {
    console.log(
      `${req.method} ${req.path} -> ${res.statusCode} ` +
      `(${Date.now() - started}ms)`
    );
  });

  next();
});

// ============================================================
// Middleware
// ============================================================

app.use(express.json({ limit: MAX_BODY_BYTES }));

// Optional CORS support for browser-based clients.
// JanitorAI server-to-server requests do not normally need CORS.

const allowedOrigins = (process.env.CORS_ORIGINS || '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use((req, res, next) => {
  const origin = req.get('origin');

  if (origin && allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
    res.setHeader(
      'Access-Control-Allow-Headers',
      'Authorization, Content-Type'
    );
    res.setHeader(
      'Access-Control-Allow-Methods',
      'GET, POST, OPTIONS'
    );
  }

  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }

  next();
});

// Limit concurrent requests to the proxy.

let activeRequests = 0;

app.use((req, res, next) => {
  if (!req.path.startsWith('/v1/')) {
    return next();
  }

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
// Model profiles
// ============================================================

// The profile ID is what JanitorAI sends.
// The upstream ID is what NVIDIA receives.
//
// Profiles can apply model-specific reasoning options.
// They do not guarantee that NVIDIA currently provides access
// to the corresponding upstream model.

const MODEL_PROFILES = {
  // ----------------------------------------------------------
  // Kimi K3
  // ----------------------------------------------------------

  'moonshotai/kimi-k3': {
    upstream: 'moonshotai/kimi-k3',
    label: 'Kimi K3',
    adapter: 'generic'
  },

  // ----------------------------------------------------------
  // Mistral Large 4
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // Mistral Nemotron
  // ----------------------------------------------------------

  'mistralai/mistral-nemotron': {
    upstream: 'mistralai/mistral-nemotron',
    label: 'Mistral Nemotron',
    adapter: 'generic'
  },

  // ----------------------------------------------------------
  // Nemotron 3 Ultra
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // Nemotron 3.5 Super VL
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // GLM 5.3 Flash
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // DeepSeek V4.1 Flash
  // ----------------------------------------------------------

  'deepseek-ai/deepseek-v4.1-flash': {
    upstream: 'deepseek-ai/deepseek-v4.1-flash',
    label: 'DeepSeek V4.1 Flash',
    adapter: 'deepseek'
  },

  // ----------------------------------------------------------
  // Step 5 Preview
  // ----------------------------------------------------------

  'stepfun-ai/step-5-preview': {
    upstream: 'stepfun-ai/step-5-preview',
    label: 'Step 5 Preview',
    adapter: 'step'
  },

  // ----------------------------------------------------------
  // MiniMax M3.1 Flash
  // ----------------------------------------------------------

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

  // ----------------------------------------------------------
  // Other models
  // ----------------------------------------------------------

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
// Helper functions
// ============================================================

function jsonError(res, status, message, type = 'proxy_error') {
  if (res.headersSent) {
    res.end();
    return;
  }

  res.status(status).json({
    error: {
      message,
      type
    }
  });
}

function getProfile(requestedModel) {
  const profile = MODEL_PROFILES[requestedModel];

  if (profile) {
    return {
      requestedModel,
      profile
    };
  }

  if (
    ALLOW_UNKNOWN_MODELS &&
    typeof requestedModel === 'string' &&
    requestedModel.trim()
  ) {
    return {
      requestedModel,
      profile: {
        upstream: requestedModel,
        label: requestedModel,
        adapter: 'generic'
      }
    };
  }

  return null;
}

function cleanText(value) {
  if (typeof value !== 'string') {
    return value;
  }

  return value
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(
      /<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>/gi,
      ''
    )
    .replace(/^\s*<think>\s*/i, '')
    .trim();
}

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
    return typeof value === 'string' ? cleanText(value) : value;
  }

  const output = {};

  for (const [key, child] of Object.entries(value)) {
    if (STRIP_REASONING && REASONING_KEYS.has(key.toLowerCase())) {
      continue;
    }

    output[key] = stripReasoningFields(child);
  }

  return output;
}

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

    case 'mistral-large-4': {
      if (profile.reasoningEffort) {
        body.reasoning_effort = profile.reasoningEffort;
      }

      break;
    }

    case 'muse-glimmer': {
      if (profile.reasoningEffort) {
        body.reasoning_effort = profile.reasoningEffort;
      }

      break;
    }

    case 'deepseek':
    case 'step':
    case 'generic':
    default:
      // Preserve the client's generation settings.
      break;
  }

  return body;
}

function validateChatRequest(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return 'Request body must be a JSON object.';
  }

  if (typeof body.model !== 'string' || !body.model.trim()) {
    return 'A model name is required.';
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return 'messages must be a non-empty array.';
  }

  for (const message of body.messages) {
    if (
      !message ||
      typeof message !== 'object' ||
      typeof message.role !== 'string'
    ) {
      return 'Each message must be an object with a role.';
    }
  }

  if (
    body.temperature !== undefined &&
    (
      typeof body.temperature !== 'number' ||
      !Number.isFinite(body.temperature)
    )
  ) {
    return 'temperature must be a finite number.';
  }

  if (
    body.top_p !== undefined &&
    (
      typeof body.top_p !== 'number' ||
      !Number.isFinite(body.top_p)
    )
  ) {
    return 'top_p must be a finite number.';
  }

  if (
    body.max_tokens !== undefined &&
    (
      !Number.isInteger(body.max_tokens) ||
      body.max_tokens < 1
    )
  ) {
    return 'max_tokens must be a positive integer.';
  }

  if (
    body.max_completion_tokens !== undefined &&
    (
      !Number.isInteger(body.max_completion_tokens) ||
      body.max_completion_tokens < 1
    )
  ) {
    return 'max_completion_tokens must be a positive integer.';
  }

  return null;
}

function upstreamHeaders(streaming = false) {
  return {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    'Content-Type': 'application/json',
    Accept: streaming ? 'text/event-stream' : 'application/json'
  };
}

function getUpstreamErrorMessage(error) {
  const data = error?.response?.data;

  if (data && typeof data === 'object') {
    const message =
      data.error?.message ||
      data.message ||
      data.detail;

    if (typeof message === 'string') {
      return message.slice(0, 2000);
    }
  }

  if (typeof data === 'string') {
    return data.slice(0, 2000);
  }

  if (error?.code === 'ECONNABORTED') {
    return 'The NVIDIA request timed out.';
  }

  if (error?.code === 'ERR_CANCELED') {
    return 'The upstream request was canceled.';
  }

  return error?.message || 'Unknown upstream error.';
}

function parseUpstreamError(rawError, status) {
  let message =
    rawError || `NVIDIA returned HTTP ${status}`;

  try {
    const parsed = JSON.parse(rawError);

    message =
      parsed.error?.message ||
      parsed.message ||
      message;
  } catch {
    // Keep the raw response when it is not JSON.
  }

  return String(message).slice(0, 2000);
}

// ============================================================
// Health and model endpoints
// ============================================================

app.get('/', (req, res) => {
  res.json({
    name: 'NVIDIA NIM JanitorAI Proxy',
    status: 'ok',
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
    maxConcurrentRequests: MAX_CONCURRENT_REQUESTS
  });
});

app.get('/v1/models', (req, res) => {
  const created = Math.floor(Date.now() / 1000);

  const models = Object.entries(MODEL_PROFILES).map(
    ([id, profile]) => ({
      id,
      object: 'model',
      created,
      owned_by: 'nvidia-nim-proxy',
      name: profile.label
    })
  );

  // Include the default model if it is not already listed.
  if (!models.some((model) => model.id === DEFAULT_MODEL)) {
    models.unshift({
      id: DEFAULT_MODEL,
      object: 'model',
      created,
      owned_by: 'nvidia-nim-proxy',
      name: DEFAULT_MODEL
    });
  }

  res.json({
    object: 'list',
    data: models
  });
});

// ============================================================
// Chat completions
// ============================================================

app.post('/v1/chat/completions', async (req, res) => {
  // Fall back to DEFAULT_MODEL only if the client omitted a model.
  // A model explicitly supplied by JanitorAI takes precedence.

  if (
    req.body &&
    typeof req.body === 'object' &&
    !Array.isArray(req.body) &&
    !req.body.model
  ) {
    req.body.model = DEFAULT_MODEL;
  }

  const validationError = validateChatRequest(req.body);

  if (validationError) {
    return jsonError(
      res,
      400,
      validationError,
      'invalid_request_error'
    );
  }

  const requestedModel = req.body.model;
  const resolved = getProfile(requestedModel);

  if (!resolved) {
    return jsonError(
      res,
      400,
      `Unknown model "${requestedModel}". Check GET /v1/models.`,
      'invalid_request_error'
    );
  }

  const { profile } = resolved;
  const body = applyProfile(req.body, profile);

  // Preserve the generation limit supplied by JanitorAI.
  // Apply a fallback only when neither token-limit field exists.

  if (
    body.max_tokens === undefined &&
    body.max_completion_tokens === undefined
  ) {
    body.max_tokens = 8192;
  }

  const controller = new AbortController();

  let clientDisconnected = false;
  let upstreamStream = null;

  const cancelUpstream = () => {
    clientDisconnected = true;
    controller.abort();

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  };

  req.on('aborted', cancelUpstream);

  res.on('close', () => {
    if (!res.writableEnded) {
      cancelUpstream();
    }
  });

  try {
    // --------------------------------------------------------
    // Streaming requests
    // --------------------------------------------------------

    if (body.stream === true) {
      const upstream = await axios.post(
        `${NVIDIA_BASE_URL}/chat/completions`,
        body,
        {
          headers: upstreamHeaders(true),
          responseType: 'stream',
          timeout: REQUEST_TIMEOUT_MS,
          signal: controller.signal,
          validateStatus: () => true
        }
      );

      upstreamStream = upstream.data;

      if (upstream.status < 200 || upstream.status >= 300) {
        const chunks = [];
        let totalBytes = 0;

        for await (const chunk of upstreamStream) {
          const bufferChunk = Buffer.from(chunk);

          chunks.push(bufferChunk);
          totalBytes += bufferChunk.length;

          if (totalBytes > 1024 * 1024) {
            break;
          }
        }

        if (clientDisconnected) return;

        const rawError = Buffer.concat(chunks).toString('utf8');

        return jsonError(
          res,
          upstream.status,
          parseUpstreamError(rawError, upstream.status),
          'upstream_error'
        );
      }

      res.status(200);
      res.setHeader(
        'Content-Type',
        'text/event-stream; charset=utf-8'
      );
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      res.flushHeaders?.();

      const decoder = new StringDecoder('utf8');
      let buffer = '';

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

        if (data.trim() === '[DONE]') {
          const prefix = otherLines.length
            ? `${otherLines.join('\n')}\n`
            : '';

          return `${prefix}data: [DONE]\n\n`;
        }

        let transformedData = data;

        if (STRIP_REASONING) {
          try {
            const parsed = JSON.parse(data);

            transformedData = JSON.stringify(
              stripReasoningFields(parsed)
            );
          } catch {
            transformedData = cleanText(data);
          }
        }

        const prefix = otherLines.length
          ? `${otherLines.join('\n')}\n`
          : '';

        return `${prefix}data: ${transformedData}\n\n`;
      }

      upstreamStream.on('data', (chunk) => {
        if (clientDisconnected || res.destroyed) return;

        buffer += decoder.write(chunk);

        let boundary;

        while ((boundary = buffer.search(/\r?\n\r?\n/)) !== -1) {
          const eventText = buffer.slice(0, boundary);

          const separator = buffer
            .slice(boundary)
            .match(/^\r?\n\r?\n/)[0];

          buffer = buffer.slice(boundary + separator.length);

          const output = transformSseEvent(eventText);

          if (output && !res.destroyed) {
            res.write(output);
          }
        }
      });

      upstreamStream.on('end', () => {
        if (clientDisconnected || res.destroyed) return;

        buffer += decoder.end();

        if (buffer.length > 0) {
          res.write(transformSseEvent(buffer));
        }

        res.end();
      });

      upstreamStream.on('error', (error) => {
        if (clientDisconnected || res.destroyed) return;

        console.error('NVIDIA streaming error:', {
          code: error?.code || 'UNKNOWN',
          message: error?.message || 'Unknown streaming error',
          model: requestedModel
        });

        if (!res.headersSent) {
          jsonError(
            res,
            502,
            'Streaming connection to NVIDIA failed.',
            'upstream_error'
          );
        } else {
          res.write(
            `data: ${JSON.stringify({
              error: {
                message: 'Streaming connection to NVIDIA failed.',
                type: 'upstream_error'
              }
            })}\n\n`
          );

          res.end();
        }
      });

      return;
    }

    // --------------------------------------------------------
    // Non-streaming requests
    // --------------------------------------------------------

    const upstream = await axios.post(
      `${NVIDIA_BASE_URL}/chat/completions`,
      body,
      {
        headers: upstreamHeaders(false),
        timeout: REQUEST_TIMEOUT_MS,
        signal: controller.signal,
        validateStatus: () => true
      }
    );

    if (clientDisconnected) return;

    if (upstream.status < 200 || upstream.status >= 300) {
      const message = getUpstreamErrorMessage({
        response: { data: upstream.data }
      });

      console.error('NVIDIA returned an error:', {
        status: upstream.status,
        model: requestedModel,
        message
      });

      return jsonError(
        res,
        upstream.status,
        message,
        'upstream_error'
      );
    }

    const responseData = STRIP_REASONING
      ? stripReasoningFields(upstream.data)
      : upstream.data;

    return res.status(upstream.status).json(responseData);
  } catch (error) {
    if (
      clientDisconnected ||
      error?.code === 'ERR_CANCELED'
    ) {
      return;
    }

    const status = error?.response?.status || 502;
    const message = getUpstreamErrorMessage(error);

    // Never log credentials or full conversation contents.

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

  if (err instanceof SyntaxError && 'body' in err) {
    return jsonError(
      res,
      400,
      'Malformed JSON request body.',
      'invalid_request_error'
    );
  }

  console.error('Unhandled server error:', err?.message || err);

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

app.listen(PORT, '0.0.0.0', () => {
  console.log(`NVIDIA NIM proxy listening on port ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`NVIDIA base URL: ${NVIDIA_BASE_URL}`);
  console.log(`Reasoning stripping: ${STRIP_REASONING}`);
  console.log(`Allow unknown models: ${ALLOW_UNKNOWN_MODELS}`);
  console.log(`Concurrent request limit: ${MAX_CONCURRENT_REQUESTS}`);
});
