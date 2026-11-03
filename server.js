'use strict';

/*
 * JANITORAI MULTI-PROVIDER PROXY
 * NVIDIA NIM + Google Gemini
 *
 * Requirements: Node.js 20+
 * Dependencies: none
 *
 * Endpoints:
 *   GET  /
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/chat/completions
 *
 * No automatic system prompts.
 * No roleplay continuity prompts.
 * No synthetic user messages.
 * No conversation rewriting.
 */

const http = require('node:http');
const crypto = require('node:crypto');

// ============================================================
// CONFIGURATION
// ============================================================

const env = process.env;

function number(name, fallback, min = 1) {
  const value = env[name] === undefined
    ? fallback
    : Number(env[name]);

  if (!Number.isSafeInteger(value) || value < min) {
    throw new Error(`${name} must be an integer >= ${min}`);
  }

  return value;
}

function list(name, defaults) {
  if (!env[name]) return defaults;

  return [
    ...new Set(
      env[name]
        .split(',')
        .map(value => value.trim())
        .filter(Boolean)
    )
  ];
}

const PORT = number('PORT', 3000);

if (PORT > 65535) {
  throw new Error('Invalid PORT.');
}

const NVIDIA_KEY = env.NVIDIA_API_KEY || '';
const GEMINI_KEY = env.GEMINI_API_KEY || '';

const NVIDIA_URL = (
  env.NVIDIA_BASE_URL ||
  'https://integrate.api.nvidia.com/v1'
).replace(/\/+$/, '');

const GEMINI_URL = (
  env.GEMINI_BASE_URL ||
  'https://generativelanguage.googleapis.com/v1beta/openai'
).replace(/\/+$/, '');

const DEFAULT_MAX_TOKENS = number(
  'DEFAULT_MAX_TOKENS',
  8192
);

const MAX_BODY = number(
  'MAX_BODY_BYTES',
  20 * 1024 * 1024
);

const TIMEOUT = number(
  'REQUEST_TIMEOUT_MS',
  600000,
  1000
);

const NVIDIA_CONCURRENCY = number(
  'NVIDIA_MAX_CONCURRENT_REQUESTS',
  2
);

const GEMINI_CONCURRENCY = number(
  'GEMINI_MAX_CONCURRENT_REQUESTS',
  2
);

const QUEUE_LIMIT = number(
  'REQUEST_QUEUE_LIMIT',
  8,
  0
);

const QUEUE_TIMEOUT = number(
  'REQUEST_QUEUE_TIMEOUT_MS',
  30000
);

const STRIP_REASONING = env.STRIP_REASONING === 'true';
const ALLOW_CUSTOM_MODELS = env.ALLOW_CUSTOM_MODELS === 'true';

const PROXY_KEY = env.PROXY_API_KEY || '';

if (!NVIDIA_KEY && !GEMINI_KEY) {
  throw new Error(
    'Configure NVIDIA_API_KEY or GEMINI_API_KEY.'
  );
}

for (const providerUrl of [NVIDIA_URL, GEMINI_URL]) {
  if (!providerUrl.startsWith('https://')) {
    throw new Error('Provider URLs must use HTTPS.');
  }
}

// ============================================================
// MODEL CATALOG
// ============================================================

// These are configurable catalog entries.
// Availability depends on your provider account.

const NVIDIA_MODELS = list('NVIDIA_MODELS', [
  'mistralai/mistral-large-4-1t-a52b',
  'nvidia/nemotron-3.5-bolt-e4b',
  'minimaxai/minimax-m3.1-flash',
  'nvidia/nemotron-3.5-super-vl-120b-a12b',
  'nvidia/ising-calibration-1.5-31b',
  'stepfun-ai/step-5-preview',
  'z-ai/glm-5.3-flash',
  'poolside/laguna-xs-2.1'
]);

const GEMINI_MODELS = list('GEMINI_MODELS', [
  'gemini-3.8-flash',
  'gemini-3.6-flash',
  'gemini-3.5-flash-lite',
  'gemini-3.1-flash-lite',
  'gemini-2.5-pro',
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite'
]);

const DEFAULT_MODEL = env.DEFAULT_MODEL || (
  NVIDIA_KEY
    ? NVIDIA_MODELS[0]
    : `google/${GEMINI_MODELS[0]}`
);

// ============================================================
// MODEL RESOLUTION
// ============================================================

function resolveModel(requested) {
  if (!requested) {
    requested = DEFAULT_MODEL;
  }

  if (typeof requested !== 'string') {
    return null;
  }

  // Gemini format: google/gemini-model
  if (requested.startsWith('google/')) {
    const id = requested.slice(7);

    if (!GEMINI_KEY) return null;

    if (
      !ALLOW_CUSTOM_MODELS &&
      !GEMINI_MODELS.includes(id)
    ) {
      return null;
    }

    return {
      provider: 'gemini',
      model: id,
      id: requested
    };
  }

  // Alternative Gemini format: gemini/gemini-model
  if (requested.startsWith('gemini/')) {
    const id = requested.slice(7);

    if (!GEMINI_KEY) return null;

    if (
      !ALLOW_CUSTOM_MODELS &&
      !GEMINI_MODELS.includes(id)
    ) {
      return null;
    }

    return {
      provider: 'gemini',
      model: id,
      id: requested
    };
  }

  // NVIDIA models use their regular model IDs.
  if (!NVIDIA_KEY) return null;

  if (
    !ALLOW_CUSTOM_MODELS &&
    !NVIDIA_MODELS.includes(requested)
  ) {
    return null;
  }

  return {
    provider: 'nvidia',
    model: requested,
    id: requested
  };
}

// ============================================================
// CONCURRENCY QUEUE
// ============================================================

class Queue {
  constructor(limit, maxQueue, timeout) {
    this.limit = limit;
    this.maxQueue = maxQueue;
    this.timeout = timeout;
    this.active = 0;
    this.waiting = [];
  }

  acquire(signal) {
    if (signal.aborted) {
      return Promise.reject(
        new Error('Request aborted.')
      );
    }

    if (this.active < this.limit) {
      this.active++;

      return Promise.resolve(
        this.releaseFunction()
      );
    }

    if (this.waiting.length >= this.maxQueue) {
      const error = new Error('Provider queue is full.');
      error.status = 503;

      return Promise.reject(error);
    }

    return new Promise((resolve, reject) => {
      const entry = {
        settled: false,
        timer: null,
        abort: null,
        grant: null
      };

      const remove = () => {
        const index = this.waiting.indexOf(entry);

        if (index !== -1) {
          this.waiting.splice(index, 1);
        }
      };

      const cleanup = () => {
        clearTimeout(entry.timer);

        if (entry.abort) {
          signal.removeEventListener(
            'abort',
            entry.abort
          );
        }
      };

      const rejectEntry = error => {
        if (entry.settled) return;

        entry.settled = true;
        remove();
        cleanup();
        reject(error);
      };

      entry.abort = () => {
        rejectEntry(
          new Error('Request aborted.')
        );
      };

      entry.grant = () => {
        if (entry.settled) return;

        entry.settled = true;
        cleanup();

        resolve(this.releaseFunction());
      };

      entry.timer = setTimeout(() => {
        const error = new Error(
          'Provider queue timeout.'
        );

        error.status = 503;
        rejectEntry(error);
      }, this.timeout);

      entry.timer.unref?.();

      signal.addEventListener(
        'abort',
        entry.abort,
        { once: true }
      );

      this.waiting.push(entry);

      // Handle a signal that was aborted while registering.
      if (signal.aborted) {
        entry.abort();
      }
    });
  }

  releaseFunction() {
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
      this.waiting.length > 0 &&
      this.active < this.limit
    ) {
      const entry = this.waiting.shift();

      if (!entry || entry.settled) {
        continue;
      }

      this.active++;
      entry.grant();
    }
  }
}

const queues = {
  nvidia: new Queue(
    NVIDIA_CONCURRENCY,
    QUEUE_LIMIT,
    QUEUE_TIMEOUT
  ),

  gemini: new Queue(
    GEMINI_CONCURRENCY,
    QUEUE_LIMIT,
    QUEUE_TIMEOUT
  )
};

// ============================================================
// HTTP HELPERS
// ============================================================

function json(res, status, data) {
  if (res.destroyed || res.writableEnded) {
    return;
  }

  if (!res.headersSent) {
    res.writeHead(status, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store'
    });
  }

  res.end(JSON.stringify(data));
}

function errorResponse(
  res,
  status,
  message,
  type = 'proxy_error'
) {
  json(res, status, {
    error: {
      message,
      type
    }
  });
}

async function readBody(req) {
  const chunks = [];
  let size = 0;
  let tooLarge = false;

  for await (const chunk of req) {
    size += chunk.length;

    if (size > MAX_BODY) {
      tooLarge = true;
      chunks.length = 0;
      continue;
    }

    if (!tooLarge) {
      chunks.push(chunk);
    }
  }

  if (tooLarge) {
    const error = new Error('Request body too large.');
    error.status = 413;
    throw error;
  }

  try {
    return JSON.parse(
      Buffer.concat(chunks).toString('utf8')
    );
  } catch {
    const error = new Error('Invalid JSON body.');
    error.status = 400;
    throw error;
  }
}

// ============================================================
// AUTHENTICATION
// ============================================================

function authenticate(req, res) {
  // Proxy authentication is optional.
  if (!PROXY_KEY) return true;

  const match = String(
    req.headers.authorization || ''
  ).match(/^Bearer\s+(.+)$/i);

  const supplied = Buffer.from(match?.[1] || '');
  const expected = Buffer.from(PROXY_KEY);

  const valid =
    supplied.length === expected.length &&
    crypto.timingSafeEqual(supplied, expected);

  if (!valid) {
    errorResponse(
      res,
      401,
      'Invalid proxy API key.',
      'authentication_error'
    );

    return false;
  }

  return true;
}

// ============================================================
// PROVIDER CONFIGURATION
// ============================================================

function providerDetails(provider) {
  if (provider === 'gemini') {
    return {
      url: GEMINI_URL,
      key: GEMINI_KEY
    };
  }

  return {
    url: NVIDIA_URL,
    key: NVIDIA_KEY
  };
}

// ============================================================
// JANITORAI MESSAGE HANDLING
// ============================================================

// Do not add, remove, summarize, reorder, or rewrite messages.
// JanitorAI supplies the conversation and selected prompt.

function preserveMessages(messages) {
  return messages;
}

// ============================================================
// PROVIDER REQUEST ADAPTER
// ============================================================

function prepareRequest(input, profile) {
  const body = { ...input };

  // Translate the proxy model name into the provider model ID.
  body.model = profile.model;

  // Support both OpenAI token parameter names.
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

  // Keep JanitorAI's messages exactly as supplied.
  body.messages = preserveMessages(input.messages);

  // Other request parameters remain unchanged.
  // No server-generated prompts or prefill messages.

  return body;
}

// ============================================================
// RESPONSE HELPERS
// ============================================================

function removeReasoning(value) {
  if (Array.isArray(value)) {
    return value.map(removeReasoning);
  }

  if (!value || typeof value !== 'object') {
    return value;
  }

  const result = {};

  for (const [key, child] of Object.entries(value)) {
    if ([
      'reasoning',
      'reasoning_content',
      'internal_reasoning',
      'thinking'
    ].includes(key.toLowerCase())) {
      continue;
    }

    result[key] = removeReasoning(child);
  }

  return result;
}

async function readLimited(
  response,
  limit = 32 * 1024 * 1024
) {
  if (!response.body) {
    return {
      text: '',
      truncated: false
    };
  }

  const chunks = [];
  let total = 0;
  let truncated = false;

  for await (const chunk of response.body) {
    const buffer = Buffer.from(chunk);
    const remaining = limit - total;

    if (remaining <= 0) {
      truncated = true;
      break;
    }

    if (buffer.length > remaining) {
      chunks.push(buffer.subarray(0, remaining));
      total += remaining;
      truncated = true;
      break;
    }

    chunks.push(buffer);
    total += buffer.length;
  }

  return {
    text: Buffer.concat(chunks).toString('utf8'),
    truncated
  };
}

function parseResponse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// ============================================================
// STREAMING HELPERS
// ============================================================

function transformSSEFrame(frame) {
  const lines = frame.split(/\r?\n/);

  const dataIndexes = [];

  for (let i = 0; i < lines.length; i++) {
    if (/^data:/.test(lines[i])) {
      dataIndexes.push(i);
    }
  }

  // Standard OpenAI-compatible streaming uses one data line.
  if (dataIndexes.length !== 1) {
    return frame;
  }

  const index = dataIndexes[0];

  const raw = lines[index]
    .slice(5)
    .replace(/^ /, '');

  if (!raw || raw === '[DONE]') {
    return frame;
  }

  try {
    const parsed = JSON.parse(raw);
    const cleaned = removeReasoning(parsed);

    lines[index] = `data: ${JSON.stringify(cleaned)}`;

    return lines.join('\n');
  } catch {
    return frame;
  }
}

function waitForDrain(res, signal) {
  return new Promise(resolve => {
    let finished = false;

    const cleanup = () => {
      res.removeListener('drain', onDrain);
      res.removeListener('close', onClose);

      signal.removeEventListener(
        'abort',
        onAbort
      );
    };

    const done = result => {
      if (finished) return;

      finished = true;
      cleanup();
      resolve(result);
    };

    const onDrain = () => done(true);
    const onClose = () => done(false);
    const onAbort = () => done(false);

    res.once('drain', onDrain);
    res.once('close', onClose);

    signal.addEventListener(
      'abort',
      onAbort,
      { once: true }
    );

    if (res.destroyed || signal.aborted) {
      done(false);
    }
  });
}

async function writeStreamChunk(
  res,
  chunk,
  signal
) {
  if (res.destroyed || signal.aborted) {
    return false;
  }

  if (res.write(chunk)) {
    return true;
  }

  return waitForDrain(res, signal);
}

// ============================================================
// STREAMING RESPONSE
// ============================================================

async function streamResponse(
  upstream,
  res,
  signal
) {
  res.writeHead(upstream.status, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no'
  });

  res.flushHeaders?.();

  if (!upstream.body) {
    res.end();
    return;
  }

  try {
    // Normal streaming: forward provider chunks unchanged.
    if (!STRIP_REASONING) {
      for await (const chunk of upstream.body) {
        if (signal.aborted || res.destroyed) {
          return;
        }

        const written = await writeStreamChunk(
          res,
          Buffer.from(chunk),
          signal
        );

        if (!written) return;
      }
    } else {
      // Reasoning filtering uses a buffer so SSE frames
      // split across network chunks are not corrupted.

      const decoder = new TextDecoder();
      let buffer = '';

      for await (const chunk of upstream.body) {
        if (signal.aborted || res.destroyed) {
          return;
        }

        buffer += decoder.decode(chunk, {
          stream: true
        });

        while (true) {
          const match = /\r?\n\r?\n/.exec(buffer);

          if (!match) break;

          const frame = buffer.slice(0, match.index);
          const separator = match[0];

          buffer = buffer.slice(
            match.index + separator.length
          );

          const transformed = transformSSEFrame(frame);

          const written = await writeStreamChunk(
            res,
            Buffer.from(transformed + separator),
            signal
          );

          if (!written) return;
        }
      }

      buffer += decoder.decode();

      if (buffer.length > 0) {
        const transformed = transformSSEFrame(buffer);

        const written = await writeStreamChunk(
          res,
          Buffer.from(transformed),
          signal
        );

        if (!written) return;
      }
    }

    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }
  } catch (error) {
    console.error(
      '[STREAM ERROR]',
      error.message
    );

    if (!res.destroyed && !res.writableEnded) {
      res.destroy();
    }
  }
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

async function chatCompletions(
  req,
  res,
  requestId
) {
  const started = Date.now();

  const controller = new AbortController();

  let release = null;
  let timedOut = false;

  const timeout = setTimeout(() => {
    timedOut = true;

    controller.abort(
      new Error('Request timeout.')
    );
  }, TIMEOUT);

  timeout.unref?.();

  const onRequestAbort = () => {
    controller.abort(
      new Error('Client disconnected.')
    );
  };

  const onResponseClose = () => {
    if (!res.writableEnded) {
      controller.abort(
        new Error('Client disconnected.')
      );
    }
  };

  req.once('aborted', onRequestAbort);
  res.once('close', onResponseClose);

  try {
    const input = await readBody(req);

    if (!input || typeof input !== 'object') {
      return errorResponse(
        res,
        400,
        'Invalid request body.',
        'invalid_request_error'
      );
    }

    if (
      !Array.isArray(input.messages) ||
      input.messages.length === 0
    ) {
      return errorResponse(
        res,
        400,
        'A non-empty messages array is required.',
        'invalid_request_error'
      );
    }

    if (
      input.model !== undefined &&
      (
        typeof input.model !== 'string' ||
        !input.model.trim()
      )
    ) {
      return errorResponse(
        res,
        400,
        'Invalid model.',
        'invalid_request_error'
      );
    }

    const profile = resolveModel(
      input.model || DEFAULT_MODEL
    );

    if (!profile) {
      return errorResponse(
        res,
        400,
        `Unknown or unavailable model: ${
          input.model || DEFAULT_MODEL
        }`,
        'invalid_request_error'
      );
    }

    const body = prepareRequest(input, profile);
    const provider = providerDetails(profile.provider);

    // Wait for the provider concurrency slot.
    release = await queues[profile.provider].acquire(
      controller.signal
    );

    if (controller.signal.aborted) {
      return;
    }

    console.log(JSON.stringify({
      type: 'provider_request',
      requestId,
      provider: profile.provider,
      model: profile.model,
      stream: Boolean(body.stream)
    }));

    const upstream = await fetch(
      `${provider.url}/chat/completions`,
      {
        method: 'POST',

        headers: {
          Authorization: `Bearer ${provider.key}`,
          'Content-Type': 'application/json',

          Accept: body.stream
            ? 'text/event-stream'
            : 'application/json'
        },

        body: JSON.stringify(body),
        signal: controller.signal
      }
    );

    if (controller.signal.aborted) {
      return;
    }

    // Provider returned an error.
    if (!upstream.ok) {
      const result = await readLimited(
        upstream,
        1024 * 1024
      );

      const data = result.truncated
        ? null
        : parseResponse(result.text);

      const message =
        data?.error?.message ||
        data?.message ||
        result.text.slice(0, 3000) ||
        `Provider returned HTTP ${upstream.status}.`;

      console.error(JSON.stringify({
        type: 'provider_error',
        requestId,
        provider: profile.provider,
        model: profile.model,
        status: upstream.status,
        duration: Date.now() - started
      }));

      return errorResponse(
        res,
        upstream.status,
        message,
        'upstream_error'
      );
    }

    // Streaming response.
    if (body.stream) {
      await streamResponse(
        upstream,
        res,
        controller.signal
      );

      return;
    }

    // Normal JSON response.
    const result = await readLimited(upstream);

    if (result.truncated) {
      return errorResponse(
        res,
        502,
        'Provider response exceeded the response size limit.',
        'upstream_error'
      );
    }

    let data = parseResponse(result.text);

    if (data === null) {
      return errorResponse(
        res,
        502,
        'Provider returned invalid JSON.',
        'upstream_error'
      );
    }

    if (STRIP_REASONING) {
      data = removeReasoning(data);
    }

    json(res, upstream.status, data);

    console.log(JSON.stringify({
      type: 'provider_success',
      requestId,
      provider: profile.provider,
      model: profile.model,
      duration: Date.now() - started
    }));

  } catch (error) {
    if (controller.signal.aborted && !timedOut) {
      return;
    }

    if (res.destroyed || res.writableEnded) {
      return;
    }

    if (res.headersSent) {
      res.destroy();
      return;
    }

    if (timedOut) {
      console.error(JSON.stringify({
        type: 'provider_timeout',
        requestId,
        duration: Date.now() - started
      }));

      return errorResponse(
        res,
        504,
        'Provider request timed out.',
        'upstream_timeout'
      );
    }

    console.error(JSON.stringify({
      type: 'proxy_error',
      requestId,
      error: error.message,
      duration: Date.now() - started
    }));

    errorResponse(
      res,
      error.status || 502,
      error.message || 'Provider request failed.',
      'proxy_error'
    );

  } finally {
    clearTimeout(timeout);

    req.removeListener(
      'aborted',
      onRequestAbort
    );

    res.removeListener(
      'close',
      onResponseClose
    );

    release?.();
  }
}

// ============================================================
// MODEL LIST
// ============================================================

function modelList() {
  const models = [];

  const created = Math.floor(
    Date.now() / 1000
  );

  if (NVIDIA_KEY) {
    for (const id of NVIDIA_MODELS) {
      models.push({
        id,
        object: 'model',
        created,
        owned_by: 'nvidia'
      });
    }
  }

  if (GEMINI_KEY) {
    for (const id of GEMINI_MODELS) {
      models.push({
        id: `google/${id}`,
        object: 'model',
        created,
        owned_by: 'google'
      });
    }
  }

  return {
    object: 'list',
    data: models
  };
}

// ============================================================
// CORS
// ============================================================

function applyCors(req, res) {
  const origin = req.headers.origin;

  res.setHeader(
    'Access-Control-Allow-Origin',
    origin || '*'
  );

  if (origin) {
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
}

// ============================================================
// ROUTER
// ============================================================

const server = http.createServer(async (req, res) => {
  const requestId = crypto.randomUUID();
  const started = Date.now();

  let path = '/';

  try {
    path = new URL(
      req.url,
      'http://localhost'
    ).pathname;
  } catch {
    return errorResponse(
      res,
      400,
      'Invalid request URL.',
      'invalid_request_error'
    );
  }

  res.setHeader('X-Request-ID', requestId);
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // CORS headers must be added before handling routes.
  applyCors(req, res);

  // Log every request, without logging prompts or API keys.
  let logged = false;

  const logRequest = () => {
    if (logged) return;
    logged = true;

    console.log(JSON.stringify({
      type: 'http_request',
      requestId,
      method: req.method,
      path,
      status: res.statusCode,
      duration: Date.now() - started
    }));
  };

  res.once('finish', logRequest);
  res.once('close', logRequest);

  // Browser preflight must not require proxy authentication.
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // Authenticate API routes if PROXY_API_KEY is configured.
  if (
    path.startsWith('/v1/') &&
    !authenticate(req, res)
  ) {
    return;
  }

  // Root endpoint.
  if (
    req.method === 'GET' &&
    path === '/'
  ) {
    return json(res, 200, {
      name: 'JanitorAI Multi-Provider Proxy',
      status: 'online',

      providers: {
        nvidia: Boolean(NVIDIA_KEY),
        gemini: Boolean(GEMINI_KEY)
      },

      default_model: DEFAULT_MODEL
    });
  }

  // Health endpoint.
  if (
    req.method === 'GET' &&
    path === '/health'
  ) {
    return json(res, 200, {
      status: 'ok',

      providers: {
        nvidia: {
          configured: Boolean(NVIDIA_KEY),
          active: queues.nvidia.active,
          queued: queues.nvidia.waiting.length
        },

        gemini: {
          configured: Boolean(GEMINI_KEY),
          active: queues.gemini.active,
          queued: queues.gemini.waiting.length
        }
      },

      uptime_seconds: Math.floor(
        process.uptime()
      )
    });
  }

  // Model listing.
  if (
    req.method === 'GET' &&
    path === '/v1/models'
  ) {
    return json(res, 200, modelList());
  }

  // Chat completions.
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

  return errorResponse(
    res,
    404,
    'Endpoint not found.',
    'not_found'
  );
});

// ============================================================
// SERVER SETTINGS
// ============================================================

server.requestTimeout = 0;
server.timeout = 0;
server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;

server.on('error', error => {
  console.error('[SERVER ERROR]', error);
  process.exit(1);
});

// ============================================================
// START SERVER
// ============================================================

server.listen(PORT, '0.0.0.0', () => {
  console.log('');
  console.log('==========================================');
  console.log(' JANITORAI MULTI-PROVIDER PROXY');
  console.log('==========================================');

  console.log(`Port:               ${PORT}`);
  console.log(`NVIDIA:             ${Boolean(NVIDIA_KEY)}`);
  console.log(`Gemini:              ${Boolean(GEMINI_KEY)}`);
  console.log(`Default model:       ${DEFAULT_MODEL}`);

  console.log(`NVIDIA models:       ${NVIDIA_MODELS.length}`);
  console.log(`Gemini models:       ${GEMINI_MODELS.length}`);

  console.log(`NVIDIA concurrency:  ${NVIDIA_CONCURRENCY}`);
  console.log(`Gemini concurrency:  ${GEMINI_CONCURRENCY}`);

  console.log(`Queue limit:         ${QUEUE_LIMIT}`);
  console.log(`Request timeout:     ${TIMEOUT}ms`);

  console.log(`Strip reasoning:     ${STRIP_REASONING}`);
  console.log(`Proxy authentication: ${Boolean(PROXY_KEY)}`);

  console.log('Prompt injection:    DISABLED');
  console.log('==========================================');
  console.log('');
});

// ============================================================
// SHUTDOWN
// ============================================================

function shutdown() {
  console.log('[SERVER] Shutting down...');

  server.close(() => {
    process.exit(0);
  });

  setTimeout(() => {
    server.closeAllConnections?.();
    process.exit(1);
  }, 10000).unref();
}

process.once('SIGINT', shutdown);
process.once('SIGTERM', shutdown);
