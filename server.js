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
  const value = env[name] === undefined ? fallback : Number(env[name]);
  if (!Number.isSafeInteger(value) || value < min) {
    throw new Error(`${name} must be an integer >= ${min}`);
  }
  return value;
}

function list(name, defaults) {
  if (!env[name]) return defaults;
  return [...new Set(env[name].split(',').map(x => x.trim()).filter(Boolean))];
}

const PORT = number('PORT', 3000);
if (PORT > 65535) throw new Error('Invalid PORT');

const NVIDIA_KEY = env.NVIDIA_API_KEY || '';
const GEMINI_KEY = env.GEMINI_API_KEY || '';

const NVIDIA_URL = (
  env.NVIDIA_BASE_URL || 'https://integrate.api.nvidia.com/v1'
).replace(/\/+$/, '');

const GEMINI_URL = (
  env.GEMINI_BASE_URL ||
  'https://generativelanguage.googleapis.com/v1beta/openai'
).replace(/\/+$/, '');

const DEFAULT_MAX_TOKENS = number('DEFAULT_MAX_TOKENS', 8192);
const MAX_BODY = number('MAX_BODY_BYTES', 20 * 1024 * 1024);
const TIMEOUT = number('REQUEST_TIMEOUT_MS', 600000, 1000);

const NVIDIA_CONCURRENCY = number('NVIDIA_MAX_CONCURRENT_REQUESTS', 2);
const GEMINI_CONCURRENCY = number('GEMINI_MAX_CONCURRENT_REQUESTS', 2);

const QUEUE_LIMIT = number('REQUEST_QUEUE_LIMIT', 8, 0);
const QUEUE_TIMEOUT = number('REQUEST_QUEUE_TIMEOUT_MS', 30000);

const STRIP_REASONING = env.STRIP_REASONING === 'true';
const ALLOW_CUSTOM_MODELS = env.ALLOW_CUSTOM_MODELS === 'true';

const PROXY_KEY = env.PROXY_API_KEY || '';

if (!NVIDIA_KEY && !GEMINI_KEY) {
  throw new Error('Configure NVIDIA_API_KEY or GEMINI_API_KEY.');
}

for (const url of [NVIDIA_URL, GEMINI_URL]) {
  if (!url.startsWith('https://')) {
    throw new Error('Provider URLs must use HTTPS.');
  }
}

// ============================================================
// MODEL CATALOG
// ============================================================

// These are configurable catalog entries, not a guarantee that
// every model is currently available to every provider account.

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

const DEFAULT_MODEL = env.DEFAULT_MODEL ||
  (NVIDIA_KEY
    ? NVIDIA_MODELS[0]
    : `google/${GEMINI_MODELS[0]}`);

function resolveModel(requested) {
  if (!requested) requested = DEFAULT_MODEL;

  if (typeof requested !== 'string') return null;

  if (requested.startsWith('google/')) {
    const id = requested.slice(7);

    if (!GEMINI_KEY) return null;
    if (!ALLOW_CUSTOM_MODELS && !GEMINI_MODELS.includes(id)) return null;

    return {
      provider: 'gemini',
      model: id,
      id: requested
    };
  }

  if (requested.startsWith('gemini/')) {
    const id = requested.slice(7);

    if (!GEMINI_KEY) return null;
    if (!ALLOW_CUSTOM_MODELS && !GEMINI_MODELS.includes(id)) return null;

    return {
      provider: 'gemini',
      model: id,
      id: requested
    };
  }

  if (!NVIDIA_KEY) return null;
  if (!ALLOW_CUSTOM_MODELS && !NVIDIA_MODELS.includes(requested)) {
    return null;
  }

  return {
    provider: 'nvidia',
    model: requested,
    id: requested
  };
}

// ============================================================
// CONCURRENCY MANAGEMENT
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
      return Promise.reject(new Error('Request aborted.'));
    }

    if (this.active < this.limit) {
      this.active++;
      return Promise.resolve(this.releaseFunction());
    }

    if (this.waiting.length >= this.maxQueue) {
      const error = new Error('Provider queue is full.');
      error.status = 503;
      return Promise.reject(error);
    }

    return new Promise((resolve, reject) => {
      const entry = { resolve, reject, timer: null };

      const remove = () => {
        const index = this.waiting.indexOf(entry);
        if (index !== -1) this.waiting.splice(index, 1);
      };

      const cleanup = () => {
        clearTimeout(entry.timer);
        signal.removeEventListener('abort', abort);
      };

      const abort = () => {
        remove();
        cleanup();
        reject(new Error('Request aborted.'));
      };

      entry.timer = setTimeout(() => {
        remove();
        cleanup();
        const error = new Error('Provider queue timeout.');
        error.status = 503;
        reject(error);
      }, this.timeout);

      entry.timer.unref?.();
      signal.addEventListener('abort', abort, { once: true });
      this.waiting.push(entry);
    });
  }

  releaseFunction() {
    let released = false;

    return () => {
      if (released) return;
      released = true;

      this.active--;

      while (this.waiting.length && this.active < this.limit) {
        const entry = this.waiting.shift();
        clearTimeout(entry.timer);
        this.active++;

        entry.resolve(this.releaseFunction());
      }
    };
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
  if (res.destroyed || res.writableEnded) return;

  if (!res.headersSent) {
    res.writeHead(status, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store'
    });
  }

  res.end(JSON.stringify(data));
}

function errorResponse(res, status, message, type = 'proxy_error') {
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

  for await (const chunk of req) {
    size += chunk.length;

    if (size > MAX_BODY) {
      const error = new Error('Request body too large.');
      error.status = 413;
      throw error;
    }

    chunks.push(chunk);
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    const error = new Error('Invalid JSON body.');
    error.status = 400;
    throw error;
  }
}

function authenticate(req, res) {
  if (!PROXY_KEY) return true;

  const match = String(req.headers.authorization || '')
    .match(/^Bearer\s+(.+)$/i);

  const supplied = Buffer.from(match?.[1] || '');
  const expected = Buffer.from(PROXY_KEY);

  const valid = supplied.length === expected.length &&
    crypto.timingSafeEqual(supplied, expected);

  if (!valid) {
    errorResponse(res, 401, 'Invalid proxy API key.', 'authentication_error');
    return false;
  }

  return true;
}

function providerDetails(provider) {
  return provider === 'gemini'
    ? { url: GEMINI_URL, key: GEMINI_KEY }
    : { url: NVIDIA_URL, key: NVIDIA_KEY };
}

// ============================================================
// MESSAGE HANDLING
// ============================================================

// IMPORTANT:
// This function intentionally does not insert, remove, reorder,
// summarize, or rewrite any messages supplied by JanitorAI.

function preserveMessages(messages) {
  return messages;
}

// ============================================================
// PROVIDER REQUEST ADAPTER
// ============================================================

function prepareRequest(input, profile) {
  const body = { ...input };

  // Only translate the requested proxy model name into the upstream ID.
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

  body.messages = preserveMessages(body.messages);

  // Preserve Gemini's documented reasoning controls when supplied by
  // JanitorAI or the calling client. Do not invent a reasoning prompt.
  //
  // Do not attach a server-generated prefill or safety-bypass instruction.
  //
  // Keep all other request parameters intact, allowing the upstream
  // provider to validate its own supported options.

  return body;
}

// ============================================================
// RESPONSE HELPERS
// ============================================================

function removeReasoning(value) {
  if (Array.isArray(value)) {
    return value.map(removeReasoning);
  }

  if (!value || typeof value !== 'object') return value;

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

async function readLimited(response, limit = 32 * 1024 * 1024) {
  if (!response.body) return '';

  const chunks = [];
  let total = 0;

  for await (const chunk of response.body) {
    const remaining = limit - total;
    if (remaining <= 0) break;

    const part = Buffer.from(chunk).subarray(0, remaining);
    chunks.push(part);
    total += part.length;

    if (total >= limit) break;
  }

  return Buffer.concat(chunks).toString('utf8');
}

function parseResponse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// ============================================================
// STREAMING
// ============================================================

async function streamResponse(upstream, res, signal) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no'
  });

  res.flushHeaders?.();

  try {
    for await (const chunk of upstream.body) {
      if (signal.aborted || res.destroyed) return;

      let output = Buffer.from(chunk);

      if (STRIP_REASONING) {
        const text = output.toString('utf8');

        // Only transform complete SSE data lines. Incomplete JSON
        // fragments are forwarded unchanged to avoid corrupting streams.
        const frames = text.split(/(?<=\n\n)/);

        output = Buffer.from(frames.map(frame => {
          const match = frame.match(/^data:\s*(\{.*\})\s*$/m);
          if (!match) return frame;

          try {
            const parsed = removeReasoning(JSON.parse(match[1]));
            return frame.replace(match[1], JSON.stringify(parsed));
          } catch {
            return frame;
          }
        }).join(''));
      }

      if (!res.write(output)) {
        await new Promise(resolve => {
          res.once('drain', resolve);
          res.once('close', resolve);
          signal.addEventListener('abort', resolve, { once: true });
        });
      }
    }

    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }
  } catch (error) {
    if (!res.destroyed && !res.writableEnded) {
      res.end();
    }
  }
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

async function chatCompletions(req, res) {
  const requestId = crypto.randomUUID();
  const started = Date.now();

  const controller = new AbortController();

  let release = null;
  let timedOut = false;

  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort(new Error('Request timeout.'));
  }, TIMEOUT);

  timeout.unref?.();

  const abort = () => controller.abort(new Error('Client disconnected.'));

  req.once('aborted', abort);

  res.once('close', () => {
    if (!res.writableEnded) abort();
  });

  try {
    const input = await readBody(req);

    if (!Array.isArray(input.messages) || input.messages.length === 0) {
      return errorResponse(
        res,
        400,
        'A non-empty messages array is required.',
        'invalid_request_error'
      );
    }

    if (input.model !== undefined &&
        (typeof input.model !== 'string' || !input.model.trim())) {
      return errorResponse(
        res, 400, 'Invalid model.', 'invalid_request_error'
      );
    }

    const profile = resolveModel(input.model || DEFAULT_MODEL);

    if (!profile) {
      return errorResponse(
        res,
        400,
        `Unknown or unavailable model: ${input.model || DEFAULT_MODEL}`,
        'invalid_request_error'
      );
    }

    const body = prepareRequest(input, profile);
    const provider = providerDetails(profile.provider);

    release = await queues[profile.provider].acquire(controller.signal);

    if (controller.signal.aborted) return;

    const upstream = await fetch(
      `${provider.url}/chat/completions`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${provider.key}`,
          'Content-Type': 'application/json',
          Accept: body.stream ? 'text/event-stream' : 'application/json'
        },
        body: JSON.stringify(body),
        signal: controller.signal
      }
    );

    if (controller.signal.aborted) return;

    if (!upstream.ok) {
      const text = await readLimited(upstream, 1024 * 1024);
      const data = parseResponse(text);

      const message =
        data?.error?.message ||
        data?.message ||
        text.slice(0, 3000) ||
        `Provider returned HTTP ${upstream.status}.`;

      console.error(JSON.stringify({
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

    if (body.stream) {
      await streamResponse(upstream, res, controller.signal);
      return;
    }

    const text = await readLimited(upstream);
    let result = parseResponse(text);

    if (result === null) {
      return errorResponse(
        res, 502, 'Provider returned invalid JSON.', 'upstream_error'
      );
    }

    if (STRIP_REASONING) {
      result = removeReasoning(result);
    }

    json(res, upstream.status, result);

  } catch (error) {
    if (controller.signal.aborted && !timedOut) return;

    if (timedOut) {
      return errorResponse(
        res, 504, 'Provider request timed out.', 'upstream_timeout'
      );
    }

    console.error(JSON.stringify({
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
    req.removeListener('aborted', abort);
    release?.();
  }
}

// ============================================================
// MODEL LIST
// ============================================================

function modelList() {
  const models = [];
  const created = Math.floor(Date.now() / 1000);

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
// ROUTER
// ============================================================

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const path = url.pathname;

  res.setHeader('X-Content-Type-Options', 'nosniff');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  if (path.startsWith('/v1/') && !authenticate(req, res)) {
    return;
  }

  if (req.method === 'GET' && path === '/') {
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

  if (req.method === 'GET' && path === '/health') {
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
      uptime_seconds: Math.floor(process.uptime())
    });
  }

  if (req.method === 'GET' && path === '/v1/models') {
    return json(res, 200, modelList());
  }

  if (req.method === 'POST' && path === '/v1/chat/completions') {
    return chatCompletions(req, res);
  }

  return errorResponse(res, 404, 'Endpoint not found.', 'not_found');
});

// ============================================================
// SERVER
// ============================================================

server.requestTimeout = 0;
server.timeout = 0;
server.headersTimeout = 65000;
server.keepAliveTimeout = 65000;

server.listen(PORT, '0.0.0.0', () => {
  console.log('');
  console.log('==========================================');
  console.log(' JANITORAI MULTI-PROVIDER PROXY');
  console.log('==========================================');
  console.log(`Port:              ${PORT}`);
  console.log(`NVIDIA:            ${Boolean(NVIDIA_KEY)}`);
  console.log(`Gemini:            ${Boolean(GEMINI_KEY)}`);
  console.log(`Default model:     ${DEFAULT_MODEL}`);
  console.log(`NVIDIA models:     ${NVIDIA_MODELS.length}`);
  console.log(`Gemini models:     ${GEMINI_MODELS.length}`);
  console.log(`NVIDIA concurrency:${NVIDIA_CONCURRENCY}`);
  console.log(`Gemini concurrency:${GEMINI_CONCURRENCY}`);
  console.log(`Request timeout:   ${TIMEOUT}ms`);
  console.log(`Strip reasoning:   ${STRIP_REASONING}`);
  console.log('Prompt injection:  DISABLED');
  console.log('==========================================');
  console.log('');
});

function shutdown() {
  console.log('Shutting down...');
  server.close(() => process.exit(0));

  setTimeout(() => {
    server.closeAllConnections?.();
    process.exit(1);
  }, 10000).unref();
}

process.once('SIGINT', shutdown);
process.once('SIGTERM', shutdown);
