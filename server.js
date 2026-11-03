'use strict';

/*
 * JANITORAI MULTI-PROVIDER PROXY
 * NVIDIA NIM + Google Gemini
 *
 * Node.js 20+
 * No external dependencies.
 *
 * Endpoints:
 *   GET  /
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/chat/completions
 *
 * FEATURES:
 * - Accepts arbitrary NVIDIA model IDs.
 * - Accepts arbitrary Gemini model IDs.
 * - Dynamically retrieves provider model catalogs.
 * - Supports streaming and non-streaming responses.
 * - Supports concurrency queues.
 * - Supports optional proxy authentication.
 * - Supports CORS.
 *
 * IMPORTANT:
 * - No server-generated system prompts.
 * - No synthetic user messages.
 * - No conversation rewriting.
 * - No model allow-list.
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

function csv(name, fallback = []) {
  if (!env[name]) return fallback;

  return env[name]
    .split(',')
    .map(value => value.trim())
    .filter(Boolean);
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

const DEFAULT_MODEL = env.DEFAULT_MODEL || '';

const DEFAULT_MAX_TOKENS = number(
  'DEFAULT_MAX_TOKENS',
  8192
);

const MAX_BODY = number(
  'MAX_BODY_BYTES',
  20 * 1024 * 1024
);

const MAX_RESPONSE = number(
  'MAX_RESPONSE_BYTES',
  32 * 1024 * 1024
);

const TIMEOUT = number(
  'REQUEST_TIMEOUT_MS',
  600000,
  1000
);

const MODEL_LIST_TIMEOUT = number(
  'MODEL_LIST_TIMEOUT_MS',
  15000
);

const MODEL_CACHE_TIME = number(
  'MODEL_CACHE_SECONDS',
  60
) * 1000;

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

// Optional manually added model IDs.
// These only supplement the dynamic model list.
// They NEVER restrict which models can be requested.

const EXTRA_NVIDIA_MODELS = csv('EXTRA_NVIDIA_MODELS');
const EXTRA_GEMINI_MODELS = csv('EXTRA_GEMINI_MODELS');

// ============================================================
// MODEL ROUTING
// ============================================================

function resolveModel(requested) {
  if (typeof requested !== 'string' || !requested.trim()) {
    return null;
  }

  const original = requested.trim();

  // Explicit Gemini prefix.
  if (original.startsWith('gemini/')) {
    return {
      provider: 'gemini',
      model: original.slice(7),
      id: original
    };
  }

  // Backward compatibility with google/gemini-model.
  //
  // NVIDIA also has models with IDs beginning google/.
  // Therefore, only route google/ IDs containing Gemini
  // model names to Google. Other google/ IDs go to NVIDIA.

  if (original.startsWith('google/')) {
    const id = original.slice(7);

    if (
      id.startsWith('gemini-') ||
      id.startsWith('models/gemini-')
    ) {
      return {
        provider: 'gemini',
        model: id,
        id: original
      };
    }
  }

  // Anything else is an NVIDIA model ID.
  // The original ID is passed through unchanged.

  return {
    provider: 'nvidia',
    model: original,
    id: original
  };
}

// ============================================================
// PROVIDER CONFIGURATION
// ============================================================

function providerDetails(provider) {
  if (provider === 'gemini') {
    return {
      name: 'Gemini',
      url: GEMINI_URL,
      key: GEMINI_KEY
    };
  }

  return {
    name: 'NVIDIA',
    url: NVIDIA_URL,
    key: NVIDIA_KEY
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
        abort: null
      };

      const remove = () => {
        const index = this.waiting.indexOf(entry);

        if (index !== -1) {
          this.waiting.splice(index, 1);
        }
      };

      const cleanup = () => {
        clearTimeout(entry.timer);

        signal.removeEventListener(
          'abort',
          entry.abort
        );
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

      this.active = Math.max(
        0,
        this.active - 1
      );

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
    const error = new Error(
      'Request body too large.'
    );

    error.status = 413;

    throw error;
  }

  try {
    return JSON.parse(
      Buffer.concat(chunks).toString('utf8')
    );
  } catch {
    const error = new Error(
      'Invalid JSON body.'
    );

    error.status = 400;

    throw error;
  }
}

async function readLimited(
  response,
  limit = MAX_RESPONSE
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
      chunks.push(
        buffer.subarray(0, remaining)
      );

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
// AUTHENTICATION
// ============================================================

function authenticate(req, res) {
  if (!PROXY_KEY) return true;

  const match = String(
    req.headers.authorization || ''
  ).match(/^Bearer\s+(.+)$/i);

  const supplied = Buffer.from(
    match?.[1] || ''
  );

  const expected = Buffer.from(PROXY_KEY);

  const valid =
    supplied.length === expected.length &&
    crypto.timingSafeEqual(
      supplied,
      expected
    );

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
// DYNAMIC MODEL DISCOVERY
// ============================================================

// Fetch real models directly from each provider.
//
// NVIDIA:
//   GET {NVIDIA_BASE_URL}/models
//
// Gemini:
//   GET {GEMINI_BASE_URL}/models
//
// Gemini models are prefixed with gemini/ in our response
// so the proxy can distinguish them from NVIDIA model IDs.
//
// No hard-coded allow-list is used.

let modelCache = null;
let modelCacheTime = 0;
let modelFetchPromise = null;

async function fetchProviderModels(providerName) {
  const provider = providerDetails(providerName);

  if (!provider.key) {
    return [];
  }

  const controller = new AbortController();

  const timer = setTimeout(() => {
    controller.abort();
  }, MODEL_LIST_TIMEOUT);

  timer.unref?.();

  try {
    const response = await fetch(
      `${provider.url}/models`,
      {
        method: 'GET',

        headers: {
          Authorization: `Bearer ${provider.key}`,
          Accept: 'application/json'
        },

        signal: controller.signal
      }
    );

    const result = await readLimited(
      response,
      5 * 1024 * 1024
    );

    if (!response.ok) {
      throw new Error(
        `${provider.name} model listing returned HTTP ${response.status}: ` +
        result.text.slice(0, 1000)
      );
    }

    if (result.truncated) {
      throw new Error(
        `${provider.name} model list exceeded the size limit.`
      );
    }

    const parsed = parseResponse(result.text);

    if (!Array.isArray(parsed?.data)) {
      throw new Error(
        `${provider.name} returned an unexpected model-list response.`
      );
    }

    const models = [];

    for (const item of parsed.data) {
      if (
        !item ||
        typeof item.id !== 'string' ||
        !item.id.trim()
      ) {
        continue;
      }

      const id = item.id.trim();

      if (providerName === 'gemini') {
        models.push({
          ...item,
          id: id.startsWith('gemini/')
            ? id
            : `gemini/${id}`,
          owned_by: 'google'
        });
      } else {
        models.push({
          ...item,
          id,
          owned_by: item.owned_by || 'nvidia'
        });
      }
    }

    return models;

  } finally {
    clearTimeout(timer);
  }
}

async function buildModelList() {
  const providers = [];

  if (NVIDIA_KEY) {
    providers.push('nvidia');
  }

  if (GEMINI_KEY) {
    providers.push('gemini');
  }

  const results = await Promise.allSettled(
    providers.map(fetchProviderModels)
  );

  const models = [];
  const errors = [];

  for (let i = 0; i < results.length; i++) {
    const result = results[i];
    const provider = providers[i];

    if (result.status === 'fulfilled') {
      models.push(...result.value);
    } else {
      errors.push(
        `${provider}: ${result.reason?.message || 'Model listing failed.'}`
      );

      console.error(
        '[MODEL DISCOVERY ERROR]',
        provider,
        result.reason?.message
      );
    }
  }

  // Optional manually supplied IDs supplement discovered models.
  for (const id of EXTRA_NVIDIA_MODELS) {
    models.push({
      id,
      object: 'model',
      owned_by: 'nvidia'
    });
  }

  for (const id of EXTRA_GEMINI_MODELS) {
    models.push({
      id: id.startsWith('gemini/')
        ? id
        : `gemini/${id}`,
      object: 'model',
      owned_by: 'google'
    });
  }

  // Remove duplicate model IDs.
  const unique = new Map();

  for (const model of models) {
    unique.set(model.id, model);
  }

  if (providers.length > 0 && errors.length === providers.length) {
    throw new Error(
      `Unable to retrieve models from configured providers: ${errors.join(' | ')}`
    );
  }

  return {
    object: 'list',
    data: [...unique.values()]
  };
}

async function getModelList(forceRefresh = false) {
  const now = Date.now();

  if (
    !forceRefresh &&
    modelCache &&
    now - modelCacheTime < MODEL_CACHE_TIME
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
    modelCacheTime = Date.now();

    return result;

  } finally {
    modelFetchPromise = null;
  }
}

// ============================================================
// JANITORAI MESSAGES
// ============================================================

// Preserve the actual messages JanitorAI provides.
// No messages are inserted, deleted, summarized, or reordered.

function preserveMessages(messages) {
  return messages;
}

// ============================================================
// REQUEST PREPARATION
// ============================================================

function prepareRequest(input, profile) {
  const body = { ...input };

  // Only translate the provider routing prefix.
  body.model = profile.model;

  // Handle OpenAI token parameter compatibility.
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

  body.messages = preserveMessages(input.messages);

  // Everything else supplied by JanitorAI remains untouched.

  return body;
}

// ============================================================
// REASONING FILTER
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

// ============================================================
// STREAMING
// ============================================================

function transformSSEFrame(frame) {
  const lines = frame.split(/\r?\n/);

  const indexes = [];

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('data:')) {
      indexes.push(i);
    }
  }

  if (indexes.length !== 1) {
    return frame;
  }

  const index = indexes[0];

  const raw = lines[index]
    .slice(5)
    .replace(/^ /, '');

  if (!raw || raw === '[DONE]') {
    return frame;
  }

  try {
    const parsed = JSON.parse(raw);

    lines[index] = (
      'data: ' +
      JSON.stringify(removeReasoning(parsed))
    );

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

    const done = value => {
      if (finished) return;

      finished = true;
      cleanup();
      resolve(value);
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

async function writeStreamChunk(res, chunk, signal) {
  if (res.destroyed || signal.aborted) {
    return false;
  }

  if (res.write(chunk)) {
    return true;
  }

  return waitForDrain(res, signal);
}

async function streamResponse(upstream, res, signal) {
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
    // Default: forward the provider stream without modification.
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
      // Buffer complete SSE frames before filtering.
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

          const frame = buffer.slice(
            0,
            match.index
          );

          const separator = match[0];

          buffer = buffer.slice(
            match.index + separator.length
          );

          const output = transformSSEFrame(frame);

          const written = await writeStreamChunk(
            res,
            Buffer.from(output + separator),
            signal
          );

          if (!written) return;
        }
      }

      buffer += decoder.decode();

      if (buffer.length > 0) {
        const output = transformSSEFrame(buffer);

        const written = await writeStreamChunk(
          res,
          Buffer.from(output),
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

async function chatCompletions(req, res, requestId) {
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

    const requestedModel = (
      input.model || DEFAULT_MODEL
    );

    if (
      typeof requestedModel !== 'string' ||
      !requestedModel.trim()
    ) {
      return errorResponse(
        res,
        400,
        'No model was specified. Set DEFAULT_MODEL or send a model in the request.',
        'invalid_request_error'
      );
    }

    // Resolve provider without checking a model allow-list.
    const profile = resolveModel(requestedModel);

    if (!profile || !profile.model) {
      return errorResponse(
        res,
        400,
        'Invalid model identifier.',
        'invalid_request_error'
      );
    }

    const provider = providerDetails(
      profile.provider
    );

    if (!provider.key) {
      return errorResponse(
        res,
        503,
        `${provider.name} is not configured. Add its API key to the server environment.`,
        'provider_not_configured'
      );
    }

    const body = prepareRequest(
      input,
      profile
    );

    console.log(JSON.stringify({
      type: 'generation_started',
      requestId,
      provider: profile.provider,
      model: profile.model,
      streaming: Boolean(body.stream)
    }));

    // Acquire concurrency slot.
    release = await queues[profile.provider].acquire(
      controller.signal
    );

    if (controller.signal.aborted) {
      return;
    }

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

    // Upstream error.
    if (!upstream.ok) {
      const result = await readLimited(
        upstream,
        1024 * 1024
      );

      const parsed = result.truncated
        ? null
        : parseResponse(result.text);

      const message =
        parsed?.error?.message ||
        parsed?.message ||
        result.text.slice(0, 3000) ||
        `Provider returned HTTP ${upstream.status}.`;

      console.error(JSON.stringify({
        type: 'generation_failed',
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

      console.log(JSON.stringify({
        type: 'generation_finished',
        requestId,
        provider: profile.provider,
        model: profile.model,
        duration: Date.now() - started
      }));

      return;
    }

    // Normal JSON response.
    const result = await readLimited(upstream);

    if (result.truncated) {
      return errorResponse(
        res,
        502,
        'Provider response exceeded the configured size limit.',
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
      type: 'generation_finished',
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
      console.error(
        '[TIMEOUT]',
        requestId,
        profileSafeModel(requestId),
      );

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

function profileSafeModel(requestId) {
  return requestId;
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

  let path;

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

  applyCors(req, res);

  // Log requests without exposing prompts or credentials.
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

  // Always allow CORS preflight.
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // Authenticate API routes if configured.
  if (
    path.startsWith('/v1/') &&
    !authenticate(req, res)
  ) {
    return;
  }

  // Root.
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

      default_model: DEFAULT_MODEL || null,
      model_restrictions: false
    });
  }

  // Health.
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

  // Dynamic model catalog.
  if (
    req.method === 'GET' &&
    path === '/v1/models'
  ) {
    try {
      const forceRefresh =
        new URL(req.url, 'http://localhost')
          .searchParams.get('refresh') === 'true';

      const models = await getModelList(
        forceRefresh
      );

      return json(res, 200, models);

    } catch (error) {
      console.error(
        '[MODEL LIST ERROR]',
        error.message
      );

      return errorResponse(
        res,
        502,
        error.message,
        'model_discovery_error'
      );
    }
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

  console.log(`Port:                ${PORT}`);
  console.log(`NVIDIA:              ${Boolean(NVIDIA_KEY)}`);
  console.log(`Gemini:               ${Boolean(GEMINI_KEY)}`);

  console.log(
    `Default model:        ${DEFAULT_MODEL || 'Request must specify model'}`
  );

  console.log(`Model restrictions:   DISABLED`);
  console.log(`Dynamic model list:   ENABLED`);

  console.log(`NVIDIA concurrency:   ${NVIDIA_CONCURRENCY}`);
  console.log(`Gemini concurrency:    ${GEMINI_CONCURRENCY}`);

  console.log(`Queue limit:          ${QUEUE_LIMIT}`);
  console.log(`Request timeout:      ${TIMEOUT}ms`);

  console.log(`Strip reasoning:      ${STRIP_REASONING}`);
  console.log(`Proxy authentication: ${Boolean(PROXY_KEY)}`);

  console.log('Prompt injection:     DISABLED');
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
