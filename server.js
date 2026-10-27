"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");
const crypto = require("crypto");

// Create the Express app before declaring any routes or middleware.
const app = express();
app.disable("x-powered-by");

const PORT = Number(process.env.PORT) || 10000;

const NIM_API_BASE = (
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NIM_API_KEY =
  process.env.NIM_API_KEY ||
  process.env.NVIDIA_API_KEY ||
  "";

const PROXY_API_KEY = String(
  process.env.PROXY_API_KEY || ""
);

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "nvidia/nemotron-3-ultra-550b-a55b";

const DEBUG_PROXY = parseBoolean(
  process.env.DEBUG_PROXY,
  false
);

const ALLOW_UNKNOWN_MODELS = parseBoolean(
  process.env.ALLOW_UNKNOWN_MODELS,
  true
);

const STRIP_REASONING = parseBoolean(
  process.env.STRIP_REASONING_FROM_RESPONSE,
  true
);

const NIM_TIMEOUT_MS = positiveNumber(
  process.env.NIM_TIMEOUT_MS,
  900000
);

const MODEL_CACHE_TTL_MS = positiveNumber(
  process.env.MODEL_CACHE_TTL_MS,
  300000
);

const DEFAULT_MAX_TOKENS = positiveNumber(
  process.env.DEFAULT_MAX_TOKENS,
  16384
);

const DEFAULT_TEMPERATURE = numberOrDefault(
  process.env.DEFAULT_TEMPERATURE,
  1.0
);

const DEFAULT_TOP_P = numberOrDefault(
  process.env.DEFAULT_TOP_P,
  0.95
);

const DEFAULT_REASONING_EFFORT = String(
  process.env.DEFAULT_REASONING_EFFORT || "high"
).trim().toLowerCase();

const DEFAULT_REASONING_BUDGET = positiveNumber(
  process.env.DEFAULT_REASONING_BUDGET,
  16384
);

const DEFAULT_NEMOTRON_THINKING = parseBoolean(
  process.env.NEMOTRON_ENABLE_THINKING,
  true
);

const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 64,
  maxFreeSockets: 16
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 64,
  maxFreeSockets: 16
});

/* ============================================================
   BASIC HELPERS
============================================================ */

function parseBoolean(value, fallback) {
  if (value === undefined || value === null) {
    return fallback;
  }

  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return value !== 0;
  }

  const normalized = String(value)
    .trim()
    .toLowerCase();

  if (["true", "1", "yes", "on"].includes(normalized)) {
    return true;
  }

  if (["false", "0", "no", "off"].includes(normalized)) {
    return false;
  }

  return fallback;
}

function isObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

function isFiniteNumber(value) {
  if (typeof value === "number") {
    return Number.isFinite(value);
  }

  return (
    typeof value === "string" &&
    value.trim() !== "" &&
    Number.isFinite(Number(value))
  );
}

function numberOrDefault(value, fallback) {
  return isFiniteNumber(value)
    ? Number(value)
    : fallback;
}

function positiveNumber(value, fallback) {
  const number = Number(value);

  return (
    Number.isFinite(number) &&
    number > 0
  )
    ? number
    : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(
    maximum,
    Math.max(minimum, value)
  );
}

function normalizeModel(value) {
  return String(value || "")
    .trim()
    .toLowerCase();
}

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

/* ============================================================
   MODEL CONFIGURATION

   These entries are local proxy configuration, not a guarantee
   that every NVIDIA Free Endpoint remains available indefinitely.
   Check NVIDIA's current catalog if an upstream model returns 404.
============================================================ */

const MODELS = {
  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 1048576,
    adapter: "nemotron"
  },

  "nvidia/nemotron-3-super-120b-a12b": {
    name: "NVIDIA Nemotron 3 Super 120B - verify availability",
    provider: "NVIDIA",
    endpointStatus: "unverified-legacy",
    maxOutputTokens: 16384,
    contextTokens: 1048576,
    adapter: "generic"
  },

  "nvidia/nemotron-3.5-super-vl-120b-a12b": {
    name: "NVIDIA Nemotron 3.5 Super VL 120B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 262144,
    adapter: "generic"
  },

  "nvidia/nemotron-3.5-bolt-e4b": {
    name: "NVIDIA Nemotron 3.5 Bolt E4B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    adapter: "generic"
  },

  "nvidia/nemotron-3.5-lightning-30b-a3b": {
    name: "NVIDIA Nemotron 3.5 Lightning 30B A3B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 1048576,
    adapter: "generic"
  },

  "z-ai/glm-5.3": {
    name: "GLM 5.3 - check deprecation status",
    provider: "Z.ai",
    endpointStatus: "deprecation-announced",
    reasoningLevels: ["none", "low", "high", "max"],
    defaultReasoningEffort: "high",
    maxOutputTokens: 32768,
    contextTokens: 1048576,
    adapter: "glm53"
  },

  "z-ai/glm-5.3-flash": {
    name: "GLM 5.3 Flash",
    provider: "Z.ai",
    endpointStatus: "free-endpoint",
    reasoningLevels: ["low", "high", "max"],
    defaultReasoningEffort: "max",
    maxOutputTokens: 32768,
    contextTokens: 1048576,
    adapter: "glm53-flash"
  },

  "deepseek-ai/deepseek-v4.1-flash": {
    name: "DeepSeek V4.1 Flash",
    provider: "DeepSeek AI",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 1048576,
    adapter: "generic"
  },

  "stepfun-ai/step-5-preview": {
    name: "Step 5 Preview",
    provider: "StepFun AI",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 1048576,
    adapter: "generic"
  },

  "minimaxai/minimax-m3.1-flash": {
    name: "MiniMax M3.1 Flash",
    provider: "MiniMax AI",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 262144,
    adapter: "generic"
  },

  "poolside/laguna-xs-2.1": {
    name: "Laguna XS 2.1",
    provider: "Poolside",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 8192,
    contextTokens: 131072,
    adapter: "generic"
  },

  "openai/gpt-oss-20b": {
    name: "GPT-OSS 20B",
    provider: "OpenAI",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 8192,
    contextTokens: 131072,
    adapter: "generic"
  },

  "google/gemma-4-31b-it": {
    name: "Gemma 4 31B IT",
    provider: "Google",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 262144,
    adapter: "generic"
  },

  "meta/muse-glimmer-30b": {
    name: "Muse Glimmer 30B",
    provider: "Meta / Muse",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 8192,
    contextTokens: 131072,
    adapter: "generic"
  }
};

/* ============================================================
   MODEL PROFILES

   JanitorAI can select these profile IDs as if they were models.
   The proxy maps each profile to its actual NVIDIA model.
============================================================ */

const PROFILES = {
  "z-ai/glm-5.3-no": {
    baseModel: "z-ai/glm-5.3",
    label: "No Thinking",
    reasoningEffort: "none"
  },

  "z-ai/glm-5.3-fast": {
    baseModel: "z-ai/glm-5.3",
    label: "Fast",
    reasoningEffort: "low"
  },

  "z-ai/glm-5.3-balanced": {
    baseModel: "z-ai/glm-5.3",
    label: "Balanced",
    reasoningEffort: "high"
  },

  "z-ai/glm-5.3-deep": {
    baseModel: "z-ai/glm-5.3",
    label: "Deep",
    reasoningEffort: "max"
  },

  "z-ai/glm-5.3-flash-minimum": {
    baseModel: "z-ai/glm-5.3-flash",
    label: "Minimum Reasoning",
    reasoningEffort: "low"
  },

  "z-ai/glm-5.3-flash-fast": {
    baseModel: "z-ai/glm-5.3-flash",
    label: "Fast",
    reasoningEffort: "low"
  },

  "z-ai/glm-5.3-flash-balanced": {
    baseModel: "z-ai/glm-5.3-flash",
    label: "Balanced",
    reasoningEffort: "high"
  },

  "z-ai/glm-5.3-flash-deep": {
    baseModel: "z-ai/glm-5.3-flash",
    label: "Deep",
    reasoningEffort: "max"
  },

  "nvidia/nemotron-3-ultra-550b-a55b-no": {
    baseModel: "nvidia/nemotron-3-ultra-550b-a55b",
    label: "No Thinking",
    thinking: false
  },

  "nvidia/nemotron-3-ultra-550b-a55b-fast": {
    baseModel: "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Fast",
    thinking: false
  },

  "nvidia/nemotron-3-ultra-550b-a55b-balanced": {
    baseModel: "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Balanced",
    thinking: true,
    mediumEffort: true
  },

  "nvidia/nemotron-3-ultra-550b-a55b-deep": {
    baseModel: "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Deep",
    thinking: true,
    mediumEffort: false,
    thinkingTokenBudget: 32768
  }
};

/* ============================================================
   MODEL RESOLUTION
============================================================ */

function resolveModel(requestedModel) {
  const requested = normalizeModel(requestedModel);
  const profile = PROFILES[requested] || null;

  if (profile) {
    return {
      requestedModel: requested,
      model: profile.baseModel,
      profile,
      config: MODELS[profile.baseModel] || null
    };
  }

  if (MODELS[requested]) {
    return {
      requestedModel: requested,
      model: requested,
      profile: null,
      config: MODELS[requested]
    };
  }

  if (ALLOW_UNKNOWN_MODELS && requested) {
    return {
      requestedModel: requested,
      model: requested,
      profile: null,
      config: null
    };
  }

  return null;
}

function normalizeEffort(value, config) {
  const levels = Array.isArray(config?.reasoningLevels)
    ? config.reasoningLevels
    : ["none", "low", "medium", "high", "max"];

  const effort = String(value || "")
    .trim()
    .toLowerCase();

  if (levels.includes(effort)) {
    return effort;
  }

  if (["off", "disabled", "false", "none"].includes(effort)) {
    return levels.includes("none") ? "none" : levels[0];
  }

  if (["fast", "minimal"].includes(effort)) {
    return levels.includes("low") ? "low" : levels[0];
  }

  if (["balanced", "medium"].includes(effort)) {
    if (levels.includes("medium")) return "medium";
    if (levels.includes("high")) return "high";
    return levels[0];
  }

  if (effort === "deep") {
    if (levels.includes("max")) return "max";
    if (levels.includes("high")) return "high";
    return levels[0];
  }

  return config?.defaultReasoningEffort ||
    DEFAULT_REASONING_EFFORT;
}

function requestedEffort(body, resolution) {
  if (resolution.profile?.reasoningEffort) {
    return normalizeEffort(
      resolution.profile.reasoningEffort,
      resolution.config
    );
  }

  if (body.reasoning_effort !== undefined) {
    return normalizeEffort(
      body.reasoning_effort,
      resolution.config
    );
  }

  if (body.reasoning_mode !== undefined) {
    return normalizeEffort(
      body.reasoning_mode,
      resolution.config
    );
  }

  if (body.reasoning_budget !== undefined) {
    const budget = Number(body.reasoning_budget);

    if (!Number.isFinite(budget)) {
      return normalizeEffort("max", resolution.config);
    }

    if (budget <= 0) {
      return normalizeEffort("none", resolution.config);
    }

    if (budget <= 4096) {
      return normalizeEffort("low", resolution.config);
    }

    if (budget <= 16384) {
      return normalizeEffort("high", resolution.config);
    }

    return normalizeEffort("max", resolution.config);
  }

  return normalizeEffort(
    DEFAULT_REASONING_EFFORT,
    resolution.config
  );
}

/* ============================================================
   REQUEST CONSTRUCTION
============================================================ */

function buildBaseRequest(body, resolution) {
  const config = resolution.config;
  const requestedMax = Number(body.max_tokens);

  let maxTokens =
    Number.isFinite(requestedMax) && requestedMax > 0
      ? Math.floor(requestedMax)
      : config?.maxOutputTokens || DEFAULT_MAX_TOKENS;

  if (Number.isFinite(config?.maxOutputTokens)) {
    maxTokens = Math.min(maxTokens, config.maxOutputTokens);
  }

  const request = {
    model: resolution.model,

    messages: Array.isArray(body.messages)
      ? body.messages.map(message =>
          isObject(message) ? { ...message } : message
        )
      : [],

    max_tokens: maxTokens,

    temperature: clamp(
      numberOrDefault(body.temperature, DEFAULT_TEMPERATURE),
      0,
      2
    ),

    stream: Boolean(body.stream)
  };

  // Do not silently override model sampling defaults when top_p
  // was not provided by the client.
  if (body.top_p !== undefined) {
    request.top_p = clamp(
      numberOrDefault(body.top_p, DEFAULT_TOP_P),
      0,
      1
    );
  }

  const passthrough = [
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "tools",
    "tool_choice",
    "response_format",
    "logprobs",
    "top_k",
    "min_tokens",
    "ignore_eos",
    "logit_bias",
    "user",
    "parallel_tool_calls",
    "stream_options",
    "service_tier",
    "modalities",
    "audio"
  ];

  for (const key of passthrough) {
    if (body[key] !== undefined) {
      request[key] = body[key];
    }
  }

  if (isObject(body.chat_template_kwargs)) {
    request.chat_template_kwargs = {
      ...body.chat_template_kwargs
    };
  }

  if (isObject(body.extra_body)) {
    Object.assign(request, body.extra_body);
  }

  // The resolved model must remain authoritative.
  request.model = resolution.model;

  return request;
}

/* ============================================================
   MODEL-SPECIFIC ADAPTERS
============================================================ */

function buildNimRequest(body, resolution) {
  const request = buildBaseRequest(body, resolution);
  const model = resolution.model;
  const profile = resolution.profile;

  // GLM 5.3 and GLM 5.3 Flash.
  if (
    model === "z-ai/glm-5.3" ||
    model === "z-ai/glm-5.3-flash"
  ) {
    let effort = requestedEffort(body, resolution);

    // GLM 5.3 Flash does not expose a true no-reasoning profile.
    if (model === "z-ai/glm-5.3-flash" && effort === "none") {
      effort = "low";
    }

    request.reasoning_effort = effort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),
      clear_thinking: true
    };

    delete request.reasoning_budget;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  // NVIDIA Nemotron 3 Ultra.
  if (model === "nvidia/nemotron-3-ultra-550b-a55b") {
    let thinking = profile?.thinking;

    if (
      thinking === undefined &&
      body.enable_thinking !== undefined
    ) {
      thinking = parseBoolean(
        body.enable_thinking,
        DEFAULT_NEMOTRON_THINKING
      );
    }

    if (
      thinking === undefined &&
      typeof body.chat_template_kwargs?.enable_thinking === "boolean"
    ) {
      thinking = body.chat_template_kwargs.enable_thinking;
    }

    if (thinking === undefined) {
      thinking = DEFAULT_NEMOTRON_THINKING;
    }

    const template = isObject(request.chat_template_kwargs)
      ? { ...request.chat_template_kwargs }
      : {};

    for (const key of [
      "reasoning_effort",
      "reasoning_budget",
      "thinking_token_budget",
      "enable_thinking",
      "nvext",
      "medium_effort",
      "force_nonempty_content"
    ]) {
      delete template[key];
    }

    template.enable_thinking = Boolean(thinking);

    const mediumEffort =
      profile?.mediumEffort ??
      body.chat_template_kwargs?.medium_effort ??
      false;

    if (thinking && mediumEffort) {
      template.medium_effort = true;
    }

    request.chat_template_kwargs = template;

    const budgetInput =
      profile?.thinkingTokenBudget ??
      body.thinking_token_budget ??
      body.reasoning_budget;

    const budget = Number(budgetInput);

    if (thinking && Number.isFinite(budget) && budget > 0) {
      request.thinking_token_budget = clamp(
        Math.floor(budget),
        1,
        32768
      );
    } else {
      delete request.thinking_token_budget;
    }

    delete request.reasoning_effort;
    delete request.reasoning_budget;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.nvext;
  }

  // Other models receive the standard OpenAI-compatible request.
  // Avoid guessing vendor-specific reasoning parameters.
  return request;
}

/* ============================================================
   HEADERS AND LOGGING
============================================================ */

function buildHeaders(streaming) {
  const headers = {
    "Content-Type": "application/json",
    Accept: streaming
      ? "text/event-stream"
      : "application/json"
  };

  if (NIM_API_KEY) {
    headers.Authorization = `Bearer ${NIM_API_KEY}`;
  }

  return headers;
}

function debugLog(label, data) {
  if (!DEBUG_PROXY) return;

  // Do not log prompts, conversation history, or generated text.
  console.log(
    `[proxy:debug] ${label} ${JSON.stringify(data)}`
  );
}

/* ============================================================
   RESPONSE CLEANUP
============================================================ */

function stripThinkTags(text) {
  return String(text || "")
    .replace(
      /<think(?:ing)?\b[^>]*>[\s\S]*?<\/(?:think|thinking)>/gi,
      ""
    )
    .replace(
      /<\/?(?:think|thinking)\b[^>]*>/gi,
      ""
    );
}

function stripReasoningObject(data) {
  if (!data || typeof data !== "object") {
    return data;
  }

  if (Array.isArray(data)) {
    return data.map(stripReasoningObject);
  }

  delete data.reasoning_content;
  delete data.reasoning;
  delete data.thinking;

  if (typeof data.content === "string") {
    data.content = stripThinkTags(data.content);
  }

  for (const [key, value] of Object.entries(data)) {
    if (value && typeof value === "object") {
      data[key] = stripReasoningObject(value);
    }
  }

  return data;
}

/* ============================================================
   ERROR HANDLING
============================================================ */

function extractErrorBody(data) {
  if (Buffer.isBuffer(data)) {
    data = data.toString("utf8");
  }

  if (typeof data === "string") {
    return safeJsonParse(data) || {
      message: data.slice(0, 4000)
    };
  }

  if (isObject(data)) {
    return data;
  }

  return {
    message: String(data || "Upstream error")
  };
}

function makeProxyError(error, resolution, elapsedMs) {
  const status = Number(error?.response?.status) || 502;

  const upstream = extractErrorBody(
    error?.response?.data
  );

  const message =
    upstream?.error?.message ||
    upstream?.message ||
    error?.message ||
    "NVIDIA NIM request failed";

  return {
    error: {
      message: String(message),
      type: "upstream_error",
      code: status,
      model: resolution?.model || null,
      profile: resolution?.profile?.label || null,
      elapsed_ms: elapsedMs
    }
  };
}

/* ============================================================
   MODEL LIST
============================================================ */

function localModelList() {
  const created = Math.floor(Date.now() / 1000);

  const data = Object.entries(MODELS).map(([id, config]) => ({
    id,
    object: "model",
    created,
    owned_by: config.provider,
    name: config.name,
    endpoint_status: config.endpointStatus
  }));

  for (const [id, profile] of Object.entries(PROFILES)) {
    const config = MODELS[profile.baseModel];

    data.push({
      id,
      object: "model",
      created,
      owned_by: config?.provider || "proxy",
      name: `${config?.name || profile.baseModel} - ${profile.label}`,
      base_model: profile.baseModel
    });
  }

  return data;
}

let modelCache = {
  timestamp: 0,
  data: null
};

async function fetchRemoteModels() {
  if (
    modelCache.data &&
    Date.now() - modelCache.timestamp < MODEL_CACHE_TTL_MS
  ) {
    return modelCache.data;
  }

  try {
    const response = await axios.get(
      `${NIM_API_BASE}/models`,
      {
        headers: buildHeaders(false),
        timeout: 30000,
        httpAgent,
        httpsAgent,
        validateStatus: () => true
      }
    );

    if (
      response.status >= 200 &&
      response.status < 300 &&
      isObject(response.data)
    ) {
      modelCache = {
        timestamp: Date.now(),
        data: response.data
      };

      return response.data;
    }
  } catch (error) {
    debugLog("model_list_fetch_failed", {
      message: error.message
    });
  }

  return null;
}

/* ============================================================
   OPTIONAL PROXY AUTHENTICATION

   If PROXY_API_KEY is configured on Render, requests to /v1
   must include Authorization: Bearer YOUR_PROXY_API_KEY.
============================================================ */

function proxyAuth(req, res, next) {
  if (!PROXY_API_KEY) {
    return next();
  }

  const authorization = String(
    req.get("authorization") || ""
  );

  const match = authorization.match(
    /^Bearer\s+(.+)$/i
  );

  if (!match) {
    return res.status(401).json({
      error: {
        message: "Missing or invalid proxy API key",
        type: "authentication_error"
      }
    });
  }

  const supplied = Buffer.from(match[1], "utf8");
  const expected = Buffer.from(PROXY_API_KEY, "utf8");

  if (
    supplied.length !== expected.length ||
    !crypto.timingSafeEqual(supplied, expected)
  ) {
    return res.status(401).json({
      error: {
        message: "Missing or invalid proxy API key",
        type: "authentication_error"
      }
    });
  }

  return next();
}

/* ============================================================
   SERVER-SENT EVENTS HELPERS
============================================================ */

function writeSSEEvent(res, eventText) {
  const lines = eventText.split(/\r?\n/);

  const dataLines = [];
  let eventName = null;
  let eventId = null;

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("id:")) {
      eventId = line.slice(3).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (eventName) {
    res.write(`event: ${eventName}\n`);
  }

  if (eventId) {
    res.write(`id: ${eventId}\n`);
  }

  if (dataLines.length === 0) {
    res.write("\n");
    return;
  }

  const raw = dataLines.join("\n");

  if (raw === "[DONE]") {
    res.write("data: [DONE]\n\n");
    return;
  }

  const parsed = safeJsonParse(raw);

  if (!parsed) {
    for (const line of dataLines) {
      res.write(`data: ${line}\n`);
    }

    res.write("\n");
    return;
  }

  const cleaned = STRIP_REASONING
    ? stripReasoningObject(parsed)
    : parsed;

  res.write(`data: ${JSON.stringify(cleaned)}\n\n`);
}

function createSSEParser(onEvent) {
  let buffer = "";

  return {
    push(chunk) {
      buffer += chunk;

      let match;

      while ((match = /\r?\n\r?\n/.exec(buffer)) !== null) {
        const event = buffer.slice(0, match.index);

        buffer = buffer.slice(
          match.index + match[0].length
        );

        if (event.trim()) {
          onEvent(event);
        }
      }
    },

    end() {
      if (buffer.trim()) {
        onEvent(buffer);
      }

      buffer = "";
    }
  };
}

/* ============================================================
   EXPRESS MIDDLEWARE

   app is initialized above. Keep it above every app.use/app.get/
   app.post call; otherwise Node throws "app is not defined".
============================================================ */

app.use(cors({
  origin: true,
  credentials: true
}));

app.use(express.json({
  limit: process.env.JSON_BODY_LIMIT || "10mb"
}));

app.use("/v1", proxyAuth);

/* ============================================================
   ROOT AND HEALTH
============================================================ */

app.get("/", (req, res) => {
  res.json({
    name: "NVIDIA NIM OpenAI-Compatible Proxy",
    status: "ok",
    default_model: DEFAULT_MODEL,
    upstream: NIM_API_BASE,
    profile_count: Object.keys(PROFILES).length
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    upstream: NIM_API_BASE,
    default_model: DEFAULT_MODEL,
    api_key_configured: Boolean(NIM_API_KEY),
    proxy_auth_enabled: Boolean(PROXY_API_KEY)
  });
});

/* ============================================================
   MODEL ENDPOINTS
============================================================ */

app.get("/v1/models", async (req, res) => {
  const remote = await fetchRemoteModels();
  const local = localModelList();

  if (Array.isArray(remote?.data)) {
    const seen = new Set(
      remote.data
        .filter(isObject)
        .map(model => model.id)
    );

    const merged = [
      ...remote.data,
      ...local.filter(model => !seen.has(model.id))
    ];

    return res.json({
      object: "list",
      data: merged
    });
  }

  return res.json({
    object: "list",
    data: local
  });
});

app.post("/v1/models/refresh", async (req, res) => {
  modelCache = {
    timestamp: 0,
    data: null
  };

  const remote = await fetchRemoteModels();

  return res.json({
    object: "list",
    data: Array.isArray(remote?.data)
      ? remote.data
      : localModelList()
  });
});

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post("/v1/chat/completions", async (req, res) => {
  const startedAt = Date.now();

  const body = isObject(req.body)
    ? req.body
    : {};

  const requestedModel = normalizeModel(
    body.model || DEFAULT_MODEL
  );

  const resolution = resolveModel(requestedModel);

  if (!resolution) {
    return res.status(400).json({
      error: {
        message: `Unknown model: ${requestedModel}`,
        type: "invalid_request_error",
        code: "unknown_model"
      }
    });
  }

  if (!Array.isArray(body.messages)) {
    return res.status(400).json({
      error: {
        message: "The messages field must be an array",
        type: "invalid_request_error"
      }
    });
  }

  const upstreamBody = buildNimRequest(
    body,
    resolution
  );

  const streaming = Boolean(upstreamBody.stream);

  debugLog("request", {
    model: resolution.model,
    requested_model: requestedModel,
    profile: resolution.profile?.label || null,
    stream: streaming,
    message_count: body.messages.length,
    max_tokens: upstreamBody.max_tokens,
    temperature: upstreamBody.temperature,
    adapter: resolution.config?.adapter || "generic"
  });

  let upstream;

  try {
    upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      upstreamBody,
      {
        headers: buildHeaders(streaming),
        timeout: NIM_TIMEOUT_MS,
        httpAgent,
        httpsAgent,
        responseType: streaming ? "stream" : "json",
        validateStatus: () => true,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      }
    );
  } catch (error) {
    console.error(
      `[proxy] request failed for ${resolution.model}: ` +
      `${error?.message || "unknown error"}`
    );

    if (res.headersSent) {
      return res.end();
    }

    return res
      .status(Number(error?.response?.status) || 502)
      .json(
        makeProxyError(
          error,
          resolution,
          Date.now() - startedAt
        )
      );
  }

  // Forward upstream errors in a useful JSON format.
  if (upstream.status < 200 || upstream.status >= 300) {
    let data = upstream.data;

    if (data && typeof data.pipe === "function") {
      try {
        data = await new Promise((resolve, reject) => {
          const chunks = [];

          data.on("data", chunk => chunks.push(chunk));

          data.on("end", () => {
            resolve(
              Buffer.concat(chunks).toString("utf8")
            );
          });

          data.on("error", reject);
        });
      } catch {
        data = "NVIDIA NIM returned an unreadable error response";
      }
    }

    const errorObject = makeProxyError(
      {
        response: {
          status: upstream.status,
          data
        }
      },
      resolution,
      Date.now() - startedAt
    );

    console.error(
      `[proxy] NIM ${upstream.status} for ${resolution.model} ` +
      `profile=${resolution.profile?.label || "default"}`
    );

    debugLog("upstream_error", {
      status: upstream.status,
      model: resolution.model,
      message: errorObject.error.message
    });

    return res
      .status(upstream.status)
      .json(errorObject);
  }

  // Non-streaming response.
  if (!streaming) {
    const data = STRIP_REASONING
      ? stripReasoningObject(upstream.data)
      : upstream.data;

    return res
      .status(upstream.status)
      .json(data);
  }

  // Streaming response.
  res.status(upstream.status);

  res.setHeader(
    "Content-Type",
    "text/event-stream; charset=utf-8"
  );

  res.setHeader(
    "Cache-Control",
    "no-cache, no-transform"
  );

  res.setHeader(
    "Connection",
    "keep-alive"
  );

  res.setHeader(
    "X-Accel-Buffering",
    "no"
  );

  if (typeof res.flushHeaders === "function") {
    res.flushHeaders();
  }

  const stream = upstream.data;

  const parser = createSSEParser(event => {
    if (res.writableEnded) {
      return;
    }

    if (!STRIP_REASONING) {
      res.write(`${event}\n\n`);
      return;
    }

    writeSSEEvent(res, event);
  });

  stream.on("data", chunk => {
    if (!res.writableEnded) {
      parser.push(chunk.toString("utf8"));
    }
  });

  stream.on("end", () => {
    if (!res.writableEnded) {
      parser.end();
      res.end();
    }
  });

  stream.on("error", error => {
    console.error(
      `[proxy] upstream stream error for ${resolution.model}: ` +
      `${error?.message || "unknown error"}`
    );

    if (!res.writableEnded) {
      res.end();
    }
  });

  // Stop consuming the upstream stream if the client disconnects.
  res.on("close", () => {
    if (
      !res.writableEnded &&
      stream &&
      typeof stream.destroy === "function"
    ) {
      stream.destroy();
    }
  });
});

/* ============================================================
   404 AND GLOBAL ERROR HANDLING
============================================================ */

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Not found",
      type: "not_found"
    }
  });
});

app.use((error, req, res, next) => {
  console.error(
    `[proxy] unhandled error: ${error?.message || "unknown error"}`
  );

  if (res.headersSent) {
    return next(error);
  }

  return res.status(500).json({
    error: {
      message: error?.message || "Internal proxy error",
      type: "internal_error"
    }
  });
});

/* ============================================================
   START SERVER
============================================================ */

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`[proxy] listening on port ${PORT}`);
  console.log(`[proxy] upstream=${NIM_API_BASE}`);
  console.log(`[proxy] default_model=${DEFAULT_MODEL}`);
  console.log(`[proxy] NVIDIA API key configured=${Boolean(NIM_API_KEY)}`);
  console.log(`[proxy] proxy authentication enabled=${Boolean(PROXY_API_KEY)}`);
  console.log(`[proxy] reasoning stripping=${STRIP_REASONING}`);
  console.log(`[proxy] model profiles=${Object.keys(PROFILES).length}`);
});

/* ============================================================
   GRACEFUL SHUTDOWN
============================================================ */

function shutdown(signal) {
  console.log(`[proxy] received ${signal}; shutting down`);

  server.close(() => {
    httpAgent.destroy();
    httpsAgent.destroy();
    process.exit(0);
  });

  setTimeout(() => {
    process.exit(1);
  }, 10000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
