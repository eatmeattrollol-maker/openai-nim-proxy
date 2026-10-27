
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");
const crypto = require("crypto");

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

const PROXY_API_KEY = String(process.env.PROXY_API_KEY || "");

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "nvidia/nemotron-3-ultra-550b-a55b";

const DEBUG_PROXY = parseBoolean(process.env.DEBUG_PROXY, false);
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
   HELPERS
============================================================ */

function parseBoolean(value, fallback) {
  if (value === undefined || value === null) return fallback;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;

  const normalized = String(value).trim().toLowerCase();

  if (["true", "1", "yes", "on"].includes(normalized)) return true;
  if (["false", "0", "no", "off"].includes(normalized)) return false;

  return fallback;
}

function isObject(value) {
  return value !== null &&
    typeof value === "object" &&
    !Array.isArray(value);
}

function isFiniteNumber(value) {
  if (typeof value === "number") return Number.isFinite(value);

  return typeof value === "string" &&
    value.trim() !== "" &&
    Number.isFinite(Number(value));
}

function numberOrDefault(value, fallback) {
  return isFiniteNumber(value) ? Number(value) : fallback;
}

function positiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0
    ? number
    : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizeModel(value) {
  return String(value || "").trim().toLowerCase();
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

   This configuration records the models configured in the
   proxy. It does not guarantee continued NVIDIA endpoint
   availability or free-tier access.
============================================================ */

const MODELS = {
  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 16384,
    contextTokens: 1048576,
    adapter: "nemotron",
    reasoningType: "thinking-toggle"
  },

  "nvidia/nemotron-3-super-120b-a12b": {
    name: "NVIDIA Nemotron 3 Super 120B - availability unverified",
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
    contextTokens: 1048576,
    adapter: "nemotron-vl",
    reasoningType: "thinking-toggle"
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
    name: "GLM 5.3",
    provider: "Z.ai",
    endpointStatus: "deprecation-announced",
    reasoningLevels: ["low", "high", "max"],
    defaultReasoningEffort: "high",
    maxOutputTokens: 32768,
    contextTokens: 1048576,
    adapter: "glm"
  },

  "z-ai/glm-5.3-flash": {
    name: "GLM 5.3 Flash",
    provider: "Z.ai",
    endpointStatus: "free-endpoint",
    reasoningLevels: ["low", "high", "max"],
    defaultReasoningEffort: "max",
    maxOutputTokens: 32768,
    contextTokens: 1048576,
    adapter: "glm-flash"
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
    contextTokens: 1048576,
    reasoningLevels: ["low", "medium", "high", "xhigh", "max"],
    defaultReasoningEffort: "high",
    adapter: "minimax"
  },

  "poolside/laguna-xs-2.1": {
    name: "Laguna XS 2.1",
    provider: "Poolside",
    endpointStatus: "free-endpoint",
    maxOutputTokens: 8192,
    contextTokens: 262144,
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
    reasoningLevels: ["low", "medium", "high", "xhigh"],
    defaultReasoningEffort: "high",
    adapter: "muse-glimmer"
  }
};

/* ============================================================
   PROFILE GENERATION

   Each profile maps a selectable JanitorAI model ID to its
   actual upstream model and documented reasoning control.

   Profiles intentionally do not set temperature or top_p.
   JanitorAI's values are passed through unchanged.
============================================================ */

const PROFILES = {};

function addEffortProfiles(modelId, levels) {
  const config = MODELS[modelId];

  for (const level of levels) {
    PROFILES[`${modelId}-${level}`] = {
      baseModel: modelId,
      label: `Reasoning: ${level}`,
      reasoningEffort: level
    };
  }
}

function addThinkingProfiles(modelId, options = {}) {
  PROFILES[`${modelId}-no-thinking`] = {
    baseModel: modelId,
    label: "Thinking Off",
    thinking: false
  };

  PROFILES[`${modelId}-thinking`] = {
    baseModel: modelId,
    label: "Thinking On",
    thinking: true
  };

  if (options.mediumEffort) {
    PROFILES[`${modelId}-medium-thinking`] = {
      baseModel: modelId,
      label: "Thinking On - Medium Effort",
      thinking: true,
      mediumEffort: true
    };
  }

  if (options.deepBudget) {
    PROFILES[`${modelId}-deep-thinking`] = {
      baseModel: modelId,
      label: "Thinking On - Larger Budget",
      thinking: true,
      thinkingTokenBudget: options.deepBudget
    };
  }
}

for (const [modelId, config] of Object.entries(MODELS)) {
  if (Array.isArray(config.reasoningLevels)) {
    addEffortProfiles(modelId, config.reasoningLevels);
  }
}

addThinkingProfiles(
  "nvidia/nemotron-3-ultra-550b-a55b",
  {
    mediumEffort: true,
    deepBudget: 32768
  }
);

addThinkingProfiles(
  "nvidia/nemotron-3.5-super-vl-120b-a12b"
);

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
    : [];

  const effort = String(value || "").trim().toLowerCase();

  if (levels.includes(effort)) return effort;

  if (["fast", "minimal", "minimum"].includes(effort)) {
    if (levels.includes("low")) return "low";
    return levels[0] || DEFAULT_REASONING_EFFORT;
  }

  if (["balanced", "medium"].includes(effort)) {
    if (levels.includes("medium")) return "medium";
    if (levels.includes("high")) return "high";
    return levels[0] || DEFAULT_REASONING_EFFORT;
  }

  if (["deep", "maximum"].includes(effort)) {
    if (levels.includes("max")) return "max";
    if (levels.includes("xhigh")) return "xhigh";
    if (levels.includes("high")) return "high";
    return levels[levels.length - 1] || DEFAULT_REASONING_EFFORT;
  }

  if (levels.includes(config?.defaultReasoningEffort)) {
    return config.defaultReasoningEffort;
  }

  if (levels.includes(DEFAULT_REASONING_EFFORT)) {
    return DEFAULT_REASONING_EFFORT;
  }

  return levels[0] || DEFAULT_REASONING_EFFORT;
}

function requestedEffort(body, resolution) {
  if (resolution.profile?.reasoningEffort) {
    return normalizeEffort(
      resolution.profile.reasoningEffort,
      resolution.config
    );
  }

  if (body.reasoning_effort !== undefined) {
    return normalizeEffort(body.reasoning_effort, resolution.config);
  }

  if (body.reasoning_mode !== undefined) {
    return normalizeEffort(body.reasoning_mode, resolution.config);
  }

  if (body.reasoning_budget !== undefined) {
    const budget = Number(body.reasoning_budget);

    if (Number.isFinite(budget)) {
      if (budget <= 4096) {
        return normalizeEffort("low", resolution.config);
      }

      if (budget <= 16384) {
        return normalizeEffort("high", resolution.config);
      }

      return normalizeEffort("max", resolution.config);
    }
  }

  return normalizeEffort(
    resolution.config?.defaultReasoningEffort ||
      DEFAULT_REASONING_EFFORT,
    resolution.config
  );
}

/* ============================================================
   REQUEST CONSTRUCTION

   Sampling controls are never invented by this proxy.
   If JanitorAI sends temperature/top_p, they are forwarded.
   If it omits them, they remain omitted.
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
    stream: Boolean(body.stream)
  };

  if (body.temperature !== undefined) {
    request.temperature = clamp(
      numberOrDefault(body.temperature, 1),
      0,
      2
    );
  }

  if (body.top_p !== undefined) {
    request.top_p = clamp(
      numberOrDefault(body.top_p, 0.95),
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
    "audio",
    "reasoning_effort",
    "reasoning_budget",
    "reasoning_mode",
    "thinking_token_budget",
    "enable_thinking",
    "nvext"
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

  // The selected model must always win over passthrough fields.
  request.model = resolution.model;

  return request;
}

/* ============================================================
   MODEL-SPECIFIC REQUEST ADAPTERS
============================================================ */

function buildNimRequest(body, resolution) {
  const request = buildBaseRequest(body, resolution);
  const model = resolution.model;
  const profile = resolution.profile;
  const adapter = resolution.config?.adapter || "generic";

  // GLM 5.3 and GLM 5.3 Flash.
  if (adapter === "glm" || adapter === "glm-flash") {
    const effort = requestedEffort(body, resolution);

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

  // MiniMax M3.1 Flash:
  // NVIDIA documents reasoning_effort inside chat_template_kwargs.
  if (adapter === "minimax") {
    const effort = requestedEffort(body, resolution);

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),
      reasoning_effort: effort
    };

    // Avoid sending the same control in a second, conflicting form.
    delete request.reasoning_effort;
    delete request.reasoning_mode;
    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  // Muse Glimmer:
  // NVIDIA documents low/medium/high/xhigh reasoning strengths.
  if (adapter === "muse-glimmer") {
    const effort = requestedEffort(body, resolution);

    request.reasoning_effort = effort;

    delete request.reasoning_mode;
    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  // NVIDIA Nemotron 3 Ultra.
  if (adapter === "nemotron") {
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

    return request;
  }

  // Nemotron 3.5 Super VL:
  // NVIDIA documents configurable thinking via chat_template_kwargs.
  if (adapter === "nemotron-vl") {
    let thinking = profile?.thinking;

    if (
      thinking === undefined &&
      body.enable_thinking !== undefined
    ) {
      thinking = parseBoolean(body.enable_thinking, true);
    }

    if (
      thinking === undefined &&
      typeof body.chat_template_kwargs?.enable_thinking === "boolean"
    ) {
      thinking = body.chat_template_kwargs.enable_thinking;
    }

    if (thinking !== undefined) {
      request.chat_template_kwargs = {
        ...(request.chat_template_kwargs || {}),
        enable_thinking: Boolean(thinking)
      };
    }

    delete request.enable_thinking;

    return request;
  }

  // Other models receive standard OpenAI-compatible parameters.
  // Their vendor-specific reasoning controls are not guessed here.
  return request;
}

/* ============================================================
   HEADERS AND LOGGING
============================================================ */

function buildHeaders(streaming) {
  const headers = {
    "Content-Type": "application/json",
    Accept: streaming ? "text/event-stream" : "application/json"
  };

  if (NIM_API_KEY) {
    headers.Authorization = `Bearer ${NIM_API_KEY}`;
  }

  return headers;
}

function debugLog(label, data) {
  if (!DEBUG_PROXY) return;

  // Never log prompts, generated text, authorization headers or keys.
  console.log(`[proxy:debug] ${label} ${JSON.stringify(data)}`);
}

function timingLog(stage, startedAt, metadata = {}) {
  debugLog("ttft", {
    stage,
    elapsed_ms: Date.now() - startedAt,
    ...metadata
  });
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
    .replace(/<\/?(?:think|thinking)\b[^>]*>/gi, "");
}

function stripReasoningObject(data) {
  if (!data || typeof data !== "object") return data;

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
    return safeJsonParse(data) || { message: data.slice(0, 4000) };
  }

  if (isObject(data)) return data;

  return { message: String(data || "Upstream error") };
}

function makeProxyError(error, resolution, elapsedMs) {
  const status = Number(error?.response?.status) || 502;
  const upstream = extractErrorBody(error?.response?.data);

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
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: buildHeaders(false),
      timeout: 30000,
      httpAgent,
      httpsAgent,
      validateStatus: () => true
    });

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
============================================================ */

function proxyAuth(req, res, next) {
  if (!PROXY_API_KEY) return next();

  const authorization = String(req.get("authorization") || "");
  const match = authorization.match(/^Bearer\s+(.+)$/i);

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
   SERVER-SENT EVENTS
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

  if (eventName) res.write(`event: ${eventName}\n`);
  if (eventId) res.write(`id: ${eventId}\n`);

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
    for (const line of dataLines) res.write(`data: ${line}\n`);
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

        buffer = buffer.slice(match.index + match[0].length);

        if (event.trim()) onEvent(event);
      }
    },

    end() {
      if (buffer.trim()) onEvent(buffer);
      buffer = "";
    }
  };
}

/* ============================================================
   EXPRESS MIDDLEWARE
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
      remote.data.filter(isObject).map(model => model.id)
    );

    return res.json({
      object: "list",
      data: [
        ...remote.data,
        ...local.filter(model => !seen.has(model.id))
      ]
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
  const local = localModelList();

  if (Array.isArray(remote?.data)) {
    const seen = new Set(
      remote.data.filter(isObject).map(model => model.id)
    );

    return res.json({
      object: "list",
      data: [
        ...remote.data,
        ...local.filter(model => !seen.has(model.id))
      ]
    });
  }

  return res.json({
    object: "list",
    data: local
  });
});

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post("/v1/chat/completions", async (req, res) => {
  const startedAt = Date.now();

  const body = isObject(req.body) ? req.body : {};
  const requestedModel = normalizeModel(body.model || DEFAULT_MODEL);
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

  const diagnosticMetadata = {
    model: resolution.model,
    requested_model: requestedModel,
    profile: resolution.profile?.label || null,
    reasoning_effort: resolution.profile?.reasoningEffort || null,
    thinking: resolution.profile?.thinking ?? null,
    stream: Boolean(body.stream)
  };

  timingLog("request_received", startedAt, diagnosticMetadata);

  if (!Array.isArray(body.messages)) {
    return res.status(400).json({
      error: {
        message: "The messages field must be an array",
        type: "invalid_request_error"
      }
    });
  }

  const upstreamBody = buildNimRequest(body, resolution);
  const streaming = Boolean(upstreamBody.stream);

  let firstUpstreamChunkLogged = false;
  let firstSSEEventLogged = false;
  let firstOutputLogged = false;

  debugLog("request", {
    model: resolution.model,
    requested_model: requestedModel,
    profile: resolution.profile?.label || null,
    reasoning_effort:
      upstreamBody.reasoning_effort ||
      upstreamBody.chat_template_kwargs?.reasoning_effort ||
      null,
    thinking_enabled:
      upstreamBody.chat_template_kwargs?.enable_thinking ?? null,
    stream: streaming,
    message_count: body.messages.length,
    max_tokens: upstreamBody.max_tokens,
    temperature: upstreamBody.temperature ?? null,
    top_p: upstreamBody.top_p ?? null,
    adapter: resolution.config?.adapter || "generic"
  });

  timingLog("upstream_request_started", startedAt, {
    ...diagnosticMetadata,
    message_count: body.messages.length,
    max_tokens: upstreamBody.max_tokens
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
    const elapsedMs = Date.now() - startedAt;

    timingLog("upstream_request_failed", startedAt, {
      ...diagnosticMetadata,
      elapsed_ms: elapsedMs,
      error_code: error?.code || null,
      error_message: error?.message || "unknown error"
    });

    console.error(
      `[proxy] request failed for ${resolution.model}: ` +
      `${error?.message || "unknown error"}`
    );

    if (res.headersSent) return res.end();

    return res
      .status(Number(error?.response?.status) || 502)
      .json(makeProxyError(error, resolution, elapsedMs));
  }

  timingLog("upstream_headers_received", startedAt, {
    ...diagnosticMetadata,
    upstream_status: upstream.status,
    content_type: upstream.headers?.["content-type"] || null
  });

  if (upstream.status < 200 || upstream.status >= 300) {
    let data = upstream.data;

    if (data && typeof data.pipe === "function") {
      try {
        data = await new Promise((resolve, reject) => {
          const chunks = [];

          data.on("data", chunk => chunks.push(chunk));
          data.on("end", () => {
            resolve(Buffer.concat(chunks).toString("utf8"));
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
      profile: resolution.profile?.label || null,
      message: errorObject.error.message
    });

    return res.status(upstream.status).json(errorObject);
  }

  if (!streaming) {
    const data = STRIP_REASONING
      ? stripReasoningObject(upstream.data)
      : upstream.data;

    timingLog("nonstream_response_forwarded", startedAt, {
      ...diagnosticMetadata,
      upstream_status: upstream.status
    });

    return res.status(upstream.status).json(data);
  }

  res.status(upstream.status);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  if (typeof res.flushHeaders === "function") {
    res.flushHeaders();
  }

  const stream = upstream.data;

  const parser = createSSEParser(event => {
    if (res.writableEnded) return;

    if (!firstSSEEventLogged) {
      firstSSEEventLogged = true;

      timingLog("first_sse_event", startedAt, {
        ...diagnosticMetadata,
        event_chars: event.length
      });
    }

    if (!STRIP_REASONING) {
      res.write(`${event}\n\n`);

      if (!firstOutputLogged) {
        firstOutputLogged = true;

        timingLog("first_output_forwarded", startedAt, {
          ...diagnosticMetadata,
          output_mode: "raw_sse"
        });
      }

      return;
    }

    writeSSEEvent(res, event);

    if (!firstOutputLogged) {
      firstOutputLogged = true;

      timingLog("first_output_forwarded", startedAt, {
        ...diagnosticMetadata,
        output_mode: "processed_sse"
      });
    }
  });

  stream.on("data", chunk => {
    if (!firstUpstreamChunkLogged) {
      firstUpstreamChunkLogged = true;

      timingLog("first_upstream_chunk", startedAt, {
        ...diagnosticMetadata,
        chunk_bytes: Buffer.isBuffer(chunk)
          ? chunk.length
          : Buffer.byteLength(String(chunk))
      });
    }

    if (!res.writableEnded) {
      parser.push(chunk.toString("utf8"));
    }
  });

  stream.on("end", () => {
    if (!res.writableEnded) {
      parser.end();
      res.end();

      timingLog("upstream_stream_ended", startedAt, {
        ...diagnosticMetadata,
        received_chunk: firstUpstreamChunkLogged,
        received_sse_event: firstSSEEventLogged,
        forwarded_output: firstOutputLogged
      });
    }
  });

  stream.on("error", error => {
    timingLog("upstream_stream_error", startedAt, {
      ...diagnosticMetadata,
      error_code: error?.code || null,
      error_message: error?.message || "unknown error"
    });

    console.error(
      `[proxy] upstream stream error for ${resolution.model}: ` +
      `${error?.message || "unknown error"}`
    );

    if (!res.writableEnded) res.end();
  });

  res.on("close", () => {
    if (
      !res.writableEnded &&
      stream &&
      typeof stream.destroy === "function"
    ) {
      timingLog("client_disconnected", startedAt, {
        ...diagnosticMetadata,
        received_chunk: firstUpstreamChunkLogged,
        received_sse_event: firstSSEEventLogged,
        forwarded_output: firstOutputLogged
      });

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

  if (res.headersSent) return next(error);

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
  console.log(`[proxy] diagnostic timing enabled=${DEBUG_PROXY}`);
});

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
