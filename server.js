
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");
const crypto = require("crypto");
const { StringDecoder } = require("string_decoder");

const app = express();
app.disable("x-powered-by");
app.disable("etag");

const PORT = positiveNumber(process.env.PORT, 10000);
const NODE_ENV = process.env.NODE_ENV || "development";
const IS_PRODUCTION = NODE_ENV === "production";

const NIM_API_BASE = String(
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NIM_API_KEY = String(
  process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY || ""
);

const PROXY_API_KEY = String(process.env.PROXY_API_KEY || "");

const DEFAULT_MODEL = String(
  process.env.DEFAULT_MODEL || "nvidia/nemotron-3-ultra-550b-a55b"
).trim().toLowerCase();

const DEBUG_PROXY = parseBoolean(process.env.DEBUG_PROXY, false);
const STRIP_REASONING = parseBoolean(
  process.env.STRIP_REASONING_FROM_RESPONSE,
  true
);
const ALLOW_UNKNOWN_MODELS = parseBoolean(
  process.env.ALLOW_UNKNOWN_MODELS,
  false
);
const ENABLE_UNVERIFIED_MISTRAL_LARGE_4 = parseBoolean(
  process.env.ENABLE_UNVERIFIED_MISTRAL_LARGE_4,
  false
);

const NIM_TIMEOUT_MS = positiveNumber(
  process.env.NIM_TIMEOUT_MS,
  600000
);
const MODEL_CACHE_TTL_MS = positiveNumber(
  process.env.MODEL_CACHE_TTL_MS,
  300000
);
const DEFAULT_MAX_TOKENS = positiveNumber(
  process.env.DEFAULT_MAX_TOKENS,
  8192
);
const HARD_MAX_OUTPUT_TOKENS = positiveNumber(
  process.env.HARD_MAX_OUTPUT_TOKENS,
  16384
);
const DEFAULT_REASONING_EFFORT = String(
  process.env.DEFAULT_REASONING_EFFORT || "high"
).trim().toLowerCase();
const DEFAULT_REASONING_BUDGET = positiveNumber(
  process.env.DEFAULT_REASONING_BUDGET,
  8192
);
const DEFAULT_NEMOTRON_THINKING = parseBoolean(
  process.env.NEMOTRON_ENABLE_THINKING,
  true
);

const MAX_CONCURRENT_REQUESTS = positiveNumber(
  process.env.MAX_CONCURRENT_REQUESTS,
  4
);
const MAX_MESSAGES = positiveNumber(
  process.env.MAX_MESSAGES,
  200
);
const MAX_BODY_BYTES = process.env.JSON_BODY_LIMIT || "4mb";
const MAX_UPSTREAM_ERROR_BYTES = 1024 * 1024;

const CORS_ORIGINS = String(process.env.CORS_ORIGINS || "")
  .split(",")
  .map(value => value.trim())
  .filter(Boolean);

const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 16,
  maxFreeSockets: 4
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 16,
  maxFreeSockets: 4
});

let activeRequests = 0;
let shuttingDown = false;

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

function positiveNumber(value, fallback) {
  const number = Number(value);

  return Number.isFinite(number) && number > 0
    ? number
    : fallback;
}

function numberOrDefault(value, fallback) {
  return isFiniteNumber(value) ? Number(value) : fallback;
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

function debugLog(label, data = {}) {
  if (!DEBUG_PROXY) return;

  // Never log API keys, prompts, or generated text.
  console.log(`[proxy:debug] ${label} ${JSON.stringify(data)}`);
}

function timingLog(stage, startedAt, metadata = {}) {
  debugLog(stage, {
    elapsed_ms: Date.now() - startedAt,
    ...metadata
  });
}

/* ============================================================
   MODEL CONFIGURATION

   Catalog labels are metadata, not live availability checks.
   Verify the exact endpoint in your own NVIDIA account.
============================================================ */

const MODELS = {
  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "Nemotron 3 Ultra 550B",
    provider: "NVIDIA",
    endpointStatus: "downloadable-free-endpoint",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    adapter: "nemotron",
    reasoningType: "thinking-toggle"
  },

  "nvidia/nemotron-3-super-120b-a12b": {
    name: "Nemotron 3 Super 120B",
    provider: "NVIDIA",
    endpointStatus: "unverified-legacy",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    adapter: "generic"
  },

  "nvidia/nemotron-3.5-super-vl-120b-a12b": {
    name: "Nemotron 3.5 Super VL 120B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: 262144,
    maxOutputTokens: 16384,
    adapter: "nemotron-vl",
    reasoningType: "thinking-toggle"
  },

  "nvidia/nemotron-3.5-bolt-e4b": {
    name: "Nemotron 3.5 Bolt E4B",
    provider: "NVIDIA",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: null,
    maxOutputTokens: 8192,
    adapter: "generic"
  },

  "nvidia/nemotron-3.5-lightning-30b-a3b": {
    name: "Nemotron 3.5 Lightning 30B A3B",
    provider: "NVIDIA",
    endpointStatus: "downloadable-free-endpoint",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    adapter: "generic"
  },

  "z-ai/glm-5.3-flash": {
    name: "GLM 5.3 Flash",
    provider: "Z.ai",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 32768,
    reasoningLevels: ["low", "high", "max"],
    defaultReasoningEffort: "max",
    adapter: "glm-flash"
  },

  "deepseek-ai/deepseek-v4.1-flash": {
    name: "DeepSeek V4.1 Flash",
    provider: "DeepSeek AI",
    endpointStatus: "downloadable-free-endpoint",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    adapter: "generic"
  },

  "stepfun-ai/step-5-preview": {
    name: "Step 5 Preview",
    provider: "StepFun AI",
    endpointStatus: "free-endpoint-preview",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    adapter: "generic"
  },

  "minimaxai/minimax-m3.1-flash": {
    name: "MiniMax M3.1 Flash",
    provider: "MiniMax AI",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    reasoningLevels: ["low", "medium", "high", "xhigh", "max"],
    defaultReasoningEffort: "high",
    adapter: "minimax"
  },

  "poolside/laguna-xs-2.1": {
    name: "Laguna XS 2.1",
    provider: "Poolside",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: 262144,
    maxOutputTokens: 8192,
    adapter: "generic"
  },

  "openai/gpt-oss-20b": {
    name: "GPT-OSS 20B",
    provider: "OpenAI",
    endpointStatus: "downloadable-free-endpoint",
    enabled: true,
    contextTokens: 131072,
    maxOutputTokens: 8192,
    adapter: "generic"
  },

  "google/gemma-4-31b-it": {
    name: "Gemma 4 31B IT",
    provider: "Google",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: 262144,
    maxOutputTokens: 16384,
    adapter: "generic"
  },

  "meta/muse-glimmer-30b": {
    name: "Muse Glimmer 30B",
    provider: "Meta / Muse",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: 131072,
    maxOutputTokens: 8192,
    reasoningLevels: ["low", "medium", "high", "xhigh"],
    defaultReasoningEffort: "high",
    adapter: "muse-glimmer"
  },

  "mistralai/mistral-nemotron": {
    name: "Mistral Nemotron",
    provider: "Mistral AI",
    endpointStatus: "free-endpoint",
    enabled: true,
    contextTokens: null,
    maxOutputTokens: 4096,
    adapter: "generic"
  },

  "mistralai/mistral-medium-3.5-128b": {
    name: "Mistral Medium 3.5 128B",
    provider: "Mistral AI",
    endpointStatus: "availability-unverified",
    enabled: true,
    contextTokens: 262144,
    maxOutputTokens: 16384,
    adapter: "generic"
  },

  // Mistral Large 4 was released on October 29, 2026.
  // I could not verify an NVIDIA NIM Free Endpoint for it.
  // Keep it opt-in until NVIDIA exposes the matching endpoint
  // to your account. Do not assume Mistral's own API ID works
  // on NVIDIA's API.
  "mistral-large-4": {
    name: "Mistral Large 4 (NIM endpoint unverified)",
    provider: "Mistral AI",
    endpointStatus: "unverified-not-confirmed-on-nim",
    enabled: false,
    contextTokens: 1048576,
    maxOutputTokens: 16384,
    adapter: "generic"
  }
};

/* ============================================================
   PROFILE GENERATION
============================================================ */

const PROFILES = {};

function addEffortProfiles(modelId, levels) {
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

addThinkingProfiles("nvidia/nemotron-3-ultra-550b-a55b", {
  mediumEffort: true,
  deepBudget: 16384
});

addThinkingProfiles("nvidia/nemotron-3.5-super-vl-120b-a12b");

/* ============================================================
   MODEL RESOLUTION
============================================================ */

function resolveModel(requestedModel) {
  const requested = normalizeModel(requestedModel);
  const profile = PROFILES[requested] || null;
  const modelId = profile ? profile.baseModel : requested;
  const config = MODELS[modelId] || null;

  if (config && !config.enabled) {
    if (
      modelId === "mistral-large-4" &&
      ENABLE_UNVERIFIED_MISTRAL_LARGE_4
    ) {
      // Explicit opt-in only; the upstream may still reject it.
    } else {
      return {
        error: "model_disabled",
        requestedModel: requested,
        model: modelId,
        config
      };
    }
  }

  if (profile) {
    return {
      requestedModel: requested,
      model: modelId,
      profile,
      config
    };
  }

  if (config) {
    return {
      requestedModel: requested,
      model: modelId,
      profile: null,
      config
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

  if (["fast", "minimal", "minimum", "none"].includes(effort)) {
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
      if (budget <= 4096) return normalizeEffort("low", resolution.config);
      if (budget <= 16384) return normalizeEffort("high", resolution.config);
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
   REQUEST VALIDATION AND CONSTRUCTION
============================================================ */

const STANDARD_PASSTHROUGH_FIELDS = [
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
  "repetition_penalty",
  "frequency_penalty",
  "presence_penalty"
];

function validateMessages(messages) {
  if (!Array.isArray(messages)) {
    return "The messages field must be an array.";
  }

  if (messages.length === 0) {
    return "At least one message is required.";
  }

  if (messages.length > MAX_MESSAGES) {
    return `Too many messages. Maximum: ${MAX_MESSAGES}.`;
  }

  for (const message of messages) {
    if (!isObject(message) || typeof message.role !== "string") {
      return "Every message must be an object with a string role.";
    }

    if (
      message.content !== undefined &&
      typeof message.content !== "string" &&
      !Array.isArray(message.content) &&
      message.content !== null
    ) {
      return "Message content must be a string, array, or null.";
    }
  }

  return null;
}

function buildBaseRequest(body, resolution) {
  const config = resolution.config;
  const requestedMax = Number(body.max_tokens);

  let maxTokens =
    Number.isFinite(requestedMax) && requestedMax > 0
      ? Math.floor(requestedMax)
      : config?.maxOutputTokens || DEFAULT_MAX_TOKENS;

  maxTokens = Math.min(maxTokens, HARD_MAX_OUTPUT_TOKENS);

  if (Number.isFinite(config?.maxOutputTokens)) {
    maxTokens = Math.min(maxTokens, config.maxOutputTokens);
  }

  const request = {
    model: resolution.model,
    messages: body.messages.map(message => ({ ...message })),
    max_tokens: maxTokens,
    stream: body.stream === true
  };

  if (body.temperature !== undefined) {
    const temperature = Number(body.temperature);

    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
      throw new Error("temperature must be a number between 0 and 2.");
    }

    request.temperature = temperature;
  }

  if (body.top_p !== undefined) {
    const topP = Number(body.top_p);

    if (!Number.isFinite(topP) || topP < 0 || topP > 1) {
      throw new Error("top_p must be a number between 0 and 1.");
    }

    request.top_p = topP;
  }

  for (const key of STANDARD_PASSTHROUGH_FIELDS) {
    if (body[key] !== undefined) {
      request[key] = body[key];
    }
  }

  // Only explicitly supported extension fields are accepted.
  // Arbitrary extra_body fields cannot override model/messages/
  // stream/max_tokens/temperature/top_p.
  const extraBody = isObject(body.extra_body) ? body.extra_body : {};

  if (isObject(body.chat_template_kwargs)) {
    request.chat_template_kwargs = { ...body.chat_template_kwargs };
  }

  if (isObject(extraBody.chat_template_kwargs)) {
    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),
      ...extraBody.chat_template_kwargs
    };
  }

  if (isObject(body.nvext)) {
    request.nvext = { ...body.nvext };
  } else if (isObject(extraBody.nvext)) {
    request.nvext = { ...extraBody.nvext };
  }

  if (body.thinking_token_budget !== undefined) {
    request.thinking_token_budget = body.thinking_token_budget;
  } else if (extraBody.thinking_token_budget !== undefined) {
    request.thinking_token_budget = extraBody.thinking_token_budget;
  }

  // The selected model always wins.
  request.model = resolution.model;

  return request;
}

/* ============================================================
   MODEL-SPECIFIC REQUEST ADAPTERS
============================================================ */

function buildNimRequest(body, resolution) {
  const request = buildBaseRequest(body, resolution);
  const adapter = resolution.config?.adapter || "generic";
  const profile = resolution.profile;

  if (adapter === "glm-flash") {
    const effort = requestedEffort(body, resolution);

    request.reasoning_effort = effort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),
      clear_thinking: true
    };

    delete request.reasoning_mode;
    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  if (adapter === "minimax") {
    const effort = requestedEffort(body, resolution);

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),
      reasoning_effort: effort
    };

    delete request.reasoning_effort;
    delete request.reasoning_mode;
    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  if (adapter === "muse-glimmer") {
    request.reasoning_effort = requestedEffort(body, resolution);

    delete request.reasoning_mode;
    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  if (adapter === "nemotron" || adapter === "nemotron-vl") {
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
      typeof request.chat_template_kwargs?.enable_thinking === "boolean"
    ) {
      thinking = request.chat_template_kwargs.enable_thinking;
    }

    if (thinking === undefined && adapter === "nemotron") {
      thinking = DEFAULT_NEMOTRON_THINKING;
    }

    if (thinking !== undefined) {
      request.chat_template_kwargs = {
        ...(request.chat_template_kwargs || {}),
        enable_thinking: Boolean(thinking)
      };
    }

    if (adapter === "nemotron") {
      const template = { ...(request.chat_template_kwargs || {}) };

      for (const key of [
        "reasoning_effort",
        "reasoning_budget",
        "thinking_token_budget",
        "enable_thinking",
        "medium_effort",
        "force_nonempty_content"
      ]) {
        if (key !== "enable_thinking") delete template[key];
      }

      template.enable_thinking = Boolean(thinking);

      const mediumEffort =
        profile?.mediumEffort ??
        body.chat_template_kwargs?.medium_effort ??
        false;

      if (thinking && mediumEffort) {
        template.medium_effort = true;
      } else {
        delete template.medium_effort;
      }

      request.chat_template_kwargs = template;

      const budgetInput =
        profile?.thinkingTokenBudget ??
        body.thinking_token_budget ??
        body.reasoning_budget ??
        DEFAULT_REASONING_BUDGET;

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
    } else {
      delete request.enable_thinking;
      delete request.reasoning_mode;
      delete request.reasoning_budget;
      delete request.nvext;
    }

    return request;
  }

  // Generic models receive standard OpenAI-compatible fields.
  // Do not guess undocumented vendor-specific reasoning fields.
  delete request.reasoning_mode;
  delete request.reasoning_budget;
  delete request.enable_thinking;
  delete request.thinking_token_budget;

  return request;
}

/* ============================================================
   HEADERS AND UPSTREAM HELPERS
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

function extractErrorBody(data) {
  if (Buffer.isBuffer(data)) data = data.toString("utf8");

  if (typeof data === "string") {
    return safeJsonParse(data) || {
      message: data.slice(0, 4000)
    };
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
      message: String(message).slice(0, 4000),
      type: "upstream_error",
      code: status,
      model: resolution?.model || null,
      profile: resolution?.profile?.label || null,
      elapsed_ms: elapsedMs
    }
  };
}

async function readStreamLimited(stream, maxBytes = MAX_UPSTREAM_ERROR_BYTES) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;

    stream.on("data", chunk => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      total += buffer.length;

      if (total > maxBytes) {
        stream.destroy();
        reject(new Error("Upstream error body exceeded size limit."));
        return;
      }

      chunks.push(buffer);
    });

    stream.on("end", () => {
      resolve(Buffer.concat(chunks).toString("utf8"));
    });

    stream.on("error", reject);
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
  if (Array.isArray(data)) {
    return data.map(stripReasoningObject);
  }

  if (!isObject(data)) return data;

  const result = {};

  for (const [key, value] of Object.entries(data)) {
    if (["reasoning_content", "reasoning", "thinking"].includes(key)) {
      continue;
    }

    if (key === "content" && typeof value === "string") {
      result[key] = stripThinkTags(value);
    } else {
      result[key] = stripReasoningObject(value);
    }
  }

  return result;
}

/* ============================================================
   SSE PARSING AND FORWARDING
============================================================ */

function writeSSEEvent(res, eventText) {
  const lines = eventText.split(/\r?\n/);
  const dataLines = [];
  const otherLines = [];

  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    } else if (
      line.startsWith("event:") ||
      line.startsWith("id:") ||
      line.startsWith("retry:") ||
      line.startsWith(":")
    ) {
      otherLines.push(line);
    }
  }

  if (dataLines.length === 0) {
    res.write(`${otherLines.join("\n")}\n\n`);
    return;
  }

  const raw = dataLines.join("\n");

  if (raw === "[DONE]") {
    res.write(`${otherLines.join("\n")}${otherLines.length ? "\n" : ""}data: [DONE]\n\n`);
    return;
  }

  const parsed = safeJsonParse(raw);

  if (parsed === null) {
    res.write(`${otherLines.join("\n")}${otherLines.length ? "\n" : ""}`);
    for (const line of dataLines) res.write(`data: ${line}\n`);
    res.write("\n");
    return;
  }

  const cleaned = STRIP_REASONING
    ? stripReasoningObject(parsed)
    : parsed;

  res.write(`${otherLines.join("\n")}${otherLines.length ? "\n" : ""}`);
  res.write(`data: ${JSON.stringify(cleaned)}\n\n`);
}

function createSSEParser(onEvent) {
  let buffer = "";

  return {
    push(chunk) {
      buffer += chunk;

      while (true) {
        const match = /\r?\n\r?\n/.exec(buffer);
        if (!match) break;

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
   AUTHENTICATION AND CONCURRENCY
============================================================ */

function proxyAuth(req, res, next) {
  if (!PROXY_API_KEY) {
    if (IS_PRODUCTION) {
      return res.status(503).json({
        error: {
          message: "Proxy authentication is not configured.",
          type: "configuration_error"
        }
      });
    }

    return next();
  }

  const authorization = String(req.get("authorization") || "");
  const match = authorization.match(/^Bearer\s+(.+)$/i);

  if (!match) {
    return res.status(401).json({
      error: {
        message: "Missing or invalid proxy API key.",
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
        message: "Missing or invalid proxy API key.",
        type: "authentication_error"
      }
    });
  }

  return next();
}

function concurrencyLimit(req, res, next) {
  if (shuttingDown) {
    return res.status(503).json({
      error: {
        message: "Proxy is shutting down.",
        type: "server_unavailable"
      }
    });
  }

  if (activeRequests >= MAX_CONCURRENT_REQUESTS) {
    res.setHeader("Retry-After", "5");

    return res.status(429).json({
      error: {
        message: "Too many concurrent requests. Try again shortly.",
        type: "rate_limit_error"
      }
    });
  }

  activeRequests++;
  let released = false;

  function release() {
    if (released) return;
    released = true;
    activeRequests = Math.max(0, activeRequests - 1);
  }

  res.once("finish", release);
  res.once("close", release);

  next();
}

/* ============================================================
   MODEL LIST AND CACHE
============================================================ */

function localModelList() {
  const created = Math.floor(Date.now() / 1000);
  const data = [];

  for (const [id, config] of Object.entries(MODELS)) {
    data.push({
      id,
      object: "model",
      created,
      owned_by: config.provider,
      name: config.name,
      endpoint_status: config.endpointStatus,
      enabled: config.enabled
    });
  }

  for (const [id, profile] of Object.entries(PROFILES)) {
    const config = MODELS[profile.baseModel];

    data.push({
      id,
      object: "model",
      created,
      owned_by: config?.provider || "proxy",
      name: `${config?.name || profile.baseModel} - ${profile.label}`,
      base_model: profile.baseModel,
      enabled: config?.enabled === true
    });
  }

  return data;
}

let modelCache = {
  timestamp: 0,
  data: null
};

async function fetchRemoteModels(force = false) {
  if (
    !force &&
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
      isObject(response.data) &&
      Array.isArray(response.data.data)
    ) {
      modelCache = {
        timestamp: Date.now(),
        data: response.data
      };

      return response.data;
    }

    debugLog("model_list_unavailable", {
      status: response.status
    });
  } catch (error) {
    debugLog("model_list_fetch_failed", {
      message: error.message
    });
  }

  return null;
}

function combineModelLists(remote) {
  const local = localModelList();

  if (!Array.isArray(remote?.data)) {
    return { object: "list", data: local };
  }

  const seen = new Set(
    remote.data.filter(isObject).map(model => model.id)
  );

  return {
    object: "list",
    data: [
      ...remote.data,
      ...local.filter(model => !seen.has(model.id))
    ]
  };
}

/* ============================================================
   EXPRESS MIDDLEWARE
============================================================ */

const corsOptions = {
  origin(origin, callback) {
    // Requests without an Origin header include server-to-server calls.
    if (!origin) return callback(null, true);

    if (CORS_ORIGINS.includes("*")) {
      return callback(null, true);
    }

    if (CORS_ORIGINS.includes(origin)) {
      return callback(null, true);
    }

    return callback(null, false);
  },
  credentials: false,
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Authorization", "Content-Type"]
};

app.use(cors(corsOptions));

app.use(express.json({
  limit: MAX_BODY_BYTES,
  strict: true
}));

app.use("/v1", proxyAuth);

/* ============================================================
   ROOT AND HEALTH
============================================================ */

app.get("/", (req, res) => {
  res.json({
    name: "NVIDIA NIM OpenAI-Compatible Proxy",
    status: shuttingDown ? "shutting_down" : "ok",
    default_model: DEFAULT_MODEL,
    profile_count: Object.keys(PROFILES).length
  });
});

app.get("/health", (req, res) => {
  res.status(shuttingDown ? 503 : 200).json({
    status: shuttingDown ? "shutting_down" : "ok",
    api_key_configured: Boolean(NIM_API_KEY),
    proxy_auth_enabled: Boolean(PROXY_API_KEY),
    active_requests: activeRequests,
    max_concurrent_requests: MAX_CONCURRENT_REQUESTS
  });
});

/* ============================================================
   MODEL ENDPOINTS
============================================================ */

app.get("/v1/models", async (req, res) => {
  const remote = await fetchRemoteModels();
  return res.json(combineModelLists(remote));
});

app.post("/v1/models/refresh", async (req, res) => {
  const remote = await fetchRemoteModels(true);
  return res.json(combineModelLists(remote));
});

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post(
  "/v1/chat/completions",
  concurrencyLimit,
  async (req, res) => {
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

    if (resolution.error === "model_disabled") {
      return res.status(400).json({
        error: {
          message:
            "This model is configured but disabled because its NVIDIA NIM endpoint has not been verified.",
          type: "invalid_request_error",
          code: "model_disabled",
          model: resolution.model
        }
      });
    }

    const messageError = validateMessages(body.messages);

    if (messageError) {
      return res.status(400).json({
        error: {
          message: messageError,
          type: "invalid_request_error"
        }
      });
    }

    let upstreamBody;

    try {
      upstreamBody = buildNimRequest(body, resolution);
    } catch (error) {
      return res.status(400).json({
        error: {
          message: error.message,
          type: "invalid_request_error"
        }
      });
    }

    const streaming = upstreamBody.stream === true;
    const diagnosticMetadata = {
      model: resolution.model,
      requested_model: requestedModel,
      profile: resolution.profile?.label || null,
      stream: streaming
    };

    timingLog("request_received", startedAt, diagnosticMetadata);

    debugLog("request", {
      ...diagnosticMetadata,
      adapter: resolution.config?.adapter || "generic",
      message_count: body.messages.length,
      max_tokens: upstreamBody.max_tokens,
      temperature: upstreamBody.temperature ?? null,
      top_p: upstreamBody.top_p ?? null,
      reasoning_effort:
        upstreamBody.reasoning_effort ||
        upstreamBody.chat_template_kwargs?.reasoning_effort ||
        null,
      thinking_enabled:
        upstreamBody.chat_template_kwargs?.enable_thinking ?? null
    });

    const controller = new AbortController();
    let upstreamStream = null;
    let clientDisconnected = false;

    const abortRequest = () => {
      if (clientDisconnected) return;
      clientDisconnected = true;
      controller.abort();

      if (
        upstreamStream &&
        typeof upstreamStream.destroy === "function"
      ) {
        upstreamStream.destroy();
      }
    };

    req.once("aborted", abortRequest);

    res.once("close", () => {
      if (!res.writableEnded) {
        abortRequest();

        timingLog("client_disconnected", startedAt, diagnosticMetadata);
      }
    });

    let upstream;

    try {
      upstream = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        upstreamBody,
        {
          headers: buildHeaders(streaming),
          timeout: NIM_TIMEOUT_MS,
          signal: controller.signal,
          httpAgent,
          httpsAgent,
          responseType: streaming ? "stream" : "json",
          validateStatus: () => true,
          maxContentLength: Infinity,
          maxBodyLength: Infinity
        }
      );
    } catch (error) {
      if (clientDisconnected || res.destroyed) return;

      timingLog("upstream_request_failed", startedAt, {
        ...diagnosticMetadata,
        error_code: error?.code || null,
        error_message: error?.message || "unknown error"
      });

      if (res.headersSent) return res.end();

      const status = error?.code === "ECONNABORTED" ? 504 : 502;

      return res.status(status).json(
        makeProxyError(error, resolution, Date.now() - startedAt)
      );
    }

    upstreamStream =
      streaming && upstream.data && typeof upstream.data.on === "function"
        ? upstream.data
        : null;

    timingLog("upstream_headers_received", startedAt, {
      ...diagnosticMetadata,
      upstream_status: upstream.status
    });

    if (upstream.status < 200 || upstream.status >= 300) {
      let data = upstream.data;

      if (upstreamStream) {
        try {
          data = await readStreamLimited(upstreamStream);
        } catch {
          data = "Unable to read upstream error response.";
        }
      }

      if (clientDisconnected || res.destroyed) return;

      const proxyError = makeProxyError(
        {
          response: {
            status: upstream.status,
            data
          }
        },
        resolution,
        Date.now() - startedAt
      );

      debugLog("upstream_error", {
        status: upstream.status,
        model: resolution.model,
        message: proxyError.error.message
      });

      // Preserve common API status codes; map unexpected upstream
      // server errors to 502.
      const status = upstream.status >= 400 && upstream.status < 500
        ? upstream.status
        : 502;

      return res.status(status).json(proxyError);
    }

    if (!streaming) {
      const data = STRIP_REASONING
        ? stripReasoningObject(upstream.data)
        : upstream.data;

      timingLog("nonstream_response_forwarded", startedAt, {
        ...diagnosticMetadata
      });

      return res.status(upstream.status).json(data);
    }

    if (!upstreamStream) {
      return res.status(502).json({
        error: {
          message: "Upstream did not return a readable stream.",
          type: "upstream_error"
        }
      });
    }

    res.status(upstream.status);
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("X-Accel-Buffering", "no");

    if (typeof res.flushHeaders === "function") {
      res.flushHeaders();
    }

    let firstChunkLogged = false;
    let firstEventLogged = false;
    let firstOutputLogged = false;

    const decoder = new StringDecoder("utf8");

    const parser = createSSEParser(event => {
      if (res.writableEnded || res.destroyed) return;

      if (!firstEventLogged) {
        firstEventLogged = true;

        timingLog("first_sse_event", startedAt, {
          ...diagnosticMetadata,
          event_chars: event.length
        });
      }

      if (STRIP_REASONING) {
        writeSSEEvent(res, event);
      } else {
        res.write(`${event}\n\n`);
      }

      if (!firstOutputLogged) {
        firstOutputLogged = true;

        timingLog("first_output_forwarded", startedAt, {
          ...diagnosticMetadata
        });
      }
    });

    upstreamStream.on("data", chunk => {
      if (res.writableEnded || res.destroyed) return;

      if (!firstChunkLogged) {
        firstChunkLogged = true;

        timingLog("first_upstream_chunk", startedAt, {
          ...diagnosticMetadata,
          chunk_bytes: Buffer.isBuffer(chunk)
            ? chunk.length
            : Buffer.byteLength(String(chunk))
        });
      }

      parser.push(decoder.write(chunk));
    });

    upstreamStream.on("end", () => {
      if (res.writableEnded || res.destroyed) return;

      parser.push(decoder.end());
      parser.end();
      res.end();

      timingLog("upstream_stream_ended", startedAt, {
        ...diagnosticMetadata,
        received_chunk: firstChunkLogged,
        received_event: firstEventLogged,
        forwarded_output: firstOutputLogged
      });
    });

    upstreamStream.on("error", error => {
      if (clientDisconnected || res.destroyed || res.writableEnded) return;

      timingLog("upstream_stream_error", startedAt, {
        ...diagnosticMetadata,
        error_code: error?.code || null,
        error_message: error?.message || "unknown error"
      });

      // Once SSE headers have been sent, a JSON error response is
      // no longer possible. End the stream cleanly.
      res.end();
    });
  }
);

/* ============================================================
   404 AND ERROR HANDLING
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
  if (res.headersSent) return next(error);

  if (error?.type === "entity.too.large") {
    return res.status(413).json({
      error: {
        message: "Request body is too large.",
        type: "invalid_request_error"
      }
    });
  }

  if (
    error instanceof SyntaxError &&
    error.status === 400 &&
    Object.prototype.hasOwnProperty.call(error, "body")
  ) {
    return res.status(400).json({
      error: {
        message: "Malformed JSON request body.",
        type: "invalid_request_error"
      }
    });
  }

  console.error(
    `[proxy] unhandled error: ${error?.message || "unknown error"}`
  );

  return res.status(500).json({
    error: {
      message: "Internal proxy error.",
      type: "internal_error"
    }
  });
});

/* ============================================================
   START AND SHUTDOWN
============================================================ */

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`[proxy] listening on port ${PORT}`);
  console.log(`[proxy] default_model=${DEFAULT_MODEL}`);
  console.log(`[proxy] NVIDIA API key configured=${Boolean(NIM_API_KEY)}`);
  console.log(`[proxy] proxy authentication configured=${Boolean(PROXY_API_KEY)}`);
  console.log(`[proxy] production=${IS_PRODUCTION}`);
  console.log(`[proxy] reasoning stripping=${STRIP_REASONING}`);
  console.log(`[proxy] concurrent request limit=${MAX_CONCURRENT_REQUESTS}`);
  console.log(`[proxy] profile count=${Object.keys(PROFILES).length}`);

  if (!NIM_API_KEY) {
    console.warn("[proxy] WARNING: NVIDIA API key is not configured.");
  }

  if (IS_PRODUCTION && !PROXY_API_KEY) {
    console.error(
      "[proxy] WARNING: /v1 requests will be rejected until PROXY_API_KEY is set."
    );
  }
});

server.requestTimeout = 0;
server.headersTimeout = 30000;
server.keepAliveTimeout = 5000;

function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;

  console.log(`[proxy] received ${signal}; shutting down`);

  server.close(() => {
    httpAgent.destroy();
    httpsAgent.destroy();
    process.exit(0);
  });

  // Close active sockets after a short grace period.
  const forceExit = setTimeout(() => {
    httpAgent.destroy();
    httpsAgent.destroy();
    process.exit(1);
  }, 10000);

  forceExit.unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
