"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");

const app = express();

/* ============================================================
   SERVER / NIM CONFIGURATION
============================================================ */

const PORT =
  Number(process.env.PORT) || 10000;

const NIM_API_BASE = (
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NIM_API_KEY =
  process.env.NIM_API_KEY ||
  process.env.NVIDIA_API_KEY ||
  "";

/*
 * GLM 5.3 is the quality-first default.
 *
 * You can still override this in Render with DEFAULT_MODEL.
 */
const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "z-ai/glm-5.3";

/*
 * IMPORTANT:
 *
 * Keep this exactly as permissive as the original proxy.
 *
 * Unknown models remain allowed by default.
 */
const ALLOW_UNKNOWN_MODELS =
  String(
    process.env.ALLOW_UNKNOWN_MODELS ||
      "true"
  )
    .trim()
    .toLowerCase() === "true";

/*
 * Debug logging is intentionally OFF by default.
 *
 * Enable DEBUG_PROXY=true when diagnosing the proxy.
 *
 * This never logs the actual RP messages.
 */
const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY ||
      "false"
  )
    .trim()
    .toLowerCase() === "true";

/*
 * Reasoning is stripped from responses by default.
 *
 * This keeps hidden reasoning out of JanitorAI's visible response.
 */
const STRIP_REASONING_FROM_RESPONSE =
  String(
    process.env.STRIP_REASONING_FROM_RESPONSE ||
      "true"
  )
    .trim()
    .toLowerCase() === "true";

/*
 * Long-running request timeout.
 *
 * This is deliberately large because reasoning-heavy
 * GLM requests can take a while.
 */
const NIM_TIMEOUT =
  Number.isFinite(
    Number(
      process.env.NIM_TIMEOUT_MS
    )
  )
    ? Number(
        process.env.NIM_TIMEOUT_MS
      )
    : 900000;

/*
 * Model-list cache.
 *
 * This is only for GET /v1/models discovery.
 * It has NOTHING to do with completion retries.
 */
const MODEL_CACHE_TTL =
  Number.isFinite(
    Number(
      process.env.MODEL_CACHE_TTL_MS
    )
  )
    ? Number(
        process.env.MODEL_CACHE_TTL_MS
      )
    : 300000;

const MAX_ERROR_BODY_SIZE =
  2 * 1024 * 1024;

/* ============================================================
   DEFAULT GENERATION SETTINGS
============================================================ */

const DEFAULT_REASONING_EFFORT =
  String(
    process.env.DEFAULT_REASONING_EFFORT ||
      "high"
  )
    .trim()
    .toLowerCase();

const DEFAULT_REASONING_BUDGET =
  Number.isFinite(
    Number(
      process.env.DEFAULT_REASONING_BUDGET
    )
  )
    ? Number(
        process.env.DEFAULT_REASONING_BUDGET
      )
    : 16384;

const DEFAULT_MAX_TOKENS =
  Number.isFinite(
    Number(
      process.env.DEFAULT_MAX_TOKENS
    )
  )
    ? Number(
        process.env.DEFAULT_MAX_TOKENS
      )
    : 16384;

const DEFAULT_TEMPERATURE =
  Number.isFinite(
    Number(
      process.env.DEFAULT_TEMPERATURE
    )
  )
    ? Number(
        process.env.DEFAULT_TEMPERATURE
      )
    : 1.0;

const DEFAULT_TOP_P =
  Number.isFinite(
    Number(
      process.env.DEFAULT_TOP_P
    )
  )
    ? Number(
        process.env.DEFAULT_TOP_P
      )
    : 0.95;

const DEFAULT_REPETITION_PENALTY =
  Number.isFinite(
    Number(
      process.env.DEFAULT_REPETITION_PENALTY
    )
  )
    ? Number(
        process.env.DEFAULT_REPETITION_PENALTY
      )
    : 1.0;

const DEFAULT_FREQUENCY_PENALTY =
  Number.isFinite(
    Number(
      process.env.DEFAULT_FREQUENCY_PENALTY
    )
  )
    ? Number(
        process.env.DEFAULT_FREQUENCY_PENALTY
      )
    : 0.0;

const DEFAULT_PRESENCE_PENALTY =
  Number.isFinite(
    Number(
      process.env.DEFAULT_PRESENCE_PENALTY
    )
  )
    ? Number(
        process.env.DEFAULT_PRESENCE_PENALTY
      )
    : 0.0;

/*
 * Preserve the original Nemotron default behavior.
 */
const DEFAULT_NEMOTRON_THINKING =
  parseBoolean(
    process.env.NEMOTRON_ENABLE_THINKING,
    false
  );

/* ============================================================
   HTTP AGENTS
============================================================ */

/*
 * Persistent connections reduce avoidable connection setup
 * overhead between Render and NVIDIA.
 *
 * These do NOT perform retries.
 */
const httpAgent =
  new http.Agent({
    keepAlive: true,
    maxSockets: 64,
    maxFreeSockets: 16,
    scheduling: "lifo"
  });

const httpsAgent =
  new https.Agent({
    keepAlive: true,
    maxSockets: 64,
    maxFreeSockets: 16,
    scheduling: "lifo"
  });

/* ============================================================
   BASIC HELPERS
============================================================ */

function isPlainObject(value) {
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

  if (
    typeof value === "string" &&
    value.trim() !== ""
  ) {
    return Number.isFinite(
      Number(value)
    );
  }

  return false;
}

function numberOrDefault(
  value,
  fallback
) {
  return isFiniteNumber(value)
    ? Number(value)
    : fallback;
}

function parseBoolean(
  value,
  fallback
) {
  if (
    value === undefined ||
    value === null
  ) {
    return fallback;
  }

  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return value !== 0;
  }

  if (typeof value === "string") {
    const normalized =
      value
        .trim()
        .toLowerCase();

    if (
      [
        "true",
        "1",
        "yes",
        "on"
      ].includes(normalized)
    ) {
      return true;
    }

    if (
      [
        "false",
        "0",
        "no",
        "off"
      ].includes(normalized)
    ) {
      return false;
    }
  }

  return fallback;
}

function normalizeModel(model) {
  return String(model || "")
    .trim()
    .toLowerCase();
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/* ============================================================
   MODEL CONFIGURATION
============================================================ */

/*
 * These are the actual NVIDIA model IDs handled specially
 * by this proxy.
 *
 * The model-specific adapter is selected from this table.
 */
const MODELS = {
  "z-ai/glm-5.3": {
    name: "GLM 5.3",
    provider: "Z.ai",

    /*
     * NVIDIA documents these three reasoning levels.
     */
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],

    /*
     * We intentionally use HIGH as the normal/default
     * GLM profile because your stated objective is:
     *
     *   quality + lower TTFT
     *
     * The Deep profile explicitly selects MAX.
     */
    defaultReasoningEffort:
      "high",

    /*
     * This is the maximum output ceiling exposed by
     * the proxy when Janitor does not request something
     * smaller.
     */
    maxOutputTokens:
      32768,

    maxReasoningBudget:
      null,

    adapter:
      "glm53",

    /*
     * NVIDIA documents GLM-5.3 as supporting a 1,048,576
     * token context window.
     *
     * We do NOT attempt to construct or alter Janitor's
     * context ourselves.
     */
    contextTokens:
      1048576
  },

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",

    /*
     * Kimi reasoning is never disabled by this proxy.
     */
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort:
      "max",

    maxOutputTokens:
      8192,

    maxReasoningBudget:
      32768,

    adapter:
      "kimi-k3"
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name:
      "NVIDIA Nemotron 3 Ultra 550B",

    provider:
      "NVIDIA",

    /*
     * Nemotron uses a thinking switch rather than
     * GLM's reasoning_effort mechanism.
     */
    reasoningLevels: [
      "none",
      "high"
    ],

    defaultReasoningEffort:
      "high",

    defaultThinking:
      true,

    maxReasoningBudget:
      32768,

    maxOutputTokens:
      16384,

    adapter:
      "nemotron"
  },

  "deepseek-ai/deepseek-v4-flash-0731": {
    name:
      "DeepSeek V4 Flash 0731",

    provider:
      "DeepSeek AI",

    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort:
      "max",

    maxOutputTokens:
      8192,

    maxReasoningBudget:
      16384,

    adapter:
      "deepseek-flash"
  }
};

/* ============================================================
   PROXY PROFILES
============================================================ */

/*
 * JanitorAI does not need to understand reasoning profiles.
 *
 * These are virtual model IDs exposed by OUR /v1/models
 * endpoint.
 *
 * When Janitor selects one:
 *
 *     z-ai/glm-5.3-balanced
 *
 * the proxy sends:
 *
 *     model = z-ai/glm-5.3
 *     reasoning_effort = high
 *
 * to NVIDIA.
 *
 * This means profiles can be selected from Janitor's normal
 * model selector if Janitor refreshes the model list.
 */

/* ---------------- GLM 5.3 ---------------- */

const PROFILES = {
  "z-ai/glm-5.3-fast": {
    baseModel:
      "z-ai/glm-5.3",
    label:
      "Fast",
    reasoningEffort:
      "low"
  },

  "z-ai/glm-5.3-balanced": {
    baseModel:
      "z-ai/glm-5.3",
    label:
      "Balanced",
    reasoningEffort:
      "high"
  },

  "z-ai/glm-5.3-deep": {
    baseModel:
      "z-ai/glm-5.3",
    label:
      "Deep",
    reasoningEffort:
      "max"
  },

  /* ---------------- KIMI K3 ---------------- */

  "moonshotai/kimi-k3-fast": {
    baseModel:
      "moonshotai/kimi-k3",
    label:
      "Fast",
    reasoningEffort:
      "low"
  },

  "moonshotai/kimi-k3-balanced": {
    baseModel:
      "moonshotai/kimi-k3",
    label:
      "Balanced",
    reasoningEffort:
      "high"
  },

  "moonshotai/kimi-k3-deep": {
    baseModel:
      "moonshotai/kimi-k3",
    label:
      "Deep",
    reasoningEffort:
      "max"
  },

  /* ---------------- NEMOTRON ---------------- */

  /*
   * Nemotron does not use GLM's reasoning_effort.
   *
   * Instead:
   *
   *   Fast     = thinking disabled
   *   Balanced = thinking enabled + 16K budget
   *   Deep     = thinking enabled + 32K budget
   */
  "nvidia/nemotron-3-ultra-550b-a55b-fast": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label:
      "Fast",
    thinking:
      false
  },

  "nvidia/nemotron-3-ultra-550b-a55b-balanced": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label:
      "Balanced",
    thinking:
      true,
    reasoningBudget:
      16384
  },

  "nvidia/nemotron-3-ultra-550b-a55b-deep": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label:
      "Deep",
    thinking:
      true,
    reasoningBudget:
      32768
  },

  /* ---------------- DEEPSEEK ---------------- */

  "deepseek-ai/deepseek-v4-flash-0731-fast": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label:
      "Fast",
    reasoningEffort:
      "low"
  },

  "deepseek-ai/deepseek-v4-flash-0731-balanced": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label:
      "Balanced",
    reasoningEffort:
      "high"
  },

  "deepseek-ai/deepseek-v4-flash-0731-deep": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label:
      "Deep",
    reasoningEffort:
      "max"
  }
};

function getProfile(model) {
  return (
    PROFILES[
      normalizeModel(model)
    ] || null
  );
}

function resolveModel(model) {
  const requested =
    String(model || "").trim();

  const profile =
    getProfile(requested);

  return {
    requestedModel:
      requested,

    baseModel:
      profile
        ? profile.baseModel
        : requested,

    profile
  };
}

function getModelConfig(model) {
  const resolved =
    resolveModel(model);

  return (
    MODELS[
      normalizeModel(
        resolved.baseModel
      )
    ] || null
  );
}

function isSupportedModel(model) {
  return (
    !!getModelConfig(model) ||
    ALLOW_UNKNOWN_MODELS
  );
}

/* ============================================================
   MODEL LIST HELPERS
============================================================ */

function publicModelEntry(
  id,
  config,
  extra
) {
  return {
    id,

    object:
      "model",

    created:
      Math.floor(
        Date.now() / 1000
      ),

    owned_by:
      config?.provider ||
      "proxy",

    ...(extra || {})
  };
}

function localModelEntries() {
  const entries = [];

  /*
   * Real configured models.
   */
  for (
    const [
      id,
      config
    ] of Object.entries(
      MODELS
    )
  ) {
    entries.push(
      publicModelEntry(
        id,
        config,
        {
          reasoning_levels:
            config.reasoningLevels,

          max_output_tokens:
            config.maxOutputTokens,

          context_tokens:
            config.contextTokens ||
            null,

          proxy_profile:
            false
        }
      )
    );
  }

  /*
   * Virtual profile models.
   */
  for (
    const [
      id,
      profile
    ] of Object.entries(
      PROFILES
    )
  ) {
    const config =
      MODELS[
        profile.baseModel
      ];

    entries.push(
      publicModelEntry(
        id,
        config,
        {
          reasoning_levels:
            config?.reasoningLevels ||
            [],

          max_output_tokens:
            config?.maxOutputTokens ||
            DEFAULT_MAX_TOKENS,

          context_tokens:
            config?.contextTokens ||
            null,

          proxy_profile:
            true,

          profile:
            profile.label,

          profile_base_model:
            profile.baseModel
        }
      )
    );
  }

  return entries;
}

/* ============================================================
   GENERATION HELPERS
============================================================ */

function clampTemperature(
  value,
  model
) {
  const resolved =
    resolveModel(model);

  const maximum =
    normalizeModel(
      resolved.baseModel
    ) ===
    "moonshotai/kimi-k3"
      ? 1
      : 2;

  return Math.min(
    maximum,
    Math.max(
      0,
      numberOrDefault(
        value,
        DEFAULT_TEMPERATURE
      )
    )
  );
}

function clampTopP(value) {
  return Math.min(
    1,
    Math.max(
      0,
      numberOrDefault(
        value,
        DEFAULT_TOP_P
      )
    )
  );
}

function clampMaxTokens(
  value,
  modelConfig
) {
  const requested =
    Math.max(
      1,
      Math.floor(
        numberOrDefault(
          value,
          DEFAULT_MAX_TOKENS
        )
      )
    );

  if (
    modelConfig &&
    modelConfig.maxOutputTokens
  ) {
    return Math.min(
      modelConfig.maxOutputTokens,
      requested
    );
  }

  return requested;
}

function clampReasoningBudget(
  value,
  modelConfig
) {
  const maximum =
    modelConfig?.maxReasoningBudget ||
    65536;

  return Math.min(
    maximum,
    Math.max(
      0,
      Math.floor(
        numberOrDefault(
          value,
          DEFAULT_REASONING_BUDGET
        )
      )
    )
  );
}

/* ============================================================
   REASONING HELPERS
============================================================ */

function normalizeReasoningEffort(
  value,
  modelConfig
) {
  const levels =
    modelConfig?.reasoningLevels ||
    [];

  if (
    typeof value === "string"
  ) {
    const normalized =
      value
        .trim()
        .toLowerCase();

    if (
      levels.includes(
        normalized
      )
    ) {
      return normalized;
    }

    if (
      [
        "off",
        "false",
        "0",
        "disabled",
        "disable",
        "none"
      ].includes(normalized) &&
      levels.includes("none")
    ) {
      return "none";
    }

    if (
      [
        "med",
        "medium"
      ].includes(normalized) &&
      levels.includes("medium")
    ) {
      return "medium";
    }

    if (
      [
        "maximum",
        "maximum_reasoning"
      ].includes(normalized) &&
      levels.includes("max")
    ) {
      return "max";
    }
  }

  return (
    modelConfig?.defaultReasoningEffort ||
    levels[0] ||
    "none"
  );
}

function budgetToReasoningEffort(
  budget,
  modelConfig
) {
  const value =
    clampReasoningBudget(
      budget,
      modelConfig
    );

  const levels =
    modelConfig?.reasoningLevels ||
    [];

  if (
    levels.includes("none") &&
    value <= 0
  ) {
    return "none";
  }

  if (
    levels.includes("low") &&
    value <= 8192
  ) {
    return "low";
  }

  if (
    levels.includes("medium") &&
    value <= 16384
  ) {
    return "medium";
  }

  if (
    levels.includes("high") &&
    value <= 24576
  ) {
    return "high";
  }

  if (
    levels.includes("max")
  ) {
    return "max";
  }

  if (
    levels.includes("high")
  ) {
    return "high";
  }

  if (
    levels.includes("medium")
  ) {
    return "medium";
  }

  if (
    levels.includes("low")
  ) {
    return "low";
  }

  return (
    levels[0] ||
    "none"
  );
}

function getRequestedReasoningEffort(
  incoming,
  model,
  modelConfig,
  profile
) {
  /*
   * A selected proxy profile has priority.
   *
   * This is how Janitor's model selection becomes
   * an internal reasoning profile.
   */
  if (
    profile &&
    profile.reasoningEffort
  ) {
    return normalizeReasoningEffort(
      profile.reasoningEffort,
      modelConfig
    );
  }

  const normalizedModel =
    normalizeModel(model);

  /*
   * Kimi reasoning is never disabled.
   */
  if (
    normalizedModel ===
    "moonshotai/kimi-k3"
  ) {
    if (
      incoming.reasoning_effort !==
      undefined
    ) {
      const requested =
        normalizeReasoningEffort(
          incoming.reasoning_effort,
          modelConfig
        );

      return requested ===
        "none"
        ? "max"
        : requested;
    }

    if (
      incoming.reasoning_mode !==
      undefined
    ) {
      const requested =
        normalizeReasoningEffort(
          incoming.reasoning_mode,
          modelConfig
        );

      return requested ===
        "none"
        ? "max"
        : requested;
    }

    return "max";
  }

  if (
    incoming.reasoning_effort !==
    undefined
  ) {
    return normalizeReasoningEffort(
      incoming.reasoning_effort,
      modelConfig
    );
  }

  if (
    incoming.reasoning_mode !==
    undefined
  ) {
    return normalizeReasoningEffort(
      incoming.reasoning_mode,
      modelConfig
    );
  }

  if (
    incoming.reasoning_budget !==
    undefined
  ) {
    return budgetToReasoningEffort(
      incoming.reasoning_budget,
      modelConfig
    );
  }

  return normalizeReasoningEffort(
    DEFAULT_REASONING_EFFORT,
    modelConfig
  );
}

/* ============================================================
   NEMOTRON THINKING
============================================================ */

function getNemotronThinking(
  incoming,
  profile
) {
  /*
   * Profile wins first.
   */
  if (
    profile &&
    typeof profile.thinking ===
      "boolean"
  ) {
    return profile.thinking;
  }

  /*
   * Explicit chat_template_kwargs wins next.
   */
  if (
    isPlainObject(
      incoming.chat_template_kwargs
    ) &&
    typeof
      incoming
        .chat_template_kwargs
        .enable_thinking ===
        "boolean"
  ) {
    return (
      incoming
        .chat_template_kwargs
        .enable_thinking
    );
  }

  /*
   * Convenience top-level flag.
   */
  if (
    incoming.enable_thinking !==
    undefined
  ) {
    return parseBoolean(
      incoming.enable_thinking,
      DEFAULT_NEMOTRON_THINKING
    );
  }

  /*
   * reasoning_effort.
   */
  if (
    incoming.reasoning_effort !==
    undefined
  ) {
    const value =
      String(
        incoming.reasoning_effort
      )
        .trim()
        .toLowerCase();

    if (
      [
        "none",
        "off",
        "false",
        "0",
        "disabled",
        "disable"
      ].includes(value)
    ) {
      return false;
    }

    return true;
  }

  /*
   * reasoning_mode.
   */
  if (
    incoming.reasoning_mode !==
    undefined
  ) {
    const value =
      String(
        incoming.reasoning_mode
      )
        .trim()
        .toLowerCase();

    if (
      [
        "none",
        "off",
        "false",
        "0",
        "disabled",
        "disable"
      ].includes(value)
    ) {
      return false;
    }

    return true;
  }

  return DEFAULT_NEMOTRON_THINKING;
}

/* ============================================================
   MESSAGE NORMALIZATION
============================================================ */

function normalizeMessages(
  messages
) {
  if (
    !Array.isArray(messages)
  ) {
    return [];
  }

  const result = [];

  for (
    const message of messages
  ) {
    if (
      !isPlainObject(message)
    ) {
      continue;
    }

    if (
      typeof message.role !==
        "string" ||
      !message.role.trim()
    ) {
      continue;
    }

    /*
     * Everything else in the message is preserved.
     *
     * This is important for Janitor's context.
     */
    result.push({
      ...message,

      role:
        message.role.trim()
    });
  }

  return result;
}

/* ============================================================
   BUILD NVIDIA REQUEST
============================================================ */

function buildNimRequest(
  incoming,
  requestedModel,
  messages,
  stream
) {
  const resolved =
    resolveModel(
      requestedModel
    );

  const model =
    resolved.baseModel;

  const profile =
    resolved.profile;

  const normalizedModel =
    normalizeModel(model);

  const modelConfig =
    MODELS[
      normalizedModel
    ] || null;

  const isKimi =
    normalizedModel ===
    "moonshotai/kimi-k3";

  /*
   * Start with the common OpenAI-compatible request.
   */
  const request = {
    model,

    messages,

    temperature:
      clampTemperature(
        incoming.temperature,
        model
      ),

    max_tokens:
      clampMaxTokens(
        incoming.max_tokens,
        modelConfig
      ),

    stream
  };

  /*
   * Kimi's original behavior deliberately omitted
   * top_p, so preserve that.
   */
  if (!isKimi) {
    request.top_p =
      clampTopP(
        incoming.top_p
      );
  }

  /* ==========================================================
     OPTIONAL SAMPLING PARAMETERS
  ========================================================== */

  if (
    !isKimi &&
    incoming.repetition_penalty !==
      undefined
  ) {
    request.repetition_penalty =
      numberOrDefault(
        incoming.repetition_penalty,
        DEFAULT_REPETITION_PENALTY
      );
  }

  if (
    !isKimi &&
    incoming.frequency_penalty !==
      undefined
  ) {
    request.frequency_penalty =
      numberOrDefault(
        incoming.frequency_penalty,
        DEFAULT_FREQUENCY_PENALTY
      );
  }

  if (
    !isKimi &&
    incoming.presence_penalty !==
      undefined
  ) {
    request.presence_penalty =
      numberOrDefault(
        incoming.presence_penalty,
        DEFAULT_PRESENCE_PENALTY
      );
  }

  /*
   * Preserve additional parameters Janitor/NIM may support.
   */
  const optionalParameters = [
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

  for (
    const parameter of
      optionalParameters
  ) {
    if (
      incoming[parameter] !==
      undefined
    ) {
      request[parameter] =
        incoming[parameter];
    }
  }

  /* ==========================================================
     CLIENT CHAT TEMPLATE KWARGS
  ========================================================== */

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    )
  ) {
    request.chat_template_kwargs = {
      ...incoming.chat_template_kwargs
    };
  }

  /* ==========================================================
     CLIENT extra_body
  ========================================================== */

  if (
    isPlainObject(
      incoming.extra_body
    )
  ) {
    Object.assign(
      request,
      incoming.extra_body
    );
  }

  /* ==========================================================
     GLM 5.3
  ========================================================== */

  if (
    normalizedModel ===
    "z-ai/glm-5.3"
  ) {
    /*
     * GLM-5.3 thinking is always enabled.
     *
     * reasoning_effort controls how much reasoning
     * the model performs:
     *
     * low / high / max
     */
    request.reasoning_effort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig,
        profile
      );

    /*
     * CRITICAL FOR MULTI-TURN CHAT:
     *
     * clear_thinking=true prevents prior hidden reasoning
     * from being carried forward through the chat template.
     *
     * This does NOT disable reasoning.
     */
    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      clear_thinking:
        true
    };

    /*
     * GLM uses reasoning_effort rather than Nemotron's
     * reasoning_budget mechanism.
     */
    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.reasoning_mode;
    delete request.nvext;

    return request;
  }

  /* ==========================================================
     NEMOTRON
  ========================================================== */

  if (
    normalizedModel ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    const enableThinking =
      getNemotronThinking(
        incoming,
        profile
      );

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      enable_thinking:
        enableThinking
    };

    if (
      enableThinking
    ) {
      const requestedBudget =
        profile?.reasoningBudget ??
        (
          incoming.reasoning_budget !==
          undefined
            ? incoming.reasoning_budget
            : 16384
        );

      request.reasoning_budget =
        clampReasoningBudget(
          requestedBudget,
          modelConfig
        );
    } else {
      delete request.reasoning_budget;
    }

    /*
     * NVIDIA's Nemotron documentation uses this
     * for thinking + tool scenarios.
     */
    if (
      enableThinking &&
      Array.isArray(
        request.tools
      ) &&
      request.tools.length > 0
    ) {
      request.chat_template_kwargs = {
        ...request.chat_template_kwargs,

        force_nonempty_content:
          true
      };
    }

    /*
     * Do not accidentally send GLM-style reasoning
     * controls to Nemotron.
     */
    delete request.reasoning_effort;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.nvext;

    return request;
  }

  /* ==========================================================
     DEEPSEEK V4 FLASH
  ========================================================== */

  if (
    normalizedModel ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
    const reasoningEffort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig,
        profile
      );

    request.reasoning_effort =
      reasoningEffort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      thinking:
        reasoningEffort !==
        "none",

      reasoning_effort:
        reasoningEffort
    };

    return request;
  }

  /* ==========================================================
     DEEPSEEK V4 PRO
  ========================================================== */

  /*
   * Preserve the original proxy's explicit handling of
   * DeepSeek V4 Pro even though it isn't registered in
   * MODELS above.
   *
   * ALLOW_UNKNOWN_MODELS remains untouched.
   */
  if (
    normalizedModel ===
    "deepseek-ai/deepseek-v4-pro-0813"
  ) {
    const reasoningEffort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig,
        profile
      );

    request.reasoning_effort =
      reasoningEffort;

    return request;
  }

  /* ==========================================================
     KIMI K3
  ========================================================== */

  if (
    normalizedModel ===
    "moonshotai/kimi-k3"
  ) {
    /*
     * Kimi reasoning is NEVER disabled.
     */
    const reasoningEffort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig,
        profile
      );

    request.reasoning_effort =
      reasoningEffort ===
      "none"
        ? "max"
        : reasoningEffort;

    /*
     * Prevent callers from injecting a false
     * reasoning-off configuration.
     */
    if (
      isPlainObject(
        request.chat_template_kwargs
      )
    ) {
      delete request
        .chat_template_kwargs
        .enable_thinking;
    }

    delete request.reasoning_mode;

    return request;
  }

  /* ==========================================================
     UNKNOWN MODELS
  ========================================================== */

  /*
   * IMPORTANT:
   *
   * Unknown-model behavior remains permissive.
   *
   * We do NOT change ALLOW_UNKNOWN_MODELS.
   */
  if (!modelConfig) {
    if (
      incoming.reasoning_effort !==
      undefined
    ) {
      request.reasoning_effort =
        incoming.reasoning_effort;
    }

    if (
      incoming.reasoning_mode !==
      undefined
    ) {
      request.reasoning_mode =
        incoming.reasoning_mode;
    }

    if (
      incoming.reasoning_budget !==
      undefined
    ) {
      request.reasoning_budget =
        clampReasoningBudget(
          incoming.reasoning_budget
        );
    }
  }

  return request;
}

/* ============================================================
   RESPONSE REASONING STRIPPING
============================================================ */

function stripReasoning(
  parsed
) {
  if (
    !STRIP_REASONING_FROM_RESPONSE ||
    !parsed ||
    typeof parsed !==
      "object"
  ) {
    return parsed;
  }

  if (
    Array.isArray(
      parsed.choices
    )
  ) {
    for (
      const choice of
        parsed.choices
    ) {
      if (
        !choice ||
        typeof choice !==
          "object"
      ) {
        continue;
      }

      if (
        choice.delta &&
        typeof choice.delta ===
          "object"
      ) {
        delete choice
          .delta
          .reasoning_content;

        delete choice
          .delta
          .reasoning;

        delete choice
          .delta
          .thinking;
      }

      if (
        choice.message &&
        typeof choice.message ===
          "object"
      ) {
        delete choice
          .message
          .reasoning_content;

        delete choice
          .message
          .reasoning;

        delete choice
          .message
          .thinking;
      }
    }
  }

  return parsed;
}

/* ============================================================
   SSE PROCESSING
============================================================ */

function processSSEEvent(
  event
) {
  if (
    !event ||
    !event.trim()
  ) {
    return "";
  }

  const lines =
    event.split(
      /\r?\n/
    );

  const output = [];

  for (
    const line of lines
  ) {
    /*
     * Preserve comments / non-data SSE lines.
     */
    if (
      !line.startsWith(
        "data:"
      )
    ) {
      output.push(line);
      continue;
    }

    const data =
      line
        .slice(5)
        .trim();

    if (!data) {
      output.push(line);
      continue;
    }

    if (
      data ===
      "[DONE]"
    ) {
      output.push(
        "data: [DONE]"
      );
      continue;
    }

    try {
      const parsed =
        JSON.parse(data);

      output.push(
        "data: " +
          JSON.stringify(
            stripReasoning(
              parsed
            )
          )
      );
    } catch {
      /*
       * If an upstream event isn't JSON,
       * preserve it rather than destroying it.
       */
      output.push(line);
    }
  }

  return output.length
    ? output.join("\n") +
        "\n\n"
    : "";
}

/* ============================================================
   ERROR HELPERS
============================================================ */

function readStream(
  stream
) {
  return new Promise(
    resolve => {
      let output = "";
      let finished = false;

      function finish() {
        if (finished) {
          return;
        }

        finished = true;
        resolve(output);
      }

      stream.on(
        "data",
        chunk => {
          if (
            output.length >=
            MAX_ERROR_BODY_SIZE
          ) {
            return;
          }

          output +=
            chunk.toString(
              "utf8"
            );

          if (
            output.length >
            MAX_ERROR_BODY_SIZE
          ) {
            output =
              output.slice(
                0,
                MAX_ERROR_BODY_SIZE
              );
          }
        }
      );

      stream.on(
        "end",
        finish
      );

      stream.on(
        "close",
        finish
      );

      stream.on(
        "error",
        error => {
          console.error(
            "Error while reading NVIDIA error response:",
            error?.message ||
              error
          );

          finish();
        }
      );
    }
  );
}

function extractErrorMessage(
  data
) {
  if (!data) {
    return null;
  }

  if (
    typeof data ===
    "object"
  ) {
    return (
      data.error?.message ||
      data.message ||
      null
    );
  }

  try {
    const parsed =
      JSON.parse(data);

    return (
      parsed?.error?.message ||
      parsed?.message ||
      null
    );
  } catch {
    return String(data);
  }
}

function sendError(
  res,
  status,
  message,
  details
) {
  if (
    res.headersSent
  ) {
    try {
      res.end();
    } catch {}

    return;
  }

  const response = {
    error: {
      message:
        String(message)
    }
  };

  if (
    details !== undefined &&
    details !== null
  ) {
    response.error.details =
      details;
  }

  return res
    .status(status)
    .json(response);
}

/* ============================================================
   NVIDIA MODEL DISCOVERY
============================================================ */

let modelCache = null;
let modelCacheTimestamp = 0;
let modelCachePromise = null;

async function fetchNimModels(
  forceRefresh = false
) {
  const now =
    Date.now();

  /*
   * Normal cached result.
   */
  if (
    !forceRefresh &&
    Array.isArray(
      modelCache
    ) &&
    now -
      modelCacheTimestamp <
      MODEL_CACHE_TTL
  ) {
    return modelCache;
  }

  /*
   * Coalesce simultaneous discovery requests.
   *
   * This is NOT retry logic.
   *
   * Multiple callers arriving simultaneously share the
   * same outstanding /models request.
   */
  if (
    modelCachePromise &&
    !forceRefresh
  ) {
    return modelCachePromise;
  }

  modelCachePromise =
    (async () => {
      try {
        if (
          !NIM_API_KEY
        ) {
          return Array.isArray(
            modelCache
          )
            ? modelCache
            : [];
        }

        /*
         * Exactly ONE discovery request.
         *
         * There is no retry.
         */
        const response =
          await axios.get(
            NIM_API_BASE +
              "/models",
            {
              headers: {
                Authorization:
                  "Bearer " +
                  NIM_API_KEY,

                Accept:
                  "application/json"
              },

              timeout:
                Math.min(
                  NIM_TIMEOUT,
                  30000
                ),

              httpAgent,
              httpsAgent,

              validateStatus:
                () => true
            }
          );

        if (
          response.status >=
            200 &&
          response.status <
            300 &&
          Array.isArray(
            response
              .data
              ?.data
          )
        ) {
          modelCache =
            response.data.data;

          modelCacheTimestamp =
            Date.now();

          return modelCache;
        }

        console.warn(
          "NVIDIA /models returned HTTP",
          response.status
        );

        return Array.isArray(
          modelCache
        )
          ? modelCache
          : [];
      } catch (error) {
        console.warn(
          "Unable to refresh NVIDIA model list:",
          error?.message ||
            error
        );

        return Array.isArray(
          modelCache
        )
          ? modelCache
          : [];
      } finally {
        modelCachePromise =
          null;
      }
    })();

  return modelCachePromise;
}

/* ============================================================
   NIM HTTP REQUEST CONFIGURATION
============================================================ */

function createNimAxiosConfig(
  stream
) {
  return {
    headers: {
      Authorization:
        "Bearer " +
        NIM_API_KEY,

      "Content-Type":
        "application/json",

      Accept:
        stream
          ? "text/event-stream"
          : "application/json"
    },

    /*
     * IMPORTANT:
     *
     * Axios itself does not retry requests.
     *
     * There is:
     *
     *   no retry adapter
     *   no retry interceptor
     *   no backoff
     *   no retry count
     *   no retry loop
     */
    timeout:
      NIM_TIMEOUT,

    httpAgent,
    httpsAgent,

    validateStatus:
      () => true
  };
}

/* ============================================================
   NIM COMPLETION REQUEST
============================================================ */

/*
 * HARD NO-RETRY GUARANTEE:
 *
 * This function contains exactly ONE
 *
 *     axios.post(... /chat/completions ...)
 *
 * call.
 *
 * There is deliberately:
 *
 *   - no retry loop
 *   - no retry interceptor
 *   - no retry adapter
 *   - no exponential backoff
 *   - no retry on 429
 *   - no retry on 5xx
 *   - no retry on timeout
 *   - no retry on ECONNRESET
 *   - no retry after stream failure
 *   - no retry after client disconnect
 */
async function requestNim(
  nimRequest,
  stream,
  metrics
) {
  metrics.upstreamRequestStartedAt =
    Date.now();

  const response =
    await axios.post(
      NIM_API_BASE +
        "/chat/completions",

      nimRequest,

      stream
        ? {
            ...createNimAxiosConfig(
              true
            ),

            responseType:
              "stream"
          }
        : createNimAxiosConfig(
            false
          )
    );

  metrics.upstreamHeadersAt =
    Date.now();

  return response;
}

/* ============================================================
   EXPRESS
============================================================ */

app.disable(
  "x-powered-by"
);

app.use(
  cors()
);

app.use(
  express.json({
    limit: "100mb"
  })
);

/* ============================================================
   ROOT
============================================================ */

app.get(
  "/",
  function (
    req,
    res
  ) {
    res.json({
      status:
        "online",

      service:
        "JanitorAI -> NVIDIA NIM Proxy",

      default_model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(
          MODELS
        ),

      proxy_profiles:
        Object.keys(
          PROFILES
        ),

      upstream_model_count:
        Array.isArray(
          modelCache
        )
          ? modelCache.length
          : 0,

      unknown_models_allowed:
        ALLOW_UNKNOWN_MODELS,

      default_reasoning_effort:
        DEFAULT_REASONING_EFFORT,

      default_reasoning_budget:
        DEFAULT_REASONING_BUDGET,

      nemotron_thinking_default:
        DEFAULT_NEMOTRON_THINKING,

      reasoning_stripped_from_response:
        STRIP_REASONING_FROM_RESPONSE,

      nim_api_base:
        NIM_API_BASE,

      timeout_ms:
        NIM_TIMEOUT,

      nim_completion_retries:
        0,

      kimi_memory:
        false,

      janitor_context_passthrough:
        true
    });
  }
);

/* ============================================================
   HEALTH
============================================================ */

app.get(
  "/health",
  function (
    req,
    res
  ) {
    res.json({
      ok:
        true,

      model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(
          MODELS
        ),

      proxy_profiles:
        Object.keys(
          PROFILES
        ),

      upstream_model_count:
        Array.isArray(
          modelCache
        )
          ? modelCache.length
          : 0,

      unknown_models_allowed:
        ALLOW_UNKNOWN_MODELS,

      reasoning:
        DEFAULT_REASONING_EFFORT,

      nemotron_thinking:
        DEFAULT_NEMOTRON_THINKING,

      kimi_reasoning:
        "always_enabled",

      max_tokens:
        DEFAULT_MAX_TOKENS,

      reasoning_response_stripping:
        STRIP_REASONING_FROM_RESPONSE,

      kimi_memory:
        false,

      janitor_context_passthrough:
        true,

      nim_timeout_ms:
        NIM_TIMEOUT,

      nim_completion_retries:
        0
    });
  }
);

/* ============================================================
   NVIDIA MODEL LIST
============================================================ */

app.get(
  "/v1/models",
  async function (
    req,
    res
  ) {
    try {
      const upstreamModels =
        await fetchNimModels();

      const seen =
        new Set();

      const data = [];

      /*
       * LOCAL MODELS FIRST
       *
       * This is important.
       *
       * If we simply returned NVIDIA's /models response,
       * Janitor would never see our virtual profiles.
       */
      for (
        const entry of
          localModelEntries()
      ) {
        const id =
          normalizeModel(
            entry.id
          );

        if (
          seen.has(id)
        ) {
          continue;
        }

        seen.add(id);
        data.push(entry);
      }

      /*
       * Then preserve NVIDIA's real model list.
       */
      for (
        const entry of
          upstreamModels
      ) {
        if (
          !entry ||
          typeof entry.id !==
            "string"
        ) {
          continue;
        }

        const id =
          normalizeModel(
            entry.id
          );

        if (
          seen.has(id)
        ) {
          continue;
        }

        seen.add(id);
        data.push(entry);
      }

      return res.json({
        object:
          "list",

        data
      });
    } catch (error) {
      return sendError(
        res,
        502,
        "Unable to retrieve NVIDIA NIM models.",
        error?.message ||
          String(error)
      );
    }
  }
);

/* ============================================================
   FORCE MODEL CACHE REFRESH
============================================================ */

app.post(
  "/v1/models/refresh",
  async function (
    req,
    res
  ) {
    try {
      const models =
        await fetchNimModels(
          true
        );

      return res.json({
        object:
          "list",

        data:
          models,

        count:
          models.length
      });
    } catch (error) {
      return sendError(
        res,
        502,
        "Unable to refresh NVIDIA NIM model list.",
        error?.message ||
          String(error)
      );
    }
  }
);

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post(
  "/v1/chat/completions",
  async function (
    req,
    res
  ) {
    /*
     * These timestamps are intentionally kept server-side
     * and contain no RP content.
     */
    const requestReceivedAt =
      Date.now();

    const metrics = {
      requestReceivedAt,

      upstreamRequestStartedAt:
        null,

      upstreamHeadersAt:
        null,

      firstUpstreamDataAt:
        null,

      firstContentAt:
        null,

      completedAt:
        null
    };

    try {
      if (
        !NIM_API_KEY
      ) {
        return sendError(
          res,
          500,
          "NIM_API_KEY is not configured."
        );
      }

      const incoming =
        isPlainObject(
          req.body
        )
          ? req.body
          : {};

      const requestedModel =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim()
          ? incoming.model.trim()
          : DEFAULT_MODEL;

      const resolved =
        resolveModel(
          requestedModel
        );

      const modelConfig =
        getModelConfig(
          requestedModel
        );

      /*
       * Preserve permissive unknown-model behavior.
       */
      if (
        !isSupportedModel(
          requestedModel
        )
      ) {
        return sendError(
          res,
          400,
          "Unsupported model: " +
            requestedModel,
          {
            explicitly_configured_models:
              Object.keys(
                MODELS
              ),

            proxy_profiles:
              Object.keys(
                PROFILES
              )
          }
        );
      }

      const messages =
        normalizeMessages(
          incoming.messages
        );

      if (
        !messages.length
      ) {
        return sendError(
          res,
          400,
          "No valid messages were supplied."
        );
      }

      const stream =
        parseBoolean(
          incoming.stream,
          true
        );

      /*
       * Build exactly one request.
       */
      const nimRequest =
        buildNimRequest(
          incoming,
          requestedModel,
          messages,
          stream
        );

      if (
        DEBUG_PROXY
      ) {
        console.log(
          "=================================================="
        );

        console.log(
          "JANITORAI REQUEST"
        );

        console.log(
          "REQUESTED MODEL:",
          requestedModel
        );

        console.log(
          "NIM MODEL:",
          nimRequest.model
        );

        console.log(
          "PROFILE:",
          resolved.profile?.label ||
            "base"
        );

        console.log(
          "JANITOR MESSAGES:",
          messages.length
        );

        console.log(
          "TEMPERATURE:",
          nimRequest.temperature
        );

        console.log(
          "TOP_P:",
          nimRequest.top_p
        );

        console.log(
          "MAX_TOKENS:",
          nimRequest.max_tokens
        );

        console.log(
          "STREAM:",
          nimRequest.stream
        );

        console.log(
          "REASONING_EFFORT:",
          nimRequest.reasoning_effort
        );

        console.log(
          "REASONING_BUDGET:",
          nimRequest.reasoning_budget
        );

        console.log(
          "CHAT_TEMPLATE_KWARGS:",
          nimRequest
            .chat_template_kwargs
        );

        console.log(
          "NIM COMPLETION REQUESTS: 1"
        );

        console.log(
          "NIM COMPLETION RETRIES: 0"
        );

        console.log(
          "=================================================="
        );
      }

      /*
       * ========================================================
       *
       * EXACTLY ONE NIM COMPLETION REQUEST.
       *
       * There is no retry anywhere below.
       *
       * ========================================================
       */
      const response =
        await requestNim(
          nimRequest,
          stream,
          metrics
        );

      /* ======================================================
         UPSTREAM ERROR
      ====================================================== */

      if (
        response.status < 200 ||
        response.status >= 300
      ) {
        let errorBody =
          "";

        if (
          response.data &&
          typeof response
            .data
            .on ===
            "function"
        ) {
          errorBody =
            await readStream(
              response.data
            );
        } else if (
          typeof response.data ===
          "string"
        ) {
          errorBody =
            response.data;
        } else {
          errorBody =
            safeJson(
              response.data
            );
        }

        const upstreamMessage =
          extractErrorMessage(
            errorBody
          );

        console.error(
          "NVIDIA NIM ERROR",
          {
            model:
              nimRequest.model,

            profile:
              resolved.profile
                ?.label ||
              null,

            http:
              response.status,

            message:
              upstreamMessage,

            retry:
              false,

            retries:
              0,

            upstream_headers_ms:
              metrics
                .upstreamHeadersAt -
              requestReceivedAt
          }
        );

        /*
         * We deliberately DO NOT retry here.
         *
         * A 429/500/502/503/etc. is simply returned.
         */
        const proxyStatus =
          response.status >=
          500
            ? 502
            : response.status;

        return sendError(
          res,
          proxyStatus,
          upstreamMessage ||
            "NVIDIA NIM returned HTTP " +
              response.status +
              ".",
          {
            upstream_status:
              response.status,

            upstream_body:
              errorBody,

            retries:
              0
          }
        );
      }

      /* ======================================================
         NON-STREAMING
      ====================================================== */

      if (!stream) {
        metrics.completedAt =
          Date.now();

        if (
          DEBUG_PROXY
        ) {
          console.log(
            "NIM COMPLETE",
            {
              model:
                nimRequest.model,

              profile:
                resolved.profile
                  ?.label ||
                "base",

              upstream_headers_ms:
                metrics
                  .upstreamHeadersAt -
                requestReceivedAt,

              total_ms:
                metrics
                  .completedAt -
                requestReceivedAt
            }
          );
        }

        return res
          .status(200)
          .json(
            stripReasoning(
              response.data
            )
          );
      }

      /* ======================================================
         STREAMING
      ====================================================== */

      const upstream =
        response.data;

      if (
        !upstream ||
        typeof upstream.on !==
          "function"
      ) {
        return sendError(
          res,
          502,
          "NVIDIA returned an invalid streaming response."
        );
      }

      /*
       * Do not send headers until we know the upstream
       * response is actually streamable.
       */
      res.status(200);

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

      if (
        typeof res.flushHeaders ===
        "function"
      ) {
        res.flushHeaders();
      }

      let buffer =
        "";

      let ended =
        false;

      let clientDisconnected =
        false;

      let sawContent =
        false;

      function destroyUpstream() {
        try {
          upstream.destroy();
        } catch {}
      }

      function disconnect() {
        if (
          clientDisconnected
        ) {
          return;
        }

        clientDisconnected =
          true;

        destroyUpstream();

        /*
         * IMPORTANT:
         *
         * Client cancellation is termination.
         *
         * It is NEVER a reason to issue another
         * completion request.
         */
      }

      req.on(
        "aborted",
        disconnect
      );

      res.on(
        "close",
        function () {
          if (
            !res.writableEnded
          ) {
            disconnect();
          }
        }
      );

      upstream.on(
        "data",
        function (chunk) {
          if (
            ended ||
            clientDisconnected
          ) {
            return;
          }

          /*
           * First byte / first upstream chunk.
           */
          if (
            metrics
              .firstUpstreamDataAt ===
            null
          ) {
            metrics
              .firstUpstreamDataAt =
              Date.now();

            if (
              DEBUG_PROXY
            ) {
              console.log(
                "NIM FIRST UPSTREAM DATA",
                {
                  model:
                    nimRequest.model,

                  profile:
                    resolved.profile
                      ?.label ||
                    "base",

                  ms_from_request:
                    metrics
                      .firstUpstreamDataAt -
                    requestReceivedAt,

                  ms_from_upstream_request:
                    metrics
                      .firstUpstreamDataAt -
                    metrics
                      .upstreamRequestStartedAt
                }
              );
            }
          }

          buffer +=
            chunk.toString(
              "utf8"
            );

          let separatorIndex;

          while (
            (
              separatorIndex =
                buffer.search(
                  /\r?\n\r?\n/
                )
            ) !== -1
          ) {
            const event =
              buffer.slice(
                0,
                separatorIndex
              );

            const separatorLength =
              buffer[
                separatorIndex
              ] === "\r"
                ? 4
                : 2;

            buffer =
              buffer.slice(
                separatorIndex +
                  separatorLength
              );

            /*
             * This measures the first JSON SSE event.
             *
             * It is a practical proxy-side TTFT measurement.
             */
            if (
              !sawContent &&
              /(^|\n)data:\s*\{/.test(
                event
              )
            ) {
              metrics
                .firstContentAt =
                Date.now();

              sawContent =
                true;
            }

            /*
             * IMPORTANT PERFORMANCE PATH:
             *
             * If reasoning stripping is disabled,
             * do NOT parse JSON and stringify it again.
             *
             * NVIDIA -> Render -> Janitor
             *
             * becomes almost direct SSE forwarding.
             */
            const output =
              STRIP_REASONING_FROM_RESPONSE
                ? processSSEEvent(
                    event
                  )
                : event
                    ? event +
                      "\n\n"
                    : "";

            if (!output) {
              continue;
            }

            try {
              res.write(
                output
              );
            } catch {
              disconnect();
              break;
            }
          }
        }
      );

      upstream.on(
        "end",
        function () {
          if (ended) {
            return;
          }

          ended =
            true;

          metrics.completedAt =
            Date.now();

          /*
           * Flush any remaining partial SSE event.
           */
          if (
            buffer.trim() &&
            !clientDisconnected
          ) {
            const output =
              STRIP_REASONING_FROM_RESPONSE
                ? processSSEEvent(
                    buffer
                  )
                : buffer +
                  "\n\n";

            if (output) {
              try {
                res.write(
                  output
                );
              } catch {}
            }
          }

          if (
            !clientDisconnected
          ) {
            try {
              res.end();
            } catch {}
          }

          if (
            DEBUG_PROXY
          ) {
            console.log(
              "NIM STREAM COMPLETE",
              {
                model:
                  nimRequest.model,

                profile:
                  resolved.profile
                    ?.label ||
                  "base",

                upstream_headers_ms:
                  metrics
                    .upstreamHeadersAt -
                  requestReceivedAt,

                first_data_ms:
                  metrics
                    .firstUpstreamDataAt ===
                  null
                    ? null
                    : metrics
                        .firstUpstreamDataAt -
                      requestReceivedAt,

                first_event_ms:
                  metrics
                    .firstContentAt ===
                  null
                    ? null
                    : metrics
                        .firstContentAt -
                      requestReceivedAt,

                total_ms:
                  metrics
                    .completedAt -
                  requestReceivedAt
              }
            );
          }
        }
      );

      upstream.on(
        "error",
        function (error) {
          if (ended) {
            return;
          }

          ended =
            true;

          console.error(
            "NVIDIA stream error:",
            error?.message ||
              error
          );

          console.error(
            "NIM COMPLETION RETRIES: 0"
          );

          /*
           * NEVER issue another NIM request.
           */
          if (
            !res.headersSent
          ) {
            return sendError(
              res,
              502,
              "NVIDIA streaming connection failed.",
              error?.message ||
                String(error)
            );
          }

          try {
            res.end();
          } catch {}
        }
      );
    } catch (error) {
      metrics.completedAt =
        Date.now();

      console.error(
        "=================================================="
      );

      console.error(
        "PROXY ERROR"
      );

      console.error(
        error?.stack ||
          error?.message ||
          error
      );

      console.error(
        "NIM COMPLETION RETRIES: 0"
      );

      console.error(
        "=================================================="
      );

      /*
       * If streaming headers were already sent,
       * there is no safe JSON error response to send.
       *
       * End the stream.
       */
      if (
        res.headersSent
      ) {
        try {
          res.end();
        } catch {}

        return;
      }

      if (
        error?.code ===
        "ECONNABORTED"
      ) {
        return sendError(
          res,
          504,
          "NVIDIA NIM request timed out.",
          {
            timeout_ms:
              NIM_TIMEOUT,

            retries:
              0
          }
        );
      }

      if (
        error?.code ===
        "ECONNRESET"
      ) {
        return sendError(
          res,
          502,
          "Connection to NVIDIA NIM was reset.",
          {
            retries:
              0
          }
        );
      }

      if (
        error?.code ===
        "ETIMEDOUT"
      ) {
        return sendError(
          res,
          504,
          "Connection to NVIDIA NIM timed out.",
          {
            retries:
              0
          }
        );
      }

      return sendError(
        res,
        500,
        "Proxy request failed.",
        {
          message:
            error?.message ||
            String(error),

          retries:
            0
        }
      );
    }
  }
);

/* ============================================================
   404
============================================================ */

app.use(
  function (
    req,
    res
  ) {
    return sendError(
      res,
      404,
      "Endpoint not found."
    );
  }
);

/* ============================================================
   EXPRESS ERROR HANDLER
============================================================ */

app.use(
  function (
    error,
    req,
    res,
    next
  ) {
    console.error(
      "Unhandled Express error:",
      error?.stack ||
        error?.message ||
        error
    );

    if (
      res.headersSent
    ) {
      return next(error);
    }

    return sendError(
      res,
      500,
      "Internal proxy error.",
      error?.message ||
        null
    );
  }
);

/* ============================================================
   START SERVER
============================================================ */

const server =
  app.listen(
    PORT,
    "0.0.0.0",
    function () {
      console.log(
        "=================================================="
      );

      console.log(
        "JanitorAI -> NVIDIA NIM Proxy"
      );

      console.log(
        "Listening on 0.0.0.0:" +
          PORT
      );

      console.log(
        "Default model:",
        DEFAULT_MODEL
      );

      console.log(
        "NIM endpoint:",
        NIM_API_BASE
      );

      console.log(
        "NIM timeout:",
        NIM_TIMEOUT +
          "ms"
      );

      console.log(
        "NIM completion retries: 0 (HARD DISABLED)"
      );

      console.log(
        "Kimi reasoning: ALWAYS ENABLED"
      );

      console.log(
        "Nemotron default thinking:",
        DEFAULT_NEMOTRON_THINKING
      );

      console.log(
        "Reasoning response stripping:",
        STRIP_REASONING_FROM_RESPONSE
      );

      console.log(
        "Proxy profiles:",
        Object.keys(
          PROFILES
        ).length
      );

      console.log(
        "Janitor context passthrough: ENABLED"
      );

      console.log(
        "=================================================="
      );

      /*
       * Initial model discovery.
       *
       * This is NOT a completion request and NOT a retry.
       *
       * It simply populates the /v1/models cache.
       */
      fetchNimModels()
        .then(
          models => {
            console.log(
              "NVIDIA reports",
              models.length,
              "available model(s)."
            );
          }
        )
        .catch(
          error => {
            console.warn(
              "Initial NVIDIA model discovery failed:",
              error?.message ||
                error
            );
          }
        );
    }
  );

/* ============================================================
   LONG-RUNNING AI REQUEST SETTINGS
============================================================ */

server.timeout =
  0;

server.requestTimeout =
  0;

server.keepAliveTimeout =
  Math.max(
    65000,
    Math.min(
      NIM_TIMEOUT + 5000,
      120000
    )
  );

server.headersTimeout =
  Math.max(
    66000,
    Math.min(
      NIM_TIMEOUT + 10000,
      125000
    )
  );

/* ============================================================
   GRACEFUL SHUTDOWN
============================================================ */

function shutdown(
  signal
) {
  console.log(
    signal +
      " received. Shutting down..."
  );

  server.close(
    function () {
      console.log(
        "Server closed."
      );

      process.exit(0);
    }
  );

  setTimeout(
    function () {
      console.error(
        "Forced shutdown."
      );

      process.exit(1);
    },
    10000
  ).unref();
}

process.on(
  "SIGTERM",
  () =>
    shutdown(
      "SIGTERM"
    )
);

process.on(
  "SIGINT",
  () =>
    shutdown(
      "SIGINT"
    )
);
