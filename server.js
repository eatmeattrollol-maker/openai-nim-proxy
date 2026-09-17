"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");

const app = express();

/* ============================================================
   CONFIGURATION
============================================================ */

const PORT = Number(process.env.PORT) || 10000;

const NIM_API_BASE = (
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NIM_API_KEY =
  process.env.NIM_API_KEY ||
  process.env.NVIDIA_API_KEY ||
  "";

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "z-ai/glm-5.3";

const ALLOW_UNKNOWN_MODELS =
  parseBoolean(
    process.env.ALLOW_UNKNOWN_MODELS,
    true
  );

const DEBUG_PROXY =
  parseBoolean(
    process.env.DEBUG_PROXY,
    false
  );

const STRIP_REASONING_FROM_RESPONSE =
  parseBoolean(
    process.env.STRIP_REASONING_FROM_RESPONSE,
    true
  );

const NIM_TIMEOUT =
  Number.isFinite(
    Number(process.env.NIM_TIMEOUT_MS)
  )
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const MODEL_CACHE_TTL =
  Number.isFinite(
    Number(process.env.MODEL_CACHE_TTL_MS)
  )
    ? Number(process.env.MODEL_CACHE_TTL_MS)
    : 300000;

const MAX_ERROR_BODY_SIZE =
  2 * 1024 * 1024;

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
 * NVIDIA's Nemotron 3 Ultra API defaults thinking on.
 *
 * Profiles can override this explicitly. The new -no
 * profile is the safest way to force concise/non-reasoning output.
 */
const DEFAULT_NEMOTRON_THINKING =
  parseBoolean(
    process.env.NEMOTRON_ENABLE_THINKING,
    true
  );

/* ============================================================
   HTTP AGENTS
============================================================ */

const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 64,
  maxFreeSockets: 16,
  scheduling: "lifo"
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 64,
  maxFreeSockets: 16,
  scheduling: "lifo"
});

/* ============================================================
   HELPERS
============================================================ */

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
      value.trim().toLowerCase();

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

function clamp(
  value,
  min,
  max
) {
  return Math.min(
    max,
    Math.max(min, value)
  );
}

/* ============================================================
   MODEL CONFIGURATION
============================================================ */

const MODELS = {
  "z-ai/glm-5.3": {
    name: "GLM 5.3",
    provider: "Z.ai",
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],
    defaultReasoningEffort: "high",
    maxOutputTokens: 32768,
    maxReasoningBudget: null,
    contextTokens: 1048576,
    adapter: "glm53"
  },

  "z-ai/glm-5.3-flash": {
    name: "GLM 5.3 Flash",
    provider: "Z.ai",
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],
    defaultReasoningEffort: "max",
    maxOutputTokens: 32768,
    maxReasoningBudget: null,
    contextTokens: 1048576,
    adapter: "glm53-flash"
  },

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],
    defaultReasoningEffort: "max",
    maxOutputTokens: 8192,
    maxReasoningBudget: 32768,
    adapter: "kimi-k3"
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name:
      "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",

    /*
     * Kept for proxy metadata compatibility.
     *
     * IMPORTANT: these are NOT forwarded as
     * reasoning_effort to Nemotron.
     *
     * Nemotron 3 Ultra uses chat_template_kwargs
     * and thinking_token_budget instead.
     */
    reasoningLevels: [
      "none",
      "medium",
      "high"
    ],

    defaultReasoningEffort: "high",
    defaultThinking: true,

    maxReasoningBudget: 32768,
    maxOutputTokens: 32768,
    contextTokens: 1048576,
    adapter: "nemotron"
  },

  "deepseek-ai/deepseek-v4-flash-0731": {
    name:
      "DeepSeek V4 Flash 0731",
    provider: "DeepSeek AI",
    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],
    defaultReasoningEffort: "max",
    maxOutputTokens: 8192,
    maxReasoningBudget: 16384,
    adapter: "deepseek-flash"
  }
};

/* ============================================================
   PROXY PROFILES
============================================================ */

const PROFILES = {
  /* ---------------- GLM 5.3 ---------------- */

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

  /* ---------------- GLM 5.3 FLASH ---------------- */

  "z-ai/glm-5.3-flash-fast": {
    baseModel:
      "z-ai/glm-5.3-flash",
    label: "Fast",
    reasoningEffort: "low"
  },

  "z-ai/glm-5.3-flash-balanced": {
    baseModel:
      "z-ai/glm-5.3-flash",
    label: "Balanced",
    reasoningEffort: "high"
  },

  "z-ai/glm-5.3-flash-deep": {
    baseModel:
      "z-ai/glm-5.3-flash",
    label: "Deep",
    reasoningEffort: "max"
  },

  /* ---------------- KIMI K3 ---------------- */

  "moonshotai/kimi-k3-fast": {
    baseModel: "moonshotai/kimi-k3",
    label: "Fast",
    reasoningEffort: "low"
  },

  "moonshotai/kimi-k3-balanced": {
    baseModel: "moonshotai/kimi-k3",
    label: "Balanced",
    reasoningEffort: "high"
  },

  "moonshotai/kimi-k3-deep": {
    baseModel: "moonshotai/kimi-k3",
    label: "Deep",
    reasoningEffort: "max"
  },

  /* ---------------- NEMOTRON 3 ULTRA ---------------- */

  /*
   * Explicit no-thinking profile.
   *
   * This sends only:
   *
   *   chat_template_kwargs:
   *     enable_thinking: false
   *
   * No reasoning budget is sent.
   */
  "nvidia/nemotron-3-ultra-550b-a55b-no": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label: "No Thinking",
    thinking: false
  },

  /*
   * Fast remains available for compatibility and is equivalent
   * to No Thinking.
   */
  "nvidia/nemotron-3-ultra-550b-a55b-fast": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Fast",
    thinking: false
  },

  /*
   * Balanced uses NVIDIA's documented medium_effort hint.
   */
  "nvidia/nemotron-3-ultra-550b-a55b-balanced": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Balanced",
    thinking: true,
    mediumEffort: true
  },

  /*
   * Deep uses full thinking and an explicit thinking-token
   * budget.
   */
  "nvidia/nemotron-3-ultra-550b-a55b-deep": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Deep",
    thinking: true,
    thinkingTokenBudget: 32768
  },

  /* ---------------- DEEPSEEK ---------------- */

  "deepseek-ai/deepseek-v4-flash-0731-no": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label: "No Thinking",
    reasoningEffort: "none"
  },

  "deepseek-ai/deepseek-v4-flash-0731-fast": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label: "Fast",
    reasoningEffort: "low"
  },

  "deepseek-ai/deepseek-v4-flash-0731-balanced": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label: "Balanced",
    reasoningEffort: "high"
  },

  "deepseek-ai/deepseek-v4-flash-0731-deep": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",
    label: "Deep",
    reasoningEffort: "max"
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
    requestedModel: requested,
    baseModel: profile
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
   MODEL LIST
============================================================ */

function publicModelEntry(
  id,
  config,
  extra
) {
  return {
    id,
    object: "model",
    created: Math.floor(
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

  for (
    const [
      id,
      config
    ] of Object.entries(MODELS)
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

          proxy_profile: false
        }
      )
    );
  }

  for (
    const [
      id,
      profile
    ] of Object.entries(PROFILES)
  ) {
    const config =
      MODELS[
        normalizeModel(
          profile.baseModel
        )
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

          proxy_profile: true,

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

  const normalized =
    normalizeModel(
      resolved.baseModel
    );

  const maximum =
    normalized ===
    "moonshotai/kimi-k3"
      ? 1
      : 2;

  return clamp(
    numberOrDefault(
      value,
      DEFAULT_TEMPERATURE
    ),
    0,
    maximum
  );
}

function clampTopP(value) {
  return clamp(
    numberOrDefault(
      value,
      DEFAULT_TOP_P
    ),
    0,
    1
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

  return clamp(
    Math.floor(
      numberOrDefault(
        value,
        DEFAULT_REASONING_BUDGET
      )
    ),
    0,
    maximum
  );
}

/* ============================================================
   GENERIC REASONING HELPERS
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
      value.trim().toLowerCase();

    if (
      levels.includes(normalized)
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
   * Kimi reasoning cannot be disabled.
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
   NEMOTRON-SPECIFIC THINKING RESOLUTION
============================================================ */

/*
 * NVIDIA Nemotron 3 Ultra uses:
 *
 *   chat_template_kwargs.enable_thinking
 *   chat_template_kwargs.medium_effort
 *   thinking_token_budget
 *
 * The generic reasoning_effort/reasoning_budget fields are NOT
 * forwarded to this adapter.
 */

function getNemotronThinking(
  incoming,
  profile
) {
  if (
    profile &&
    profile.thinking !== undefined
  ) {
    return Boolean(
      profile.thinking
    );
  }

  if (
    incoming.enable_thinking !==
    undefined
  ) {
    return parseBoolean(
      incoming.enable_thinking,
      DEFAULT_NEMOTRON_THINKING
    );
  }

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
    return incoming
      .chat_template_kwargs
      .enable_thinking;
  }

  return DEFAULT_NEMOTRON_THINKING;
}

function getNemotronMediumEffort(
  incoming,
  profile,
  thinking
) {
  if (!thinking) {
    return false;
  }

  if (
    profile &&
    profile.mediumEffort !==
      undefined
  ) {
    return Boolean(
      profile.mediumEffort
    );
  }

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    ) &&
    typeof
      incoming
        .chat_template_kwargs
        .medium_effort ===
      "boolean"
  ) {
    return incoming
      .chat_template_kwargs
      .medium_effort;
  }

  return false;
}

function normalizeThinkingBudget(
  value
) {
  if (
    !isFiniteNumber(value)
  ) {
    return null;
  }

  return clamp(
    Math.floor(
      Number(value)
    ),
    1,
    32768
  );
}

function getNemotronThinkingTokenBudget(
  incoming,
  profile,
  thinking
) {
  if (!thinking) {
    return null;
  }

  if (
    profile &&
    profile.thinkingTokenBudget !==
      undefined
  ) {
    return normalizeThinkingBudget(
      profile.thinkingTokenBudget
    );
  }

  /*
   * Preferred modern request field.
   */
  if (
    incoming.thinking_token_budget !==
      undefined
  ) {
    return normalizeThinkingBudget(
      incoming.thinking_token_budget
    );
  }

  /*
   * Backward-compatible client input.
   *
   * Accept it, but translate it to the actual Nemotron request
   * field rather than forwarding reasoning_budget.
   */
  if (
    incoming.reasoning_budget !==
      undefined
  ) {
    return normalizeThinkingBudget(
      incoming.reasoning_budget
    );
  }

  return null;
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

  return messages.filter(
    message =>
      isPlainObject(message) &&
      typeof message.role ===
        "string" &&
      message.role.trim()
  ).map(
    message => ({
      ...message,
      role:
        message.role.trim()
    })
  );
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
   * Kimi keeps the original proxy behavior of omitting top_p.
   */
  if (!isKimi) {
    request.top_p =
      clampTopP(
        incoming.top_p
      );
  }

  /*
   * Optional sampling controls.
   */
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

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    )
  ) {
    request.chat_template_kwargs = {
      ...incoming.chat_template_kwargs
    };
  }

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

      clear_thinking: true
    };

    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.reasoning_mode;
    delete request.nvext;

    return request;
  }

  /* ==========================================================
     GLM 5.3 FLASH
  ========================================================== */

  if (
    normalizedModel ===
    "z-ai/glm-5.3-flash"
  ) {
    let reasoningEffort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig,
        profile
      );

    if (
      reasoningEffort ===
      "none"
    ) {
      reasoningEffort = "low";
    }

    request.reasoning_effort =
      reasoningEffort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      clear_thinking: true,
      reasoning_effort:
        reasoningEffort
    };

    delete request.reasoning_budget;
    delete request.enable_thinking;
    delete request.reasoning_mode;
    delete request.nvext;

    return request;
  }

  /* ==========================================================
     NEMOTRON 3 ULTRA
  ========================================================== */

  if (
    normalizedModel ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    /*
     * CRITICAL FIX:
     *
     * Do NOT send:
     *
     *   reasoning_effort
     *   reasoning_budget
     *   reasoning_mode
     *   enable_thinking as a top-level field
     *   nvext
     *
     * for this Nemotron adapter.
     *
     * NVIDIA's current Nemotron 3 Ultra API examples control
     * thinking through chat_template_kwargs and use the top-level
     * thinking_token_budget field for explicit budgets.
     */

    const enableThinking =
      getNemotronThinking(
        incoming,
        profile
      );

    const mediumEffort =
      getNemotronMediumEffort(
        incoming,
        profile,
        enableThinking
      );

    const thinkingTokenBudget =
      getNemotronThinkingTokenBudget(
        incoming,
        profile,
        enableThinking
      );

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      enable_thinking:
        enableThinking
    };

    /*
     * A no-thinking profile must not carry any thinking controls.
     */
    if (
      enableThinking &&
      mediumEffort
    ) {
      request.chat_template_kwargs = {
        ...request.chat_template_kwargs,

        medium_effort: true
      };
    } else {
      delete request
        .chat_template_kwargs
        .medium_effort;
    }

    /*
     * NVIDIA's Nemotron 3 Ultra Chat Completions examples use
     * thinking_token_budget as a top-level request field.
     *
     * It is intentionally NOT duplicated inside
     * chat_template_kwargs.
     */
    delete request
      .chat_template_kwargs
      .thinking_token_budget;

    if (
      enableThinking &&
      thinkingTokenBudget !==
        null
    ) {
      request.thinking_token_budget =
        thinkingTokenBudget;
    } else {
      delete request.thinking_token_budget;
    }

    /*
     * With tools + reasoning, NVIDIA documents
     * force_nonempty_content in chat_template_kwargs.
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
    } else {
      delete request
        .chat_template_kwargs
        .force_nonempty_content;
    }

    /*
     * Strip all generic/legacy reasoning controls that may have
     * arrived from JanitorAI, extra_body, or an older client.
     */
    delete request.reasoning_effort;
    delete request.reasoning_budget;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.nvext;

    delete request
      .chat_template_kwargs
      .reasoning_effort;

    delete request
      .chat_template_kwargs
      .reasoning_budget;

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
     KIMI K3
  ========================================================== */

  if (
    normalizedModel ===
    "moonshotai/kimi-k3"
  ) {
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
        delete choice.delta
          .reasoning_content;

        delete choice.delta
          .reasoning;

        delete choice.delta
          .thinking;
      }

      if (
        choice.message &&
        typeof choice.message ===
          "object"
      ) {
        delete choice.message
          .reasoning_content;

        delete choice.message
          .reasoning;

        delete choice.message
          .thinking;
      }
    }
  }

  return parsed;
}

/* ============================================================
   SSE
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
    event.split(/\r?\n/);

  const output = [];

  for (
    const line of lines
  ) {
    if (
      !line.startsWith("data:")
    ) {
      output.push(line);
      continue;
    }

    const data =
      line.slice(5).trim();

    if (!data) {
      output.push(line);
      continue;
    }

    if (
      data === "[DONE]"
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
            "Error reading NVIDIA error response:",
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
  if (res.headersSent) {
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
   MODEL DISCOVERY
============================================================ */

let modelCache = null;
let modelCacheTimestamp = 0;
let modelCachePromise = null;

async function fetchNimModels(
  forceRefresh = false
) {
  const now = Date.now();

  if (
    !forceRefresh &&
    Array.isArray(modelCache) &&
    now -
      modelCacheTimestamp <
      MODEL_CACHE_TTL
  ) {
    return modelCache;
  }

  if (
    modelCachePromise &&
    !forceRefresh
  ) {
    return modelCachePromise;
  }

  modelCachePromise =
    (async () => {
      try {
        if (!NIM_API_KEY) {
          return Array.isArray(
            modelCache
          )
            ? modelCache
            : [];
        }

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
            response.data?.data
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
   NIM HTTP
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
     * Deliberately no axios retry adapter, retry interceptor,
     * backoff, or retry loop.
     */
    timeout:
      NIM_TIMEOUT,

    httpAgent,
    httpsAgent,

    validateStatus:
      () => true
  };
}

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
      status: "online",

      service:
        "JanitorAI -> NVIDIA NIM Proxy",

      default_model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      proxy_profiles:
        Object.keys(PROFILES),

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

      kimi_memory: false,

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
      ok: true,

      model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      proxy_profiles:
        Object.keys(PROFILES),

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

      kimi_memory: false,

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

      const seen = new Set();
      const data = [];

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
        object: "list",
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
        object: "list",
        data: models,
        count: models.length
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
      if (!NIM_API_KEY) {
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

      const nimRequest =
        buildNimRequest(
          incoming,
          requestedModel,
          messages,
          stream
        );

      if (DEBUG_PROXY) {
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
          "ADAPTER:",
          modelConfig?.adapter ||
            "unknown"
        );

        console.log(
          "PROFILE:",
          resolved.profile?.label ||
            "base"
        );

        console.log(
          "MESSAGES:",
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
          "THINKING_TOKEN_BUDGET:",
          nimRequest.thinking_token_budget
        );

        console.log(
          "CHAT_TEMPLATE_KWARGS:",
          nimRequest
            .chat_template_kwargs
        );

        console.log(
          "FULL NIM REQUEST:",
          safeJson(
            nimRequest
          )
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
       * Exactly one upstream completion request.
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
        let errorBody = "";

        if (
          response.data &&
          typeof response.data.on ===
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

        const diagnostics = {
          model:
            nimRequest.model,

          requested_model:
            requestedModel,

          adapter:
            modelConfig?.adapter ||
            null,

          profile:
            resolved.profile?.label ||
            null,

          http:
            response.status,

          message:
            upstreamMessage,

          retry: false,
          retries: 0,

          reasoning_effort:
            nimRequest.reasoning_effort ??
            null,

          reasoning_budget:
            nimRequest.reasoning_budget ??
            null,

          thinking_token_budget:
            nimRequest
              .thinking_token_budget ??
            null,

          chat_template_kwargs:
            nimRequest
              .chat_template_kwargs ??
            null,

          upstream_headers_ms:
            metrics
              .upstreamHeadersAt -
            requestReceivedAt
        };

        console.error(
          "NVIDIA NIM ERROR",
          diagnostics
        );

        /*
         * Preserve the real upstream status for normal errors.
         * Convert upstream 5xx to proxy 502 so callers can distinguish
         * an upstream failure from a proxy-side failure.
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

            requested_model:
              requestedModel,

            upstream_model:
              nimRequest.model,

            profile:
              resolved.profile?.label ||
              null,

            reasoning_controls: {
              reasoning_effort:
                nimRequest
                  .reasoning_effort ??
                null,

              reasoning_budget:
                nimRequest
                  .reasoning_budget ??
                null,

              thinking_token_budget:
                nimRequest
                  .thinking_token_budget ??
                null,

              chat_template_kwargs:
                nimRequest
                  .chat_template_kwargs ??
                null
            },

            retries: 0
          }
        );
      }

      /* ======================================================
         NON-STREAMING
      ====================================================== */

      if (!stream) {
        metrics.completedAt =
          Date.now();

        if (DEBUG_PROXY) {
          console.log(
            "NIM COMPLETE",
            {
              model:
                nimRequest.model,

              profile:
                resolved.profile?.label ||
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

      let buffer = "";
      let ended = false;
      let clientDisconnected =
        false;
      let sawContent = false;

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

          if (
            metrics
              .firstUpstreamDataAt ===
            null
          ) {
            metrics
              .firstUpstreamDataAt =
              Date.now();

            if (DEBUG_PROXY) {
              console.log(
                "NIM FIRST UPSTREAM DATA",
                {
                  model:
                    nimRequest.model,

                  profile:
                    resolved.profile?.label ||
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

            if (
              !sawContent &&
              /(^|\n)data:\s*\{/.test(
                event
              )
            ) {
              metrics
                .firstContentAt =
                Date.now();

              sawContent = true;
            }

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
              res.write(output);
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

          ended = true;

          metrics.completedAt =
            Date.now();

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

          if (DEBUG_PROXY) {
            console.log(
              "NIM STREAM COMPLETE",
              {
                model:
                  nimRequest.model,

                profile:
                  resolved.profile?.label ||
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

          ended = true;

          console.error(
            "NVIDIA stream error:",
            error?.message ||
              error
          );

          console.error(
            "NIM COMPLETION RETRIES: 0"
          );

          /*
           * Never issue another NIM request after a stream failure.
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
        "PROXY REQUEST ERROR:",
        error?.message ||
          error
      );

      if (
        !res.headersSent
      ) {
        return sendError(
          res,
          502,
          "NVIDIA NIM proxy request failed.",
          error?.message ||
            String(error)
        );
      }

      try {
        res.end();
      } catch {}
    }
  }
);

/* ============================================================
   START SERVER
============================================================ */

const server =
  http.createServer(app);

server.listen(
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
      "Nemotron profiles:",
      [
        "no-think",
        "fast",
        "balanced",
        "deep"
      ]
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
   LONG-RUNNING REQUEST SETTINGS
============================================================ */

server.timeout = 0;
server.requestTimeout = 0;

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
