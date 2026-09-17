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
 * Profiles can override this explicitly. The -no profiles
 * explicitly disable reasoning where the upstream model
 * supports a true non-thinking mode.
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
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  );
}

function numberOrDefault(
  value,
  fallback
) {
  const parsed =
    Number(value);

  return Number.isFinite(parsed)
    ? parsed
    : fallback;
}

function normalizeModel(model) {
  return String(
    model || ""
  ).trim();
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
      "none",
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

    /*
     * NVIDIA documents GLM-5.3-Flash as always reasoning.
     * "low" is its minimum supported reasoning setting.
     */
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
     * IMPORTANT:
     * These values are NOT forwarded as generic
     * reasoning_effort/reasoning_budget fields.
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

  /* ---------------- GLM 5.3 FLASH ---------------- */

  /*
   * GLM-5.3-Flash always reasons upstream.
   *
   * NVIDIA documents "low" as the minimum reasoning
   * setting, so this profile uses the lowest supported
   * effort rather than pretending reasoning can be disabled.
   */
  "z-ai/glm-5.3-flash-no": {
    baseModel:
      "z-ai/glm-5.3-flash",
    label: "No Thinking / Minimum",
    reasoningEffort: "low"
  },

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

  /*
   * Intentionally no "-no" Kimi profile.
   *
   * Kimi K3 remains reasoning-only in this proxy.
   */
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
   * This sends:
   *
   *   chat_template_kwargs:
   *     enable_thinking: false
   *
   * No thinking budget is sent.
   */
  "nvidia/nemotron-3-ultra-550b-a55b-no": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",
    label: "No Thinking",
    thinking: false
  },

  /*
   * Fast is also a no-thinking profile.
   *
   * It remains available as a separate compatibility
   * profile, while "-no" is the explicit naming for
   * disabling thinking.
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

  if (profile) {
    return {
      requestedModel: requested,
      model: profile.baseModel,
      profile,
      config:
        MODELS[profile.baseModel] ||
        null
    };
  }

  const direct =
    MODELS[requested];

  if (direct) {
    return {
      requestedModel: requested,
      model: requested,
      profile: null,
      config: direct
    };
  }

  if (ALLOW_UNKNOWN_MODELS) {
    return {
      requestedModel: requested,
      model: requested,
      profile: null,
      config: null
    };
  }

  return null;
}

/* ============================================================
   REASONING HELPERS
============================================================ */

function normalizeReasoningEffort(
  value,
  modelConfig
) {
  const levels =
    Array.isArray(
      modelConfig &&
      modelConfig.reasoningLevels
    )
      ? modelConfig.reasoningLevels
      : [
          "none",
          "low",
          "medium",
          "high",
          "max"
        ];

  const normalized =
    String(
      value || ""
    )
      .trim()
      .toLowerCase();

  if (
    levels.includes(
      normalized
    )
  ) {
    return normalized;
  }

  /*
   * Common aliases.
   */
  if (
    normalized ===
      "off" ||
    normalized ===
      "disabled" ||
    normalized ===
      "false"
  ) {
    return levels.includes(
      "none"
    )
      ? "none"
      : levels[0] || "none";
  }

  if (
    normalized ===
      "minimal" ||
    normalized ===
      "fast"
  ) {
    return levels.includes(
      "low"
    )
      ? "low"
      : levels[0] || "none";
  }

  if (
    normalized ===
      "balanced" ||
    normalized ===
      "medium"
  ) {
    if (
      levels.includes(
        "medium"
      )
    ) {
      return "medium";
    }

    if (
      levels.includes(
        "high"
      )
    ) {
      return "high";
    }

    return levels[0] || "none";
  }

  if (
    normalized ===
      "deep"
  ) {
    return levels.includes(
      "max"
    )
      ? "max"
      : levels.includes(
          "high"
        )
        ? "high"
        : levels[0] || "none";
  }

  if (
    normalized ===
      "true"
  ) {
    return (
      modelConfig &&
      modelConfig.defaultReasoningEffort
    ) || "high";
  }

  return (
    modelConfig &&
    modelConfig.defaultReasoningEffort
  ) || "high";
}

function budgetToReasoningEffort(
  budget,
  modelConfig
) {
  const numeric =
    Number(budget);

  if (
    !Number.isFinite(
      numeric
    )
  ) {
    return normalizeReasoningEffort(
      DEFAULT_REASONING_EFFORT,
      modelConfig
    );
  }

  if (
    numeric <= 0
  ) {
    return normalizeReasoningEffort(
      "none",
      modelConfig
    );
  }

  if (
    numeric <= 4096
  ) {
    return normalizeReasoningEffort(
      "low",
      modelConfig
    );
  }

  if (
    numeric <= 16384
  ) {
    return normalizeReasoningEffort(
      "high",
      modelConfig
    );
  }

  return normalizeReasoningEffort(
    "max",
    modelConfig
  );
}

function clampReasoningBudget(
  value
) {
  const numeric =
    Number(value);

  if (
    !Number.isFinite(
      numeric
    )
  ) {
    return null;
  }

  return clamp(
    Math.floor(numeric),
    0,
    32768
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
 * Generic reasoning_effort/reasoning_budget fields are NOT
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

  return messages.map(
    (message) => {
      if (
        !isPlainObject(
          message
        )
      ) {
        return message;
      }

      const normalized = {
        ...message
      };

      /*
       * JanitorAI and OpenAI-compatible clients sometimes send
       * undefined/null content for assistant/tool messages.
       * Preserve actual content but avoid malformed undefined
       * values.
       */
      if (
        normalized.content ===
          undefined &&
        normalized.role !==
          "assistant"
      ) {
        normalized.content = "";
      }

      return normalized;
    }
  );
}

/* ============================================================
   REQUEST VALUE HELPERS
============================================================ */

function clampTemperature(
  value
) {
  return clamp(
    numberOrDefault(
      value,
      DEFAULT_TEMPERATURE
    ),
    0,
    2
  );
}

function clampTopP(
  value
) {
  return clamp(
    numberOrDefault(
      value,
      DEFAULT_TOP_P
    ),
    0,
    1
  );
}

function getMaxTokens(
  incoming,
  modelConfig
) {
  const requested =
    Number(
      incoming.max_tokens
    );

  if (
    Number.isFinite(
      requested
    ) &&
    requested > 0
  ) {
    if (
      modelConfig &&
      Number.isFinite(
        modelConfig.maxOutputTokens
      )
    ) {
      return Math.min(
        Math.floor(
          requested
        ),
        modelConfig.maxOutputTokens
      );
    }

    return Math.floor(
      requested
    );
  }

  if (
    modelConfig &&
    Number.isFinite(
      modelConfig.maxOutputTokens
    )
  ) {
    return modelConfig.maxOutputTokens;
  }

  return DEFAULT_MAX_TOKENS;
}

/* ============================================================
   NIM REQUEST ADAPTER
============================================================ */

function buildNimRequest(
  incoming,
  resolution
) {
  const requestedModel =
    resolution.requestedModel;

  const normalizedModel =
    resolution.model;

  const modelConfig =
    resolution.config;

  const profile =
    resolution.profile;

  const messages =
    normalizeMessages(
      incoming.messages
    );

  const isKimi =
    normalizedModel ===
    "moonshotai/kimi-k3";

  const request = {
    model:
      normalizedModel,

    messages,

    max_tokens:
      getMaxTokens(
        incoming,
        modelConfig
      ),

    temperature:
      clampTemperature(
        incoming.temperature
      ),

    stream:
      Boolean(
        incoming.stream
      )
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
        normalizedModel,
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
        normalizedModel,
        modelConfig,
        profile
      );

    /*
     * GLM-5.3-Flash does not expose a true "none" mode.
     * Its minimum supported reasoning effort is low.
     */
    if (
      reasoningEffort ===
      "none"
    ) {
      reasoningEffort =
        "low";
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
        normalizedModel,
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
        normalizedModel,
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
    parsed.choices =
      parsed.choices.map(
        (choice) => {
          if (
            !choice ||
            typeof choice !==
              "object"
          ) {
            return choice;
          }

          const next = {
            ...choice
          };

          if (
            next.message &&
            typeof next.message ===
              "object"
          ) {
            next.message = {
              ...next.message
            };

            if (
              "reasoning_content" in
              next.message
            ) {
              delete next.message
                .reasoning_content;
            }

            if (
              "reasoning" in
              next.message
            ) {
              delete next.message
                .reasoning;
            }

            if (
              typeof next.message.content ===
              "string"
            ) {
              next.message.content =
                removeThinkTags(
                  next.message.content
                );
            }
          }

          if (
            next.delta &&
            typeof next.delta ===
              "object"
          ) {
            next.delta = {
              ...next.delta
            };

            if (
              "reasoning_content" in
              next.delta
            ) {
              delete next.delta
                .reasoning_content;
            }

            if (
              "reasoning" in
              next.delta
            ) {
              delete next.delta
                .reasoning;
            }

            if (
              typeof next.delta.content ===
              "string"
            ) {
              next.delta.content =
                removeThinkTags(
                  next.delta.content
                );
            }
          }

          return next;
        }
      );
  }

  return parsed;
}

function removeThinkTags(
  content
) {
  if (
    typeof content !==
    "string"
  ) {
    return content;
  }

  return content
    .replace(
      /<think>[\s\S]*?<\/think>/gi,
      ""
    )
    .replace(
      /<thinking>[\s\S]*?<\/thinking>/gi,
      ""
    )
    .trim();
}

/* ============================================================
   SSE HELPERS
============================================================ */

function writeSse(
  response,
  data
) {
  response.write(
    `data: ${data}\n\n`
  );
}

function writeJsonSse(
  response,
  object
) {
  writeSse(
    response,
    JSON.stringify(
      object
    )
  );
}

function endSse(
  response
) {
  writeSse(
    response,
    "[DONE]"
  );

  response.end();
}

function extractSseDataLines(
  buffer
) {
  const events = [];

  let remaining =
    buffer;

  while (true) {
    const newline =
      remaining.indexOf(
        "\n\n"
      );

    if (
      newline ===
      -1
    ) {
      break;
    }

    const event =
      remaining.slice(
        0,
        newline
      );

    remaining =
      remaining.slice(
        newline + 2
      );

    const lines =
      event.split(
        "\n"
      );

    const dataLines =
      lines
        .filter(
          (line) =>
            line.startsWith(
              "data:"
            )
        )
        .map(
          (line) =>
            line.slice(
              5
            ).trimStart()
        );

    if (
      dataLines.length
    ) {
      events.push(
        dataLines.join(
          "\n"
        )
      );
    }
  }

  return {
    events,
    remainder:
      remaining
  };
}

/* ============================================================
   ERROR HELPERS
============================================================ */

function safeJsonParse(
  value
) {
  try {
    return JSON.parse(
      value
    );
  } catch {
    return null;
  }
}

function extractUpstreamError(
  error
) {
  const response =
    error &&
    error.response;

  const data =
    response &&
    response.data;

  if (
    data &&
    typeof data ===
      "object"
  ) {
    return data;
  }

  if (
    typeof data ===
    "string"
  ) {
    const parsed =
      safeJsonParse(
        data
      );

    return (
      parsed || {
        error:
          data.slice(
            0,
            MAX_ERROR_BODY_SIZE
          )
      }
    );
  }

  return {
    error:
      error &&
      error.message
        ? error.message
        : "Unknown upstream error"
  };
}

function makeErrorResponse(
  error,
  resolution,
  elapsedMs
) {
  const upstream =
    extractUpstreamError(
      error
    );

  const status =
    Number(
      error &&
      error.response &&
      error.response.status
    ) || 502;

  const request =
    resolution
      ? resolution
      : null;

  return {
    error: {
      message:
        upstream.error?.message ||
        upstream.message ||
        error?.message ||
        "NVIDIA NIM request failed",

      type:
        upstream.error?.type ||
        "upstream_error",

      code:
        upstream.error?.code ||
        status,

      status,

      proxy: {
        requested_model:
          request?.requestedModel ||
          null,

        resolved_model:
          request?.model ||
          null,

        profile:
          request?.profile
            ? {
                label:
                  request.profile.label ||
                  null,
                thinking:
                  request.profile.thinking,
                reasoningEffort:
                  request.profile
                    .reasoningEffort ||
                  null
              }
            : null,

        adapter:
          request?.config?.adapter ||
          null,

        upstream_latency_ms:
          elapsedMs
      },

      upstream:
        DEBUG_PROXY
          ? upstream
          : undefined
    }
  };
}

/* ============================================================
   DEBUGGING
============================================================ */

function debugLog(
  label,
  value
) {
  if (!DEBUG_PROXY) {
    return;
  }

  console.log(
    `\n========== ${label} ==========\n` +
      JSON.stringify(
        value,
        null,
        2
      ) +
      `\n========== END ${label} ==========\n`
  );
}

/* ============================================================
   MODEL CACHE
============================================================ */

let modelCache = {
  fetchedAt: 0,
  data: null
};

async function fetchNimModels() {
  const now =
    Date.now();

  if (
    modelCache.data &&
    now -
      modelCache.fetchedAt <
      MODEL_CACHE_TTL
  ) {
    return modelCache.data;
  }

  try {
    const response =
      await axios.get(
        `${NIM_API_BASE}/models`,
        {
          headers:
            buildHeaders(),
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
      response.status >= 200 &&
      response.status < 300
    ) {
      modelCache = {
        fetchedAt:
          now,
        data:
          response.data
      };

      return response.data;
    }
  } catch {
    /* Fall through to local models. */
  }

  return null;
}

/* ============================================================
   HEADERS
============================================================ */

function buildHeaders() {
  const headers = {
    "Content-Type":
      "application/json",
    Accept:
      "application/json"
  };

  if (NIM_API_KEY) {
    headers.Authorization =
      `Bearer ${NIM_API_KEY}`;
  }

  return headers;
}

/* ============================================================
   LOCAL MODEL LIST
============================================================ */

function getLocalModels() {
  const created =
    Math.floor(
      Date.now() / 1000
    );

  const models = [];

  for (
    const [
      id,
      config
    ] of Object.entries(
      MODELS
    )
  ) {
    models.push({
      id,
      object:
        "model",
      created,
      owned_by:
        config.provider,
      name:
        config.name
    });
  }

  for (
    const [
      id,
      profile
    ] of Object.entries(
      PROFILES
    )
  ) {
    const base =
      MODELS[
        profile.baseModel
      ];

    models.push({
      id,
      object:
        "model",
      created,
      owned_by:
        base?.provider ||
        "proxy",
      name:
        `${base?.name || profile.baseModel} - ${profile.label}`
    });
  }

  return models;
}

/* ============================================================
   EXPRESS SETUP
============================================================ */

app.disable(
  "x-powered-by"
);

app.use(
  cors({
    origin: true,
    credentials: true
  })
);

app.use(
  express.json({
    limit:
      process.env.JSON_BODY_LIMIT ||
      "10mb"
  })
);

/*
 * Raw request logging when debugging.
 */
app.use(
  (req, res, next) => {
    if (
      DEBUG_PROXY &&
      req.path !==
        "/health"
    ) {
      console.log(
        `[${new Date().toISOString()}] ` +
          `${req.method} ${req.path}`
      );
    }

    next();
  }
);

/* ============================================================
   BASIC ROUTES
============================================================ */

app.get(
  "/",
  (req, res) => {
    res.json({
      name:
        "NVIDIA NIM OpenAI-Compatible Proxy",

      status:
        "ok",

      default_model:
        DEFAULT_MODEL,

      profiles:
        Object.keys(
          PROFILES
        ),

      models:
        Object.keys(
          MODELS
        )
    });
  }
);

app.get(
  "/health",
  (req, res) => {
    res.json({
      status:
        "ok",

      upstream:
        NIM_API_BASE,

      default_model:
        DEFAULT_MODEL,

      configured:
        Boolean(
          NIM_API_KEY
        )
    });
  }
);

app.get(
  "/v1/models",
  async (req, res) => {
    const remote =
      await fetchNimModels();

    /*
     * Return our local profile models first so clients such
     * as JanitorAI can discover the aliases.
     */
    const local =
      getLocalModels();

    if (
      remote &&
      Array.isArray(
        remote.data
      )
    ) {
      const existing =
        new Set(
          remote.data.map(
            (model) =>
              model.id
          )
        );

      const merged =
        [
          ...remote.data
        ];

      for (
        const model of
          local
      ) {
        if (
          !existing.has(
            model.id
          )
        ) {
          merged.push(
            model
          );
        }
      }

      return res.json({
        object:
          "list",
        data:
          merged
      });
    }

    return res.json({
      object:
        "list",
      data:
        local
    });
  }
);

app.post(
  "/v1/models/refresh",
  async (req, res) => {
    modelCache = {
      fetchedAt: 0,
      data: null
    };

    const remote =
      await fetchNimModels();

    res.json({
      ok:
        true,
      upstream:
        remote
          ? "reachable"
          : "unavailable",
      models:
        remote?.data ||
        getLocalModels()
    });
  }
);

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post(
  "/v1/chat/completions",
  async (req, res) => {
    const startedAt =
      Date.now();

    const incoming =
      isPlainObject(
        req.body
      )
        ? req.body
        : {};

    const requestedModel =
      normalizeModel(
        incoming.model ||
          DEFAULT_MODEL
      );

    const resolution =
      resolveModel(
        requestedModel
      );

    if (!resolution) {
      return res
        .status(400)
        .json({
          error: {
            message:
              `Unknown model: ${requestedModel}`,

            type:
              "invalid_request_error",

            code:
              "unknown_model"
          }
        });
    }

    const nimRequest =
      buildNimRequest(
        incoming,
        resolution
      );

    debugLog(
      "INCOMING REQUEST",
      incoming
    );

    debugLog(
      "RESOLUTION",
      resolution
    );

    debugLog(
      "FULL NIM REQUEST",
      nimRequest
    );

    const isStreaming =
      Boolean(
        nimRequest.stream
      );

    const headers =
      buildHeaders();

    if (
      isStreaming
    ) {
      headers.Accept =
        "text/event-stream";
    }

    try {
      const upstream =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers,

            timeout:
              NIM_TIMEOUT,

            httpAgent,
            httpsAgent,

            responseType:
              isStreaming
                ? "stream"
                : "json",

            /*
             * Do not retry automatically.
             *
             * This is deliberate because a 500 from NIM on a
             * reasoning profile should remain visible instead
             * of causing duplicate generations.
             */
            validateStatus:
              () => true,

            maxContentLength:
              Infinity,

            maxBodyLength:
              Infinity
          }
        );

      const elapsedMs =
        Date.now() -
        startedAt;

      /*
       * --------------------------------------------------------
       * UPSTREAM ERROR
       * --------------------------------------------------------
       */
      if (
        upstream.status <
          200 ||
        upstream.status >=
          300
      ) {
        let body =
          upstream.data;

        if (
          body &&
          typeof body.pipe ===
            "function"
        ) {
          body =
            await readStreamToString(
              body
            );
        }

        debugLog(
          "NIM ERROR RESPONSE",
          body
        );

        return res
          .status(
            upstream.status
          )
          .json(
            makeErrorResponse(
              {
                response: {
                  status:
                    upstream.status,
                  data:
                    body
                }
              },
              resolution,
              elapsedMs
            )
          );
      }

      /*
       * --------------------------------------------------------
       * STREAMING
       * --------------------------------------------------------
       */
      if (
        isStreaming
      ) {
        res.status(
          upstream.status
        );

        res.setHeader(
          "Content-Type",
          "text/event-stream"
        );

        res.setHeader(
          "Cache-Control",
          "no-cache, no-transform"
        );

        res.setHeader(
          "Connection",
          "keep-alive"
        );

        res.flushHeaders?.();

        let buffer =
          "";

        upstream.data.on(
          "data",
          (chunk) => {
            buffer +=
              chunk.toString(
                "utf8"
              );

            const parsed =
              extractSseDataLines(
                buffer
              );

            buffer =
              parsed.remainder;

            for (
              const data of
                parsed.events
            ) {
              if (
                data ===
                "[DONE]"
              ) {
                writeSse(
                  res,
                  "[DONE]"
                );

                continue;
              }

              const json =
                safeJsonParse(
                  data
                );

              if (!json) {
                writeSse(
                  res,
                  data
                );

                continue;
              }

              const cleaned =
                stripReasoning(
                  json
                );

              writeJsonSse(
                res,
                cleaned
              );
            }
          }
        );

        upstream.data.on(
          "end",
          () => {
            /*
             * Flush a final incomplete event if one exists.
             */
            if (
              buffer.trim()
            ) {
              const data =
                buffer
                  .split("\n")
                  .filter(
                    (line) =>
                      line.startsWith(
                        "data:"
                      )
                  )
                  .map(
                    (line) =>
                      line
                        .slice(5)
                        .trimStart()
                  )
                  .join("\n");

              if (
                data
              ) {
                const json =
                  safeJsonParse(
                    data
                  );

                if (
                  json
                ) {
                  writeJsonSse(
                    res,
                    stripReasoning(
                      json
                    )
                  );
                } else {
                  writeSse(
                    res,
                    data
                  );
                }
              }
            }

            if (
              !res.writableEnded
            ) {
              endSse(
                res
              );
            }
          }
        );

        upstream.data.on(
          "error",
          (streamError) => {
            console.error(
              "NIM stream error:",
              streamError
            );

            if (
              !res.writableEnded
            ) {
              res.end();
            }
          }
        );

        req.on(
          "close",
          () => {
            if (
              upstream.data &&
              typeof
                upstream.data.destroy ===
                "function"
            ) {
              upstream.data.destroy();
            }
          }
        );

        return;
      }

      /*
       * --------------------------------------------------------
       * NON-STREAMING
       * --------------------------------------------------------
       */
      const cleaned =
        stripReasoning(
          upstream.data
        );

      debugLog(
        "NIM RESPONSE",
        cleaned
      );

      return res
        .status(
          upstream.status
        )
        .json(
          cleaned
        );
    } catch (error) {
      const elapsedMs =
        Date.now() -
        startedAt;

      console.error(
        "Proxy request failed:",
        error?.message ||
          error
      );

      debugLog(
        "NIM EXCEPTION",
        {
          message:
            error?.message,
          code:
            error?.code,
          status:
            error?.response?.status,
          response:
            error?.response?.data
        }
      );

      if (
        res.headersSent
      ) {
        return res.end();
      }

      return res
        .status(
          Number(
            error?.response?.status
          ) || 502
        )
        .json(
          makeErrorResponse(
            error,
            resolution,
            elapsedMs
          )
        );
    }
  }
);

/* ============================================================
   STREAM READER
============================================================ */

function readStreamToString(
  stream
) {
  return new Promise(
    (resolve) => {
      const chunks =
        [];

      stream.on(
        "data",
        (chunk) => {
          chunks.push(
            Buffer.isBuffer(
              chunk
            )
              ? chunk
              : Buffer.from(
                  String(
                    chunk
                  )
                )
          );
        }
      );

      stream.on(
        "end",
        () => {
          resolve(
            Buffer.concat(
              chunks
            ).toString(
              "utf8"
            )
          );
        }
      );

      stream.on(
        "error",
        () => {
          resolve(
            Buffer.concat(
              chunks
            ).toString(
              "utf8"
            )
          );
        }
      );
    }
  );
}

/* ============================================================
   404
============================================================ */

app.use(
  (req, res) => {
    res.status(
      404
    ).json({
      error: {
        message:
          "Not found",
        type:
          "not_found"
      }
    });
  }
);

/* ============================================================
   GLOBAL ERROR HANDLER
============================================================ */

app.use(
  (
    error,
    req,
    res,
    next
  ) => {
    console.error(
      "Unhandled Express error:",
      error
    );

    if (
      res.headersSent
    ) {
      return next(
        error
      );
    }

    res.status(
      500
    ).json({
      error: {
        message:
          error?.message ||
          "Internal proxy error",

        type:
          "internal_error"
      }
    });
  }
);

/* ============================================================
   SERVER
============================================================ */

const server =
  app.listen(
    PORT,
    "0.0.0.0",
    () => {
      console.log(
        "============================================================"
      );

      console.log(
        " NVIDIA NIM OpenAI-Compatible Proxy"
      );

      console.log(
        "============================================================"
      );

      console.log(
        `Port:          ${PORT}`
      );

      console.log(
        `NIM endpoint:  ${NIM_API_BASE}`
      );

      console.log(
        `Default model: ${DEFAULT_MODEL}`
      );

      console.log(
        `Debug:         ${DEBUG_PROXY}`
      );

      console.log(
        `API key set:   ${Boolean(NIM_API_KEY)}`
      );

      console.log(
        ""
      );

      console.log(
        "Available profiles:"
      );

      for (
        const id of
          Object.keys(
            PROFILES
          )
      ) {
        console.log(
          `  - ${id}`
        );
      }

      console.log(
        ""
      );

      console.log(
        "Nemotron reasoning profiles:"
      );

      console.log(
        "  - nvidia/nemotron-3-ultra-550b-a55b-no"
      );

      console.log(
        "      enable_thinking=false"
      );

      console.log(
        "  - nvidia/nemotron-3-ultra-550b-a55b-fast"
      );

      console.log(
        "      enable_thinking=false"
      );

      console.log(
        "  - nvidia/nemotron-3-ultra-550b-a55b-balanced"
      );

      console.log(
        "      enable_thinking=true, medium_effort=true"
      );

      console.log(
        "  - nvidia/nemotron-3-ultra-550b-a55b-deep"
      );

      console.log(
        "      enable_thinking=true, thinking_token_budget=32768"
      );

      console.log(
        "============================================================"
      );
    }
  );

/* ============================================================
   GRACEFUL SHUTDOWN
============================================================ */

function shutdown(
  signal
) {
  console.log(
    `Received ${signal}; shutting down...`
  );

  server.close(
    () => {
      httpAgent.destroy();
      httpsAgent.destroy();

      process.exit(
        0
      );
    }
  );

  setTimeout(
    () => {
      process.exit(
        1
      );
    },
    10000
  ).unref();
}

process.on(
  "SIGINT",
  () =>
    shutdown(
      "SIGINT"
    )
);

process.on(
  "SIGTERM",
  () =>
    shutdown(
      "SIGTERM"
    )
);
