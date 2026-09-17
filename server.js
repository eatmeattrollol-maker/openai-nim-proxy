"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");

/* ============================================================
   APP
============================================================ */

const app = express();

app.disable("x-powered-by");

/* ============================================================
   CONFIGURATION
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

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "z-ai/glm-5.3";

const DEBUG_PROXY =
  parseBoolean(
    process.env.DEBUG_PROXY,
    false
  );

const ALLOW_UNKNOWN_MODELS =
  parseBoolean(
    process.env.ALLOW_UNKNOWN_MODELS,
    true
  );

const STRIP_REASONING =
  parseBoolean(
    process.env.STRIP_REASONING_FROM_RESPONSE,
    true
  );

const NIM_TIMEOUT_MS =
  positiveNumber(
    process.env.NIM_TIMEOUT_MS,
    900000
  );

const MODEL_CACHE_TTL_MS =
  positiveNumber(
    process.env.MODEL_CACHE_TTL_MS,
    300000
  );

const DEFAULT_MAX_TOKENS =
  positiveNumber(
    process.env.DEFAULT_MAX_TOKENS,
    16384
  );

const DEFAULT_TEMPERATURE =
  numberOrDefault(
    process.env.DEFAULT_TEMPERATURE,
    1.0
  );

const DEFAULT_TOP_P =
  numberOrDefault(
    process.env.DEFAULT_TOP_P,
    0.95
  );

const DEFAULT_REPETITION_PENALTY =
  numberOrDefault(
    process.env.DEFAULT_REPETITION_PENALTY,
    1.0
  );

const DEFAULT_FREQUENCY_PENALTY =
  numberOrDefault(
    process.env.DEFAULT_FREQUENCY_PENALTY,
    0
  );

const DEFAULT_PRESENCE_PENALTY =
  numberOrDefault(
    process.env.DEFAULT_PRESENCE_PENALTY,
    0
  );

const DEFAULT_REASONING_EFFORT =
  String(
    process.env.DEFAULT_REASONING_EFFORT ||
      "high"
  )
    .trim()
    .toLowerCase();

const DEFAULT_REASONING_BUDGET =
  positiveNumber(
    process.env.DEFAULT_REASONING_BUDGET,
    16384
  );

const DEFAULT_NEMOTRON_THINKING =
  parseBoolean(
    process.env.NEMOTRON_ENABLE_THINKING,
    true
  );

/* ============================================================
   HTTP AGENTS
============================================================ */

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

  if (
    typeof value ===
    "boolean"
  ) {
    return value;
  }

  if (
    typeof value ===
    "number"
  ) {
    return value !== 0;
  }

  const normalized =
    String(value)
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

  return fallback;
}

function isPlainObject(
  value
) {
  return (
    value !== null &&
    typeof value ===
      "object" &&
    !Array.isArray(value)
  );
}

function isFiniteNumber(
  value
) {
  if (
    typeof value ===
    "number"
  ) {
    return Number.isFinite(
      value
    );
  }

  if (
    typeof value ===
      "string" &&
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
  return isFiniteNumber(
    value
  )
    ? Number(value)
    : fallback;
}

function positiveNumber(
  value,
  fallback
) {
  const number =
    Number(value);

  return Number.isFinite(
    number
  ) && number > 0
    ? number
    : fallback;
}

function clamp(
  value,
  minimum,
  maximum
) {
  return Math.min(
    maximum,
    Math.max(
      minimum,
      value
    )
  );
}

function normalizeModel(
  model
) {
  return String(
    model || ""
  )
    .trim()
    .toLowerCase();
}

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

/* ============================================================
   MODEL CONFIGURATION
============================================================ */

const MODELS = {
  "z-ai/glm-5.3": {
    name:
      "GLM 5.3",

    provider:
      "Z.ai",

    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort:
      "high",

    maxOutputTokens:
      32768,

    maxReasoningBudget:
      null,

    contextTokens:
      1048576,

    adapter:
      "glm53"
  },

  "z-ai/glm-5.3-flash": {
    name:
      "GLM 5.3 Flash",

    provider:
      "Z.ai",

    /*
     * Flash does not expose a true no-thinking mode.
     * Its lowest reasoning setting is "low".
     */
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort:
      "max",

    maxOutputTokens:
      32768,

    maxReasoningBudget:
      null,

    contextTokens:
      1048576,

    adapter:
      "glm53-flash"
  },

  "moonshotai/kimi-k3": {
    name:
      "Kimi K3",

    provider:
      "Moonshot AI",

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

    contextTokens:
      1048576,

    adapter:
      "kimi-k3"
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name:
      "NVIDIA Nemotron 3 Ultra 550B",

    provider:
      "NVIDIA",

    reasoningLevels: [
      "none",
      "medium",
      "high"
    ],

    defaultReasoningEffort:
      "high",

    defaultThinking:
      true,

    maxReasoningBudget:
      32768,

    maxOutputTokens:
      32768,

    contextTokens:
      1048576,

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

    contextTokens:
      1048576,

    adapter:
      "deepseek-flash"
  }
};

/* ============================================================
   PROFILES
============================================================ */

const PROFILES = {
  /* ----------------------------------------------------------
     GLM 5.3
  ---------------------------------------------------------- */

  "z-ai/glm-5.3-no": {
    baseModel:
      "z-ai/glm-5.3",

    label:
      "No Thinking",

    reasoningEffort:
      "none"
  },

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

  /* ----------------------------------------------------------
     GLM 5.3 FLASH
  ---------------------------------------------------------- */

  /*
   * GLM-5.3-Flash does not provide a genuine disabled
   * reasoning mode, so "-no" maps to its minimum reasoning
   * level instead of pretending thinking is disabled.
   */

  "z-ai/glm-5.3-flash-no": {
    baseModel:
      "z-ai/glm-5.3-flash",

    label:
      "No Thinking / Minimum",

    reasoningEffort:
      "low"
  },

  "z-ai/glm-5.3-flash-fast": {
    baseModel:
      "z-ai/glm-5.3-flash",

    label:
      "Fast",

    reasoningEffort:
      "low"
  },

  "z-ai/glm-5.3-flash-balanced": {
    baseModel:
      "z-ai/glm-5.3-flash",

    label:
      "Balanced",

    reasoningEffort:
      "high"
  },

  "z-ai/glm-5.3-flash-deep": {
    baseModel:
      "z-ai/glm-5.3-flash",

    label:
      "Deep",

    reasoningEffort:
      "max"
  },

  /* ----------------------------------------------------------
     KIMI K3
  ---------------------------------------------------------- */

  /*
   * Intentionally NO moonshotai/kimi-k3-no profile.
   */

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

  /* ----------------------------------------------------------
     NEMOTRON 3 ULTRA
  ---------------------------------------------------------- */

  "nvidia/nemotron-3-ultra-550b-a55b-no": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",

    label:
      "No Thinking",

    thinking:
      false
  },

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

    mediumEffort:
      true
  },

  "nvidia/nemotron-3-ultra-550b-a55b-deep": {
    baseModel:
      "nvidia/nemotron-3-ultra-550b-a55b",

    label:
      "Deep",

    thinking:
      true,

    thinkingTokenBudget:
      32768
  },

  /* ----------------------------------------------------------
     DEEPSEEK
  ---------------------------------------------------------- */

  "deepseek-ai/deepseek-v4-flash-0731-no": {
    baseModel:
      "deepseek-ai/deepseek-v4-flash-0731",

    label:
      "No Thinking",

    reasoningEffort:
      "none"
  },

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

/* ============================================================
   MODEL RESOLUTION
============================================================ */

function getProfile(
  model
) {
  return (
    PROFILES[
      normalizeModel(model)
    ] || null
  );
}

function resolveModel(
  requestedModel
) {
  const requested =
    normalizeModel(
      requestedModel
    );

  const profile =
    getProfile(
      requested
    );

  if (profile) {
    return {
      requestedModel:
        requested,

      model:
        profile.baseModel,

      profile,

      config:
        MODELS[
          profile.baseModel
        ] || null
    };
  }

  if (
    MODELS[requested]
  ) {
    return {
      requestedModel:
        requested,

      model:
        requested,

      profile:
        null,

      config:
        MODELS[requested]
    };
  }

  if (
    ALLOW_UNKNOWN_MODELS
  ) {
    return {
      requestedModel:
        requested,

      model:
        requested,

      profile:
        null,

      config:
        null
    };
  }

  return null;
}

/* ============================================================
   REASONING NORMALIZATION
============================================================ */

function normalizeReasoningEffort(
  value,
  config
) {
  const levels =
    Array.isArray(
      config?.reasoningLevels
    )
      ? config.reasoningLevels
      : [
          "none",
          "low",
          "medium",
          "high",
          "max"
        ];

  const effort =
    String(
      value || ""
    )
      .trim()
      .toLowerCase();

  if (
    levels.includes(
      effort
    )
  ) {
    return effort;
  }

  if (
    [
      "off",
      "disabled",
      "false",
      "none"
    ].includes(effort)
  ) {
    return levels.includes(
      "none"
    )
      ? "none"
      : levels[0];
  }

  if (
    [
      "fast",
      "minimal"
    ].includes(effort)
  ) {
    return levels.includes(
      "low"
    )
      ? "low"
      : levels[0];
  }

  if (
    [
      "balanced",
      "medium"
    ].includes(effort)
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

    return levels[0];
  }

  if (
    effort === "deep"
  ) {
    if (
      levels.includes(
        "max"
      )
    ) {
      return "max";
    }

    if (
      levels.includes(
        "high"
      )
    ) {
      return "high";
    }

    return levels[0];
  }

  return (
    config?.defaultReasoningEffort ||
    DEFAULT_REASONING_EFFORT
  );
}

function budgetToEffort(
  budget,
  config
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
      config
    );
  }

  if (
    numeric <= 0
  ) {
    return normalizeReasoningEffort(
      "none",
      config
    );
  }

  if (
    numeric <= 4096
  ) {
    return normalizeReasoningEffort(
      "low",
      config
    );
  }

  if (
    numeric <= 16384
  ) {
    return normalizeReasoningEffort(
      "high",
      config
    );
  }

  return normalizeReasoningEffort(
    "max",
    config
  );
}

function requestedEffort(
  body,
  resolution
) {
  const {
    model,
    profile,
    config
  } = resolution;

  if (
    profile?.reasoningEffort
  ) {
    return normalizeReasoningEffort(
      profile.reasoningEffort,
      config
    );
  }

  if (
    body.reasoning_effort !==
    undefined
  ) {
    return normalizeReasoningEffort(
      body.reasoning_effort,
      config
    );
  }

  if (
    body.reasoning_mode !==
    undefined
  ) {
    return normalizeReasoningEffort(
      body.reasoning_mode,
      config
    );
  }

  if (
    body.reasoning_budget !==
    undefined
  ) {
    return budgetToEffort(
      body.reasoning_budget,
      config
    );
  }

  return normalizeReasoningEffort(
    DEFAULT_REASONING_EFFORT,
    config
  );
}

/* ============================================================
   NEMOTRON HELPERS
============================================================ */

function resolveNemotronThinking(
  body,
  profile
) {
  if (
    profile?.thinking !==
    undefined
  ) {
    return Boolean(
      profile.thinking
    );
  }

  if (
    body.enable_thinking !==
    undefined
  ) {
    return parseBoolean(
      body.enable_thinking,
      DEFAULT_NEMOTRON_THINKING
    );
  }

  if (
    isPlainObject(
      body.chat_template_kwargs
    ) &&
    typeof body
      .chat_template_kwargs
      .enable_thinking ===
      "boolean"
  ) {
    return body
      .chat_template_kwargs
      .enable_thinking;
  }

  return DEFAULT_NEMOTRON_THINKING;
}

function resolveNemotronBudget(
  body,
  profile,
  thinking
) {
  if (!thinking) {
    return null;
  }

  if (
    profile?.thinkingTokenBudget !==
    undefined
  ) {
    return clamp(
      Math.floor(
        Number(
          profile.thinkingTokenBudget
        )
      ),
      1,
      32768
    );
  }

  if (
    body.thinking_token_budget !==
    undefined
  ) {
    const budget =
      Number(
        body.thinking_token_budget
      );

    if (
      Number.isFinite(
        budget
      )
    ) {
      return clamp(
        Math.floor(budget),
        1,
        32768
      );
    }
  }

  /*
   * Accept legacy client input, but translate it rather than
   * forwarding reasoning_budget to Nemotron.
   */
  if (
    body.reasoning_budget !==
    undefined
  ) {
    const budget =
      Number(
        body.reasoning_budget
      );

    if (
      Number.isFinite(
        budget
      )
    ) {
      return clamp(
        Math.floor(budget),
        1,
        32768
      );
    }
  }

  return null;
}

function resolveNemotronMediumEffort(
  body,
  profile,
  thinking
) {
  if (!thinking) {
    return false;
  }

  if (
    profile?.mediumEffort !==
    undefined
  ) {
    return Boolean(
      profile.mediumEffort
    );
  }

  if (
    isPlainObject(
      body.chat_template_kwargs
    ) &&
    typeof body
      .chat_template_kwargs
      .medium_effort ===
      "boolean"
  ) {
    return body
      .chat_template_kwargs
      .medium_effort;
  }

  return false;
}

/* ============================================================
   REQUEST CONSTRUCTION
============================================================ */

function normalizeMessages(
  messages
) {
  if (
    !Array.isArray(
      messages
    )
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

      return {
        ...message
      };
    }
  );
}

function getMaxTokens(
  body,
  config
) {
  const requested =
    Number(
      body.max_tokens
    );

  if (
    Number.isFinite(
      requested
    ) &&
    requested > 0
  ) {
    if (
      Number.isFinite(
        config?.maxOutputTokens
      )
    ) {
      return Math.min(
        Math.floor(
          requested
        ),
        config.maxOutputTokens
      );
    }

    return Math.floor(
      requested
    );
  }

  return (
    config?.maxOutputTokens ||
    DEFAULT_MAX_TOKENS
  );
}

function buildBaseRequest(
  body,
  resolution
) {
  const request = {
    model:
      resolution.model,

    messages:
      normalizeMessages(
        body.messages
      ),

    max_tokens:
      getMaxTokens(
        body,
        resolution.config
      ),

    temperature:
      clamp(
        numberOrDefault(
          body.temperature,
          DEFAULT_TEMPERATURE
        ),
        0,
        2
      ),

    stream:
      Boolean(
        body.stream
      )
  };

  /*
   * Only include top_p if the client actually provided it.
   *
   * This prevents us from unexpectedly changing upstream
   * sampling behavior.
   */
  if (
    body.top_p !==
    undefined
  ) {
    request.top_p =
      clamp(
        numberOrDefault(
          body.top_p,
          DEFAULT_TOP_P
        ),
        0,
        1
      );
  }

  if (
    body.repetition_penalty !==
    undefined
  ) {
    request.repetition_penalty =
      numberOrDefault(
        body.repetition_penalty,
        DEFAULT_REPETITION_PENALTY
      );
  }

  if (
    body.frequency_penalty !==
    undefined
  ) {
    request.frequency_penalty =
      numberOrDefault(
        body.frequency_penalty,
        DEFAULT_FREQUENCY_PENALTY
      );
  }

  if (
    body.presence_penalty !==
    undefined
  ) {
    request.presence_penalty =
      numberOrDefault(
        body.presence_penalty,
        DEFAULT_PRESENCE_PENALTY
      );
  }

  /*
   * Pass through supported OpenAI-compatible parameters.
   */
  const passthrough = [
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
    const key of
      passthrough
  ) {
    if (
      body[key] !==
      undefined
    ) {
      request[key] =
        body[key];
    }
  }

  /*
   * Preserve explicitly supplied chat template kwargs as a
   * starting point. Model adapters are allowed to sanitize them.
   */
  if (
    isPlainObject(
      body.chat_template_kwargs
    )
  ) {
    request.chat_template_kwargs = {
      ...body.chat_template_kwargs
    };
  }

  /*
   * extra_body is accepted for compatibility with OpenAI SDK
   * clients, but is merged before model-specific cleanup.
   */
  if (
    isPlainObject(
      body.extra_body
    )
  ) {
    Object.assign(
      request,
      body.extra_body
    );
  }

  return request;
}

/* ============================================================
   MODEL ADAPTER
============================================================ */

function buildNimRequest(
  body,
  resolution
) {
  const request =
    buildBaseRequest(
      body,
      resolution
    );

  const model =
    resolution.model;

  const profile =
    resolution.profile;

  /* ----------------------------------------------------------
     GLM 5.3
  ---------------------------------------------------------- */

  if (
    model ===
    "z-ai/glm-5.3"
  ) {
    const effort =
      requestedEffort(
        body,
        resolution
      );

    request.reasoning_effort =
      effort;

    /*
     * Keep only the GLM-specific template option.
     */
    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      clear_thinking:
        true
    };

    /*
     * Prevent incompatible controls from leaking through.
     */
    delete request.reasoning_budget;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  /* ----------------------------------------------------------
     GLM 5.3 FLASH
  ---------------------------------------------------------- */

  if (
    model ===
    "z-ai/glm-5.3-flash"
  ) {
    let effort =
      requestedEffort(
        body,
        resolution
      );

    /*
     * Flash cannot truly disable reasoning.
     */
    if (
      effort ===
      "none"
    ) {
      effort =
        "low";
    }

    request.reasoning_effort =
      effort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      clear_thinking:
        true,

      reasoning_effort:
        effort
    };

    delete request.reasoning_budget;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.thinking_token_budget;
    delete request.nvext;

    return request;
  }

  /* ----------------------------------------------------------
     NEMOTRON 3 ULTRA
  ---------------------------------------------------------- */

  if (
    model ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    const thinking =
      resolveNemotronThinking(
        body,
        profile
      );

    const mediumEffort =
      resolveNemotronMediumEffort(
        body,
        profile,
        thinking
      );

    const thinkingBudget =
      resolveNemotronBudget(
        body,
        profile,
        thinking
      );

    /*
     * Start with a clean object instead of inheriting arbitrary
     * reasoning fields from JanitorAI.
     */
    const existingTemplate =
      isPlainObject(
        body.chat_template_kwargs
      )
        ? {
            ...body.chat_template_kwargs
          }
        : {};

    /*
     * Remove fields that are not part of the Nemotron-specific
     * interface we want to expose.
     */
    delete existingTemplate.reasoning_effort;
    delete existingTemplate.reasoning_budget;
    delete existingTemplate.thinking_token_budget;
    delete existingTemplate.enable_thinking;
    delete existingTemplate.nvext;

    /*
     * NVIDIA Nemotron thinking control.
     */
    existingTemplate.enable_thinking =
      thinking;

    if (
      thinking &&
      mediumEffort
    ) {
      existingTemplate.medium_effort =
        true;
    } else {
      delete existingTemplate.medium_effort;
    }

    /*
     * force_nonempty_content is useful when reasoning is used
     * alongside tools.
     */
    if (
      thinking &&
      Array.isArray(
        request.tools
      ) &&
      request.tools.length >
        0
    ) {
      existingTemplate.force_nonempty_content =
        true;
    } else {
      delete existingTemplate.force_nonempty_content;
    }

    request.chat_template_kwargs =
      existingTemplate;

    /*
     * Explicit thinking budget belongs at the top level.
     */
    if (
      thinking &&
      thinkingBudget !==
        null
    ) {
      request.thinking_token_budget =
        thinkingBudget;
    } else {
      delete request.thinking_token_budget;
    }

    /*
     * Absolutely do not send generic reasoning controls to
     * Nemotron.
     */
    delete request.reasoning_effort;
    delete request.reasoning_budget;
    delete request.reasoning_mode;
    delete request.enable_thinking;
    delete request.nvext;

    return request;
  }

  /* ----------------------------------------------------------
     DEEPSEEK
  ---------------------------------------------------------- */

  if (
    model ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
    const effort =
      requestedEffort(
        body,
        resolution
      );

    request.reasoning_effort =
      effort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      thinking:
        effort !== "none",

      reasoning_effort:
        effort
    };

    return request;
  }

  /* ----------------------------------------------------------
     KIMI K3
  ---------------------------------------------------------- */

  if (
    model ===
    "moonshotai/kimi-k3"
  ) {
    let effort =
      requestedEffort(
        body,
        resolution
      );

    /*
     * Kimi has no -no profile in this proxy.
     * Never turn it into a fake non-thinking request.
     */
    if (
      effort ===
      "none"
    ) {
      effort =
        "max";
    }

    request.reasoning_effort =
      effort;

    /*
     * Do not inject an enable_thinking flag.
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

    return request;
  }

  /* ----------------------------------------------------------
     UNKNOWN MODELS
  ---------------------------------------------------------- */

  return request;
}

/* ============================================================
   SAFE DEBUG LOGGING
============================================================ */

/*
 * IMPORTANT:
 *
 * This deliberately NEVER logs:
 *
 *   - messages
 *   - prompts
 *   - conversation history
 *   - assistant content
 *   - tool arguments
 *   - request bodies
 *
 * Even DEBUG_PROXY=true only prints metadata.
 */

function debugLog(
  label,
  data
) {
  if (!DEBUG_PROXY) {
    return;
  }

  try {
    console.log(
      `[proxy-debug] ${label} ${JSON.stringify(
        data
      )}`
    );
  } catch {
    console.log(
      `[proxy-debug] ${label}`
    );
  }
}

function requestMetadata(
  body,
  resolution
) {
  return {
    requested_model:
      resolution.requestedModel,

    resolved_model:
      resolution.model,

    profile:
      resolution.profile?.label ||
      null,

    adapter:
      resolution.config?.adapter ||
      null,

    stream:
      Boolean(
        body.stream
      ),

    message_count:
      Array.isArray(
        body.messages
      )
        ? body.messages.length
        : 0,

    has_tools:
      Array.isArray(
        body.tools
      ) &&
      body.tools.length >
        0
  };
}

/* ============================================================
   RESPONSE CLEANING
============================================================ */

/*
 * Remove explicit reasoning fields from a normal JSON
 * OpenAI-compatible response.
 */
function cleanResponseObject(
  response
) {
  if (
    !isPlainObject(
      response
    )
  ) {
    return response;
  }

  const output = {
    ...response
  };

  if (
    Array.isArray(
      output.choices
    )
  ) {
    output.choices =
      output.choices.map(
        cleanChoice
      );
  }

  return output;
}

function cleanChoice(
  choice
) {
  if (
    !isPlainObject(
      choice
    )
  ) {
    return choice;
  }

  const output = {
    ...choice
  };

  if (
    isPlainObject(
      output.message
    )
  ) {
    output.message = {
      ...output.message
    };

    delete output.message
      .reasoning_content;

    delete output.message
      .reasoning;

    delete output.message
      .thinking;

    if (
      typeof output.message.content ===
      "string"
    ) {
      output.message.content =
        removeThinkBlocks(
          output.message.content
        );
    }
  }

  if (
    isPlainObject(
      output.delta
    )
  ) {
    output.delta = {
      ...output.delta
    };

    delete output.delta
      .reasoning_content;

    delete output.delta
      .reasoning;

    delete output.delta
      .thinking;

    if (
      typeof output.delta.content ===
      "string"
    ) {
      output.delta.content =
        removeThinkBlocks(
          output.delta.content
        );
    }
  }

  return output;
}

/*
 * Non-streaming cleanup.
 */
function removeThinkBlocks(
  text
) {
  if (
    typeof text !==
    "string"
  ) {
    return text;
  }

  return text
    .replace(
      /<think\b[^>]*>[\s\S]*?<\/think\s*>/gi,
      ""
    )
    .replace(
      /<thinking\b[^>]*>[\s\S]*?<\/thinking\s*>/gi,
      ""
    )
    .trim();
}

/* ============================================================
   STREAMING REASONING FILTER
============================================================ */

/*
 * Streaming is different from normal JSON because a <think>
 * block can be split across arbitrary SSE chunks.
 *
 * Example:
 *
 *   chunk 1: "<thi"
 *   chunk 2: "nk>secret"
 *   chunk 3: "</think>Hello"
 *
 * A simple regex on each chunk would fail.
 *
 * This stateful filter keeps track of whether we're currently
 * inside a thinking block.
 */

class ThinkingStreamFilter {
  constructor() {
    this.insideThink =
      false;

    this.pending =
      "";
  }

  process(
    input
  ) {
    if (
      typeof input !==
      "string" ||
      input.length === 0
    ) {
      return "";
    }

    let text =
      this.pending +
      input;

    this.pending =
      "";

    let output =
      "";

    while (
      text.length > 0
    ) {
      if (
        this.insideThink
      ) {
        const closeIndex =
          findClosingThinkTag(
            text
          );

        if (
          closeIndex ===
          -1
        ) {
          /*
           * Keep a small suffix in case the closing tag is split
           * across chunks.
           */
          const keep =
            partialTagSuffix(
              text,
              [
                "</think>",
                "</thinking>"
              ]
            );

          if (
            keep > 0
          ) {
            text =
              text.slice(
                text.length -
                  keep
              );
          } else {
            text =
              "";
          }

          this.pending =
            text;

          break;
        }

        text =
          text.slice(
            closeIndex
          );

        const match =
          text.match(
            /^<\/thinking\s*>/i
          ) ||
          text.match(
            /^<\/think\s*>/i
          );

        if (
          match
        ) {
          text =
            text.slice(
              match[0].length
            );
        }

        this.insideThink =
          false;

        continue;
      }

      const openMatch =
        findOpeningThinkTag(
          text
        );

      if (
        !openMatch
      ) {
        /*
         * The end of this text could be the beginning of a
         * split <think> tag.
         */
        const keep =
          partialTagSuffix(
            text,
            [
              "<think>",
              "<thinking>"
            ]
          );

        if (
          keep > 0
        ) {
          output +=
            text.slice(
              0,
              text.length -
                keep
            );

          this.pending =
            text.slice(
              text.length -
                keep
            );
        } else {
          output +=
            text;
        }

        break;
      }

      output +=
        text.slice(
          0,
          openMatch.index
        );

      text =
        text.slice(
          openMatch.index +
            openMatch.length
        );

      this.insideThink =
        true;
    }

    return output;
  }

  flush() {
    /*
     * If the stream ends while inside a thinking block, don't
     * leak the unfinished reasoning.
     */
    if (
      this.insideThink
    ) {
      this.pending =
        "";

      return "";
    }

    const output =
      this.pending;

    this.pending =
      "";

    return output;
  }
}

function findOpeningThinkTag(
  text
) {
  const think =
    text.search(
      /<think\s*>/i
    );

  const thinking =
    text.search(
      /<thinking\s*>/i
    );

  if (
    think === -1 &&
    thinking === -1
  ) {
    return null;
  }

  if (
    think === -1
  ) {
    const match =
      text.match(
        /<thinking\s*>/i
      );

    return {
      index:
        thinking,
      length:
        match[0].length
    };
  }

  if (
    thinking === -1
  ) {
    const match =
      text.match(
        /<think\s*>/i
      );

    return {
      index:
        think,
      length:
        match[0].length
    };
  }

  if (
    think <
    thinking
  ) {
    const match =
      text.match(
        /<think\s*>/i
      );

    return {
      index:
        think,
      length:
        match[0].length
    };
  }

  const match =
    text.match(
      /<thinking\s*>/i
    );

  return {
    index:
      thinking,
    length:
      match[0].length
  };
}

function findClosingThinkTag(
  text
) {
  const think =
    text.search(
      /<\/think\s*>/i
    );

  const thinking =
    text.search(
      /<\/thinking\s*>/i
    );

  if (
    think === -1
  ) {
    return thinking;
  }

  if (
    thinking === -1
  ) {
    return think;
  }

  return Math.min(
    think,
    thinking
  );
}

function partialTagSuffix(
  text,
  tags
) {
  const lower =
    text.toLowerCase();

  let best =
    0;

  for (
    const tag of
      tags
  ) {
    const normalized =
      tag.toLowerCase();

    const maximum =
      Math.min(
        normalized.length - 1,
        text.length
      );

    for (
      let size = 1;
      size <= maximum;
      size++
    ) {
      const suffix =
        lower.slice(
          lower.length -
            size
        );

      const prefix =
        normalized.slice(
          0,
          size
        );

      if (
        suffix ===
        prefix
      ) {
        best =
          Math.max(
            best,
            size
          );
      }
    }
  }

  return best;
}

/* ============================================================
   SSE PARSER
============================================================ */

class SSEParser {
  constructor(
    onEvent
  ) {
    this.buffer =
      "";

    this.onEvent =
      onEvent;
  }

  push(
    chunk
  ) {
    this.buffer +=
      chunk;

    /*
     * Normalize CRLF/CR to LF.
     */
    this.buffer =
      this.buffer.replace(
        /\r\n/g,
        "\n"
      );

    this.buffer =
      this.buffer.replace(
        /\r/g,
        "\n"
      );

    while (true) {
      const separator =
        this.buffer.indexOf(
          "\n\n"
        );

      if (
        separator ===
        -1
      ) {
        break;
      }

      const rawEvent =
        this.buffer.slice(
          0,
          separator
        );

      this.buffer =
        this.buffer.slice(
          separator + 2
        );

      this.parseEvent(
        rawEvent
      );
    }
  }

  end() {
    if (
      this.buffer.length > 0
    ) {
      this.parseEvent(
        this.buffer
      );
    }

    this.buffer =
      "";
  }

  parseEvent(
    rawEvent
  ) {
    if (
      !rawEvent
    ) {
      return;
    }

    const lines =
      rawEvent.split(
        "\n"
      );

    const dataLines =
      [];

    let eventName =
      null;

    let eventId =
      null;

    for (
      const line of
        lines
    ) {
      if (
        line.startsWith(
          "data:"
        )
      ) {
        dataLines.push(
          line.slice(
            5
          ).replace(
            /^ /,
            ""
          )
        );

        continue;
      }

      if (
        line.startsWith(
          "event:"
        )
      ) {
        eventName =
          line.slice(
            6
          ).trim();
        continue;
      }

      if (
        line.startsWith(
          "id:"
        )
      ) {
        eventId =
          line.slice(
            3
          ).trim();
      }
    }

    if (
      dataLines.length ===
      0
    ) {
      return;
    }

    this.onEvent({
      data:
        dataLines.join(
          "\n"
        ),

      event:
        eventName,

      id:
        eventId
    });
  }
}

/* ============================================================
   SSE EVENT CLEANING
============================================================ */

function cleanSSEJson(
  json,
  thinkingFilter
) {
  if (
    !isPlainObject(
      json
    )
  ) {
    return json;
  }

  const output = {
    ...json
  };

  if (
    Array.isArray(
      output.choices
    )
  ) {
    output.choices =
      output.choices.map(
        (choice) => {
          if (
            !isPlainObject(
              choice
            )
          ) {
            return choice;
          }

          const cleaned =
            {
              ...choice
            };

          if (
            isPlainObject(
              cleaned.delta
            )
          ) {
            cleaned.delta = {
              ...cleaned.delta
            };

            /*
             * These fields are what can cause reasoning to
             * appear as visible content in clients.
             */
            delete cleaned.delta
              .reasoning_content;

            delete cleaned.delta
              .reasoning;

            delete cleaned.delta
              .thinking;

            if (
              typeof cleaned.delta.content ===
              "string"
            ) {
              cleaned.delta.content =
                thinkingFilter.process(
                  cleaned.delta.content
                );
            }
          }

          if (
            isPlainObject(
              cleaned.message
            )
          ) {
            cleaned.message = {
              ...cleaned.message
            };

            delete cleaned.message
              .reasoning_content;

            delete cleaned.message
              .reasoning;

            delete cleaned.message
              .thinking;

            if (
              typeof cleaned.message.content ===
              "string"
            ) {
              cleaned.message.content =
                removeThinkBlocks(
                  cleaned.message.content
                );
            }
          }

          return cleaned;
        }
      );
  }

  /*
   * Some upstream responses can expose reasoning at the top
   * level. Remove it as well.
   */
  delete output.reasoning;
  delete output.reasoning_content;
  delete output.thinking;

  return output;
}

/* ============================================================
   SSE OUTPUT
============================================================ */

function writeSSEEvent(
  res,
  event
) {
  /*
   * We intentionally output standard OpenAI-compatible SSE:
   *
   * data: {...}
   *
   * No extra prefixes, JSON wrappers, markdown, or logging.
   */
  if (
    event.event
  ) {
    res.write(
      `event: ${event.event}\n`
    );
  }

  if (
    event.id
  ) {
    res.write(
      `id: ${event.id}\n`
    );
  }

  const lines =
    String(
      event.data
    ).split(
      "\n"
    );

  for (
    const line of
      lines
  ) {
    res.write(
      `data: ${line}\n`
    );
  }

  res.write(
    "\n"
  );
}

/* ============================================================
   UPSTREAM HEADERS
============================================================ */

function buildHeaders(
  streaming
) {
  const headers = {
    "Content-Type":
      "application/json",

    Accept:
      streaming
        ? "text/event-stream"
        : "application/json"
  };

  if (
    NIM_API_KEY
  ) {
    headers.Authorization =
      `Bearer ${NIM_API_KEY}`;
  }

  return headers;
}

/* ============================================================
   UPSTREAM ERROR EXTRACTION
============================================================ */

function extractErrorBody(
  data
) {
  if (
    isPlainObject(
      data
    )
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

    if (
      parsed
    ) {
      return parsed;
    }

    return {
      message:
        data.slice(
          0,
          2000
        )
    };
  }

  return {
    message:
      "Unknown upstream error"
  };
}

function makeProxyError(
  error,
  resolution,
  elapsedMs
) {
  const status =
    Number(
      error?.response?.status
    ) || 502;

  const upstream =
    extractErrorBody(
      error?.response?.data
    );

  const upstreamMessage =
    upstream?.error?.message ||
    upstream?.message ||
    error?.message ||
    "NVIDIA NIM request failed";

  return {
    error: {
      message:
        upstreamMessage,

      type:
        upstream?.error?.type ||
        "upstream_error",

      code:
        upstream?.error?.code ||
        status,

      status,

      proxy: {
        requested_model:
          resolution?.requestedModel ||
          null,

        resolved_model:
          resolution?.model ||
          null,

        profile:
          resolution?.profile?.label ||
          null,

        adapter:
          resolution?.config?.adapter ||
          null,

        upstream_latency_ms:
          elapsedMs
      }
    }
  };
}

/* ============================================================
   STREAM ERROR BODY
============================================================ */

async function streamToBuffer(
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
                  String(chunk)
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
   MODEL CACHE
============================================================ */

let modelCache = {
  timestamp:
    0,

  data:
    null
};

async function fetchRemoteModels() {
  const now =
    Date.now();

  if (
    modelCache.data &&
    now -
      modelCache.timestamp <
      MODEL_CACHE_TTL_MS
  ) {
    return modelCache.data;
  }

  try {
    const response =
      await axios.get(
        `${NIM_API_BASE}/models`,
        {
          headers:
            buildHeaders(
              false
            ),

          timeout:
            30000,

          httpAgent,
          httpsAgent,

          validateStatus:
            () => true
        }
      );

    if (
      response.status >= 200 &&
      response.status < 300 &&
      isPlainObject(
        response.data
      )
    ) {
      modelCache = {
        timestamp:
          now,

        data:
          response.data
      };

      return response.data;
    }
  } catch {
    /*
     * Local models are used below.
     */
  }

  return null;
}

function localModelList() {
  const created =
    Math.floor(
      Date.now() / 1000
    );

  const result =
    [];

  for (
    const [
      id,
      config
    ] of Object.entries(
      MODELS
    )
  ) {
    result.push({
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
    const config =
      MODELS[
        profile.baseModel
      ];

    result.push({
      id,

      object:
        "model",

      created,

      owned_by:
        config?.provider ||
        "proxy",

      name:
        `${config?.name || profile.baseModel} - ${profile.label}`
    });
  }

  return result;
}

/* ============================================================
   CORS / BODY PARSER
============================================================ */

app.use(
  cors({
    origin:
      true,

    credentials:
      true
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
 * IMPORTANT:
 *
 * There is deliberately NO middleware here that logs
 * req.body.
 *
 * This is what prevents full Janitor conversations from being
 * dumped into Render logs.
 */

/* ============================================================
   ROOT
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

      upstream:
        NIM_API_BASE,

      profile_count:
        Object.keys(
          PROFILES
        ).length
    });
  }
);

/* ============================================================
   HEALTH
============================================================ */

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

      api_key_configured:
        Boolean(
          NIM_API_KEY
        )
    });
  }
);

/* ============================================================
   MODELS
============================================================ */

app.get(
  "/v1/models",
  async (req, res) => {
    const remote =
      await fetchRemoteModels();

    const local =
      localModelList();

    if (
      isPlainObject(
        remote
      ) &&
      Array.isArray(
        remote.data
      )
    ) {
      const ids =
        new Set(
          remote.data
            .filter(
              isPlainObject
            )
            .map(
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
          !ids.has(
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
      timestamp:
        0,

      data:
        null
    };

    const remote =
      await fetchRemoteModels();

    res.json({
      object:
        "list",

      data:
        remote?.data ||
        localModelList()
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

    const body =
      isPlainObject(
        req.body
      )
        ? req.body
        : {};

    const requestedModel =
      normalizeModel(
        body.model ||
          DEFAULT_MODEL
      );

    const resolution =
      resolveModel(
        requestedModel
      );

    if (
      !resolution
    ) {
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
        body,
        resolution
      );

    const streaming =
      Boolean(
        nimRequest.stream
      );

    /*
     * Safe debug logging ONLY.
     *
     * No conversation content.
     */
    debugLog(
      "request",
      requestMetadata(
        body,
        resolution
      )
    );

    debugLog(
      "upstream_request_metadata",
      {
        model:
          nimRequest.model,

        stream:
          Boolean(
            nimRequest.stream
          ),

        max_tokens:
          nimRequest.max_tokens,

        temperature:
          nimRequest.temperature,

        has_tools:
          Array.isArray(
            nimRequest.tools
          ) &&
          nimRequest.tools.length >
            0,

        chat_template_kwargs:
          nimRequest.chat_template_kwargs ||
          null,

        thinking_token_budget:
          nimRequest.thinking_token_budget ||
          null,

        reasoning_effort:
          nimRequest.reasoning_effort ||
          null
      }
    );

    try {
      /*
       * Exactly ONE upstream request.
       * No automatic retries.
       */
      const upstream =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers:
              buildHeaders(
                streaming
              ),

            timeout:
              NIM_TIMEOUT_MS,

            httpAgent,
            httpsAgent,

            responseType:
              streaming
                ? "stream"
                : "json",

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

      /* --------------------------------------------------------
         UPSTREAM ERROR
      -------------------------------------------------------- */

      if (
        upstream.status < 200 ||
        upstream.status >= 300
      ) {
        let errorData =
          upstream.data;

        if (
          errorData &&
          typeof errorData.pipe ===
            "function"
        ) {
          errorData =
            await streamToBuffer(
              errorData
            );
        }

        const errorObject =
          makeProxyError(
            {
              response: {
                status:
                  upstream.status,

                data:
                  errorData
              }
            },
            resolution,
            elapsedMs
          );

        /*
         * Only log status/model metadata, never the upstream
         * conversation/error body.
         */
        console.error(
          `[proxy] NIM ${upstream.status} ` +
            `${resolution.model} ` +
            `profile=${resolution.profile?.label || "default"}`
        );

        if (
          DEBUG_PROXY
        ) {
          debugLog(
            "upstream_error",
            {
              status:
                upstream.status,

              model:
                resolution.model,

              profile:
                resolution.profile?.label ||
                null,

              adapter:
                resolution.config?.adapter ||
                null,

              upstream_message:
                extractErrorBody(
                  errorData
                )?.error?.message ||
                extractErrorBody(
                  errorData
                )?.message ||
                null
            }
          );
        }

        return res
          .status(
            upstream.status
          )
          .json(
            errorObject
          );
      }

      /* --------------------------------------------------------
         NON-STREAMING
      -------------------------------------------------------- */

      if (
        !streaming
      ) {
        const cleaned =
          STRIP_REASONING
            ? cleanResponseObject(
                upstream.data
              )
            : upstream.data;

        /*
         * Send the JSON object directly.
         * Do NOT stringify it ourselves.
         * Express handles the correct JSON response.
         */
        return res
          .status(
            upstream.status
          )
          .json(
            cleaned
          );
      }

      /* --------------------------------------------------------
         STREAMING
      -------------------------------------------------------- */

      res.status(
        upstream.status
      );

      /*
       * These are the standard headers expected by OpenAI-style
       * streaming clients.
       */
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

      /*
       * Don't let intermediary compression/buffering interfere
       * with token streaming.
       */
      res.setHeader(
        "X-Accel-Buffering",
        "no"
      );

      res.flushHeaders?.();

      /*
       * If reasoning stripping is disabled, the cleanest thing
       * to do is relay the upstream SSE stream directly.
       *
       * This avoids altering the provider's formatting at all.
       */
      if (
        !STRIP_REASONING
      ) {
        upstream.data.on(
          "data",
          (chunk) => {
            if (
              !res.writableEnded
            ) {
              res.write(
                chunk
              );
            }
          }
        );

        upstream.data.on(
          "end",
          () => {
            if (
              !res.writableEnded
            ) {
              res.end();
            }
          }
        );

        upstream.data.on(
          "error",
          () => {
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
              !res.writableEnded &&
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
       * Stateful reasoning filter.
       */
      const thinkingFilter =
        new ThinkingStreamFilter();

      /*
       * Proper SSE parser.
       *
       * We do NOT treat arbitrary chunks as complete events.
       */
      const parser =
        new SSEParser(
          (event) => {
            if (
              res.writableEnded
            ) {
              return;
            }

            /*
             * OpenAI/NIM uses:
             *
             * data: [DONE]
             *
             * at the end.
             */
            if (
              event.data ===
              "[DONE]"
            ) {
              /*
               * Flush anything that is still pending.
               */
              thinkingFilter.flush();

              writeSSEEvent(
                res,
                {
                  event:
                    event.event,

                  id:
                    event.id,

                  data:
                    "[DONE]"
                }
              );

              return;
            }

            const parsed =
              safeJsonParse(
                event.data
              );

            /*
             * If this isn't JSON, preserve it rather than
             * inventing a new format.
             */
            if (
              !parsed
            ) {
              writeSSEEvent(
                res,
                event
              );

              return;
            }

            const cleaned =
              cleanSSEJson(
                parsed,
                thinkingFilter
              );

            writeSSEEvent(
              res,
              {
                event:
                  event.event,

                id:
                  event.id,

                data:
                  JSON.stringify(
                    cleaned
                  )
              }
            );
          }
        );

      upstream.data.on(
        "data",
        (chunk) => {
          if (
            !res.writableEnded
          ) {
            parser.push(
              chunk.toString(
                "utf8"
              )
            );
          }
        }
      );

      upstream.data.on(
        "end",
        () => {
          if (
            res.writableEnded
          ) {
            return;
          }

          parser.end();

          /*
           * If NIM ended without sending [DONE], don't fabricate
           * another JSON event. Just close the SSE stream cleanly.
           */
          thinkingFilter.flush();

          res.end();
        }
      );

      upstream.data.on(
        "error",
        (streamError) => {
          console.error(
            `[proxy] upstream stream error for ${resolution.model}: ` +
              `${streamError?.message || "unknown error"}`
          );

          if (
            !res.writableEnded
          ) {
            res.end();
          }
        }
      );

      /*
       * If Janitor disconnects, stop reading from NVIDIA.
       */
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
    } catch (error) {
      const elapsedMs =
        Date.now() -
        startedAt;

      console.error(
        `[proxy] request failed for ` +
          `${resolution.model}: ` +
          `${error?.message || "unknown error"}`
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
          makeProxyError(
            error,
            resolution,
            elapsedMs
          )
        );
    }
  }
);

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
      `[proxy] unhandled error: ` +
        `${error?.message || "unknown error"}`
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
   START SERVER
============================================================ */

const server =
  app.listen(
    PORT,
    "0.0.0.0",
    () => {
      console.log(
        `[proxy] listening on port ${PORT}`
      );

      console.log(
        `[proxy] upstream=${NIM_API_BASE}`
      );

      console.log(
        `[proxy] default_model=${DEFAULT_MODEL}`
      );

      console.log(
        `[proxy] api_key_configured=${Boolean(
          NIM_API_KEY
        )}`
      );

      console.log(
        `[proxy] debug=${DEBUG_PROXY}`
      );

      console.log(
        `[proxy] reasoning_stripping=${STRIP_REASONING}`
      );

      console.log(
        `[proxy] profiles=${Object.keys(
          PROFILES
        ).length}`
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
    `[proxy] received ${signal}; shutting down`
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
