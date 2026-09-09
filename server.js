"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const http = require("http");
const https = require("https");

const app = express();

/* ============================================================
   SERVER CONFIGURATION
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
  "moonshotai/kimi-k3";

/* ============================================================
   MODEL CONFIGURATION

   IMPORTANT:

   - JanitorAI remains responsible for conversation history.
   - This proxy does NOT maintain conversation memory.
   - Every valid JanitorAI completion request produces exactly
     ONE NVIDIA /chat/completions request.
   - NO RETRIES are performed.

   Nemotron 3 Ultra:
     chat_template_kwargs.enable_thinking controls reasoning.

   Kimi K3:
     Reasoning remains enabled permanently.
     There is intentionally NO "none" option for Kimi.
============================================================ */

const MODELS = {
  "deepseek-ai/deepseek-v4-flash-0731": {
    name: "DeepSeek V4 Flash 0731",
    provider: "DeepSeek AI",

    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "max",

    maxOutputTokens: 8192
  },

  "deepseek-ai/deepseek-v4-pro-0813": {
    name: "DeepSeek V4 Pro 0813",
    provider: "DeepSeek AI",

    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "max",

    maxOutputTokens: 8192
  },

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",

    /*
     * Kimi reasoning is intentionally ALWAYS enabled.
     * Do not add "none".
     */
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "max",

    maxOutputTokens: 8192
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",

    /*
     * Nemotron uses enable_thinking rather than
     * reasoning_effort.
     */
    reasoningLevels: [
      "none",
      "high"
    ],

    defaultReasoningEffort: "high",

    defaultThinking: true,

    maxReasoningBudget: 32768,

    maxOutputTokens: 16384
  }
};

const ALLOW_UNKNOWN_MODELS =
  String(
    process.env.ALLOW_UNKNOWN_MODELS || "true"
  )
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   DEFAULT GENERATION SETTINGS
============================================================ */

const DEFAULT_REASONING_EFFORT =
  String(
    process.env.DEFAULT_REASONING_EFFORT || "high"
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

const MODEL_REASONING_BUDGETS = {
  "deepseek-ai/deepseek-v4-flash-0731":
    16384,

  "deepseek-ai/deepseek-v4-pro-0813":
    24576,

  "moonshotai/kimi-k3":
    32768,

  "nvidia/ron-3-ultra-550b-a55b":
    16384
};

const MODEL_REASONING_EFFORTS = {
  "deepseek-ai/deepseek-v4-flash-0731":
    "max",

  "deepseek-ai/deepseek-v4-pro-0813":
    "max",

  "moonshotai/kimi-k3":
    "max",

  "nvidia/ron-3-ultra-550b-a55b":
    "high"
};

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

/* ============================================================
   NEMOTRON THINKING DEFAULT

   Set:

     NEMOTRON_ENABLE_THINKING=true

   or:

     NEMOTRON_ENABLE_THINKING=false

   A request can explicitly override this with:

     chat_template_kwargs.enable_thinking

   or:

     reasoning_effort
     reasoning_mode
============================================================ */

const DEFAULT_NEMOTRON_THINKING =
  parseBoolean(
    process.env.NEMOTRON_ENABLE_THINKING,
    false
  );

/* ============================================================
   RESPONSE REASONING VISIBILITY

   true:
     Remove reasoning fields before sending responses to
     JanitorAI.

   false:
     Preserve them.

   Default is true.
============================================================ */

const STRIP_REASONING_FROM_RESPONSE =
  String(
    process.env.STRIP_REASONING_FROM_RESPONSE ||
      "true"
  )
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   NETWORK
============================================================ */

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

const MAX_ERROR_BODY_SIZE =
  2 * 1024 * 1024;

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
   DEBUG
============================================================ */

const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY || "true"
  )
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   MODEL CACHE

   This cache is ONLY for /models discovery.

   It is NEVER used to retry chat completions.
============================================================ */

let modelCache = null;
let modelCacheTimestamp = 0;
let modelCachePromise = null;

const MODEL_CACHE_TTL =
  Number.isFinite(
    Number(
      process.env.MODEL_CACHE_TTL_MS
    )
  )
    ? Number(
        process.env.MODEL_CACHE_TTL_MS
      )
    )
    : 300000;

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
  if (
    typeof value === "number"
  ) {
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

  if (
    typeof value === "boolean"
  ) {
    return value;
  }

  if (
    typeof value === "number"
  ) {
    return value !== 0;
  }

  if (
    typeof value === "string"
  ) {
    const normalized =
      value
        .trim()
        .toLowerCase();

    if (
      normalized === "true" ||
      normalized === "1" ||
      normalized === "yes" ||
      normalized === "on"
    ) {
      return true;
    }

    if (
      normalized === "false" ||
      normalized === "0" ||
      normalized === "no" ||
      normalized === "off"
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

function getModelConfig(model) {
  return (
    MODELS[
      normalizeModel(model)
    ] || null
  );
}

function isSupportedModel(model) {
  return (
    !!getModelConfig(model) ||
    ALLOW_UNKNOWN_MODELS
  );
}

function clampTemperature(value) {
  return Math.min(
    2,
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
  const number =
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
    modelConfig?.maxOutputTokens
  ) {
    return Math.min(
      modelConfig.maxOutputTokens,
      number
    );
  }

  return number;
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
    levels[0] || "none"
  );
}

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
      (
        normalized === "off" ||
        normalized === "false" ||
        normalized === "0"
      ) &&
      levels.includes("none")
    ) {
      return "none";
    }

    if (
      (
        normalized === "med" ||
        normalized === "medium"
      ) &&
      levels.includes("medium")
    ) {
      return "medium";
    }

    if (
      (
        normalized === "maximum" ||
        normalized ===
          "maximum_reasoning"
      ) &&
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

function getRequestedReasoningEffort(
  incoming,
  model,
  modelConfig
) {
  /*
   * Kimi is intentionally always reasoning.
   *
   * Even if a caller tries:
   *
   *   reasoning_effort: "none"
   *
   * we normalize it back to a valid Kimi level.
   */
  if (
    normalizeModel(model) ===
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

      if (
        requested === "none"
      ) {
        return "max";
      }

      return requested;
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

      if (
        requested === "none"
      ) {
        return "max";
      }

      return requested;
    }

    return "max";
  }

  if (
    incoming.reasoning_effort !==
    undefined
  ) {
    return modelConfig
      ? normalizeReasoningEffort(
          incoming.reasoning_effort,
          modelConfig
        )
      : String(
          incoming.reasoning_effort
        ).trim();
  }

  if (
    incoming.reasoning_mode !==
    undefined
  ) {
    return modelConfig
      ? normalizeReasoningEffort(
          incoming.reasoning_mode,
          modelConfig
        )
      : String(
          incoming.reasoning_mode
        ).trim();
  }

  if (
    incoming.reasoning_budget !==
    undefined
  ) {
    return modelConfig
      ? budgetToReasoningEffort(
          incoming.reasoning_budget,
          modelConfig
        )
      : null;
  }

  if (
    MODEL_REASONING_EFFORTS[
      model
    ]
  ) {
    return normalizeReasoningEffort(
      MODEL_REASONING_EFFORTS[
        model
      ],
      modelConfig
    );
  }

  return modelConfig
    ? normalizeReasoningEffort(
        DEFAULT_REASONING_EFFORT,
        modelConfig
      )
    : null;
}

/* ============================================================
   NEMOTRON THINKING
============================================================ */

function getNemotronThinking(
  incoming
) {
  /*
   * Highest priority:
   *
   * chat_template_kwargs.enable_thinking
   */

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    ) &&
    typeof incoming.chat_template_kwargs
      .enable_thinking === "boolean"
  ) {
    return (
      incoming.chat_template_kwargs
        .enable_thinking
    );
  }

  /*
   * reasoning_effort
   */

  if (
    incoming.reasoning_effort !==
    undefined
  ) {
    const value =
      incoming.reasoning_effort;

    if (
      typeof value === "boolean"
    ) {
      return value;
    }

    if (
      typeof value === "number"
    ) {
      return value !== 0;
    }

    const normalized =
      String(value)
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
      ].includes(normalized)
    ) {
      return false;
    }

    if (
      [
        "low",
        "medium",
        "high",
        "max",
        "maximum",
        "on",
        "true",
        "1",
        "enabled",
        "enable"
      ].includes(normalized)
    ) {
      return true;
    }
  }

  /*
   * reasoning_mode
   */

  if (
    incoming.reasoning_mode !==
    undefined
  ) {
    const value =
      incoming.reasoning_mode;

    if (
      typeof value === "boolean"
    ) {
      return value;
    }

    if (
      typeof value === "number"
    ) {
      return value !== 0;
    }

    const normalized =
      String(value)
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
      ].includes(normalized)
    ) {
      return false;
    }

    if (
      [
        "low",
        "medium",
        "high",
        "max",
        "maximum",
        "on",
        "true",
        "1",
        "enabled",
        "enable"
      ].includes(normalized)
    ) {
      return true;
    }
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

    result.push({
      ...message,

      role:
        message.role.trim()
    });
  }

  return result;
}

/* ============================================================
   BUILD NIM REQUEST
============================================================ */

function buildNimRequest(
  incoming,
  model,
  messages,
  stream
) {
  const normalizedModel =
    normalizeModel(model);

  const modelConfig =
    getModelConfig(model);

  const request = {
    model,

    messages,

    temperature:
      clampTemperature(
        incoming.temperature
      ),

    top_p:
      clampTopP(
        incoming.top_p
      ),

    max_tokens:
      clampMaxTokens(
        incoming.max_tokens,
        modelConfig
      ),

    stream
  };

  /* ==========================================================
     OPTIONAL GENERATION PARAMETERS
  ========================================================== */

  if (
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
    "n",
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
     NEMOTRON
  ========================================================== */

  if (
    normalizedModel ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    const enableThinking =
      getNemotronThinking(
        incoming
      );

    /*
     * Nemotron uses this field to control
     * reasoning.
     */
    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      enable_thinking:
        enableThinking
    };

    /*
     * reasoning_budget only exists when
     * reasoning is enabled.
     */
    if (
      enableThinking
    ) {
      const requestedBudget =
        incoming.reasoning_budget !==
        undefined
          ? incoming.reasoning_budget
          : MODEL_REASONING_BUDGETS[
              normalizedModel
            ];

      request.reasoning_budget =
        clampReasoningBudget(
          requestedBudget,
          modelConfig
        );
    } else {
      delete request.reasoning_budget;
    }

    /*
     * NVIDIA recommends this for tool/coding
     * use when thinking is enabled.
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
     * These are NOT sent to Nemotron.
     */
    delete request.reasoning_effort;
    delete request.reasoning_mode;
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
        modelConfig
      );

    request.reasoning_effort =
      reasoningEffort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      thinking:
        reasoningEffort !== "none",

      reasoning_effort:
        reasoningEffort
    };

    return request;
  }

  /* ==========================================================
     DEEPSEEK V4 PRO
  ========================================================== */

  if (
    normalizedModel ===
    "deepseek-ai/deepseek-v4-pro-0813"
  ) {
    const reasoningEffort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig
      );

    request.reasoning_effort =
      reasoningEffort;

    return request;
  }

  /* ==========================================================
     KIMI K3

     IMPORTANT:

     Kimi reasoning is ALWAYS enabled.

     The client cannot turn it off through:
       reasoning_effort: none
       reasoning_mode: none
       reasoning_effort: off
       reasoning_mode: off
       reasoning_effort: false
       reasoning_mode: false
       reasoning_effort: 0
       reasoning_mode: 0

     They are normalized to "max".
  ========================================================== */

  if (
    normalizedModel ===
    "moonshotai/kimi-k3"
  ) {
    const reasoningEffort =
      getRequestedReasoningEffort(
        incoming,
        model,
        modelConfig
      );

    request.reasoning_effort =
      reasoningEffort === "none"
        ? "max"
        : reasoningEffort;

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

    return request;
  }

  return request;
}

/* ============================================================
   REMOVE REASONING FROM CLIENT RESPONSE
============================================================ */

function stripReasoning(
  parsed
) {
  if (
    !STRIP_REASONING_FROM_RESPONSE ||
    !parsed ||
    typeof parsed !== "object" ||
    !Array.isArray(
      parsed.choices
    )
  ) {
    return parsed;
  }

  for (
    const choice of
      parsed.choices
  ) {
    if (
      !choice ||
      typeof choice !== "object"
    ) {
      continue;
    }

    if (choice.delta) {
      delete choice.delta
        .reasoning_content;

      delete choice.delta
        .reasoning;

      delete choice.delta
        .thinking;
    }

    if (choice.message) {
      delete choice.message
        .reasoning_content;

      delete choice.message
        .reasoning;

      delete choice.message
        .thinking;
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

      stripReasoning(
        parsed
      );

      output.push(
        "data: " +
          JSON.stringify(parsed)
      );
    } catch {
      /*
       * If an upstream SSE data line is not JSON,
       * pass it through unchanged.
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
        function (error) {
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
    typeof data === "object"
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
    details !== null &&
    details !== undefined
  ) {
    response.error.details =
      details;
  }

  return res
    .status(status)
    .json(response);
}

/* ============================================================
   NIM MODEL DISCOVERY

   IMPORTANT:

   This endpoint is separate from chat completion.

   It does NOT retry failed chat requests.
============================================================ */

async function fetchNimModels(
  forceRefresh = false
) {
  const now = Date.now();

  if (
    !forceRefresh &&
    Array.isArray(modelCache) &&
    now - modelCacheTimestamp <
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
          response.status >= 200 &&
          response.status < 300 &&
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
   HTTP REQUEST CONFIG
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
     * Axios does NOT retry requests by itself.
     *
     * No retry adapter.
     * No retry interceptor.
     * No retry loop.
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
   NIM REQUEST

   STRICT NO-RETRY POLICY

   Every call to requestNim() performs exactly ONE:

       POST /chat/completions

   There is intentionally:
     - no retry loop
     - no axios-retry
     - no interceptor
     - no recursive request
     - no second completion request
============================================================ */

async function requestNim(
  nimRequest,
  stream
) {
  try {
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

    /*
     * Successful upstream response.
     */
    if (
      response.status >= 200 &&
      response.status < 300
    ) {
      return response;
    }

    /*
     * Upstream returned an HTTP error.
     *
     * We capture it and return it.
     *
     * We DO NOT send another completion request.
     */

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
      try {
        errorBody =
          JSON.stringify(
            response.data
          );
      } catch {
        errorBody =
          String(
            response.data
          );
      }
    }

    console.error(
      "=================================================="
    );

    console.error(
      "NVIDIA UPSTREAM RESPONSE"
    );

    console.error(
      "HTTP STATUS:",
      response.status
    );

    console.error(
      "HEADERS:",
      response.headers
    );

    console.error(
      "BODY:",
      errorBody
    );

    console.error(
      "BODY LENGTH:",
      errorBody.length
    );

    console.error(
      "=================================================="
    );

    response.data =
      errorBody;

    return response;
  } catch (error) {
    /*
     * Network/timeout error.
     *
     * IMPORTANT:
     * This error is thrown back to the caller.
     *
     * There is NO retry.
     */

    console.error(
      "=================================================="
    );

    console.error(
      "NVIDIA CONNECTION ERROR"
    );

    console.error(
      "CODE:",
      error?.code
    );

    console.error(
      "MESSAGE:",
      error?.message
    );

    console.error(
      "STATUS:",
      error?.response?.status
    );

    console.error(
      "RESPONSE HEADERS:",
      error?.response?.headers
    );

    console.error(
      "RESPONSE DATA:",
      error?.response?.data
    );

    console.error(
      "=================================================="
    );

    throw error;
  }
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

      upstream_model_count:
        Array.isArray(modelCache)
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

      kimi_reasoning:
        "always_enabled",

      nim_api_base:
        NIM_API_BASE,

      timeout_ms:
        NIM_TIMEOUT,

      nim_retries:
        "disabled",

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
      ok: true,

      model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      upstream_model_count:
        Array.isArray(modelCache)
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

      nim_retries:
        false
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

      if (
        upstreamModels.length > 0
      ) {
        return res.json({
          object: "list",
          data:
            upstreamModels
        });
      }

      const models =
        Object.entries(
          MODELS
        ).map(
          (
            [
              id,
              config
            ]
          ) => ({
            id,

            object:
              "model",

            created:
              Math.floor(
                Date.now() /
                  1000
              ),

            owned_by:
              config.provider,

            reasoning_levels:
              config.reasoningLevels,

            max_output_tokens:
              config.maxOutputTokens
          })
        );

      return res.json({
        object: "list",
        data: models
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

   This is an explicit administrative endpoint.

   It is NOT called automatically after a failed
   chat completion.
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

      const model =
        requestedModel;

      const normalizedModel =
        normalizeModel(
          model
        );

      const modelConfig =
        getModelConfig(
          model
        );

      if (
        !isSupportedModel(
          model
        )
      ) {
        return sendError(
          res,
          400,
          "Unsupported model: " +
            model,

          {
            explicitly_configured_models:
              Object.keys(
                MODELS
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

      if (DEBUG_PROXY) {
        console.log(
          "=================================================="
        );

        console.log(
          "JANITORAI REQUEST"
        );

        console.log(
          "MODEL:",
          model
        );

        console.log(
          "JANITOR MESSAGES:",
          messages.length
        );

        console.log(
          "FIRST MESSAGE ROLE:",
          messages[0]?.role
        );

        console.log(
          "LAST MESSAGE ROLE:",
          messages[
            messages.length - 1
          ]?.role
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
          model,
          messages,
          stream
        );

      if (DEBUG_PROXY) {
        console.log(
          "NIM REQUEST SUMMARY:",
          {
            model:
              nimRequest.model,

            message_count:
              Array.isArray(
                nimRequest.messages
              )
                ? nimRequest
                    .messages
                    .length
                : 0,

            temperature:
              nimRequest.temperature,

            top_p:
              nimRequest.top_p,

            max_tokens:
              nimRequest.max_tokens,

            stream:
              nimRequest.stream,

            reasoning_effort:
              nimRequest.reasoning_effort,

            reasoning_budget:
              nimRequest.reasoning_budget,

            chat_template_kwargs:
              nimRequest
                .chat_template_kwargs
          }
        );

        /*
         * Useful for verifying exactly what gets sent
         * to NIM without logging the conversation itself.
         */
        console.log(
          "NIM REQUEST KEYS:",
          Object.keys(
            nimRequest
          )
        );

        console.log(
          "NIM COMPLETION REQUEST: 1"
        );

        console.log(
          "NIM COMPLETION RETRIES: 0"
        );

        console.log(
          "=================================================="
        );
      }

      /*
       * EXACTLY ONE completion request.
       *
       * There is deliberately no retry.
       */
      const response =
        await requestNim(
          nimRequest,
          stream
        );

      /* ======================================================
         UPSTREAM ERROR
      ====================================================== */

      if (
        response.status < 200 ||
        response.status >= 300
      ) {
        const errorBody =
          typeof response.data ===
          "string"
            ? response.data
            : (() => {
                try {
                  return JSON.stringify(
                    response.data
                  );
                } catch {
                  return String(
                    response.data
                  );
                }
              })();

        const upstreamMessage =
          extractErrorMessage(
            errorBody
          );

        console.error(
          "=================================================="
        );

        console.error(
          "NVIDIA NIM ERROR"
        );

        console.error(
          "Model:",
          normalizedModel
        );

        console.error(
          "HTTP:",
          response.status
        );

        console.error(
          "Message:",
          upstreamMessage
        );

        console.error(
          "Body:",
          errorBody
        );

        console.error(
          "NIM RETRY: NONE"
        );

        console.error(
          "=================================================="
        );

        /*
         * IMPORTANT:
         *
         * Do NOT refresh /models here.
         *
         * A failed completion must not cause another
         * NVIDIA request.
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
              errorBody
          }
        );
      }

      /* ======================================================
         NON-STREAMING
      ====================================================== */

      if (!stream) {
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

      let buffer = "";
      let ended = false;
      let clientDisconnected =
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
      }

      /*
       * If JanitorAI disconnects, stop the upstream
       * request. This is cancellation, NOT retrying.
       */

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
        function (
          chunk
        ) {
          if (
            ended ||
            clientDisconnected
          ) {
            return;
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

            const output =
              processSSEEvent(
                event
              );

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

          ended = true;

          /*
           * Process a final incomplete SSE event
           * if one exists.
           */
          if (
            buffer.trim() &&
            !clientDisconnected
          ) {
            const output =
              processSSEEvent(
                buffer
              );

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
        }
      );

      upstream.on(
        "error",
        function (
          error
        ) {
          if (ended) {
            return;
          }

          ended = true;

          console.error(
            "NVIDIA stream error:",
            error?.message ||
              error
          );

          /*
           * Never start another request.
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
    } catch (
      error
    ) {
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
        "NIM RETRY: NONE"
      );

      console.error(
        "=================================================="
      );

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
              NIM_TIMEOUT
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

          error.message
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

          error.message
        );
      }

      return sendError(
        res,
        500,
        "Proxy request failed.",

        error?.message ||
          String(error)
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
        NIM_TIMEOUT + "ms"
      );

      console.log(
        "NIM retries: DISABLED"
      );

      console.log(
        "NVIDIA completion retries: 0"
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
        "Kimi server memory: DISABLED"
      );

      console.log(
        "Janitor context passthrough: ENABLED"
      );

      console.log(
        "=================================================="
      );

      /*
       * This is model discovery only.
       *
       * It does not retry chat completions.
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
