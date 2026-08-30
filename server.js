"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

/* ============================================================
   CONFIGURATION
   ============================================================ */

const PORT =
  Number(process.env.PORT) || 3000;

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
  "deepseek-ai/deepseek-v4-flash-0731";

/*
 * IMPORTANT:
 *
 * Unknown models are now ALLOWED.
 *
 * MODELS below is therefore a SPECIAL-CASE configuration
 * table, NOT an allowlist.
 *
 * If NVIDIA adds another model tomorrow, you do NOT have
 * to add it here just to make the proxy accept it.
 */

const MODELS = {
  /*
   * ==========================================================
   * DEEPSEEK V4 FLASH 0731
   * ==========================================================
   */

  "deepseek-ai/deepseek-v4-flash-0731": {
    name: "DeepSeek V4 Flash 0731",
    provider: "DeepSeek AI",

    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "high",

    maxOutputTokens: 16384,

    thinkingParameter: "chat_template"
  },

  /*
   * ==========================================================
   * DEEPSEEK V4 PRO 0813
   * ==========================================================
   */

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

    maxOutputTokens: 16384,

    thinkingParameter: "chat_template"
  },

  /*
   * ==========================================================
   * KIMI K3
   * ==========================================================
   */

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",

    reasoningLevels: [
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "high",

    maxOutputTokens: 65536,

    thinkingParameter: "native"
  },

  /*
   * ==========================================================
   * NEMOTRON 3 ULTRA
   * ==========================================================
   */

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",

    reasoningLevels: [
      "none",
      "medium",
      "high"
    ],

    defaultReasoningEffort: "high",

    maxOutputTokens: 32768,

    thinkingParameter: "nemotron"
  }
};

/*
 * ============================================================
 * ALLOW UNKNOWN NIM MODELS
 * ============================================================
 *
 * TRUE means:
 *
 *   Any model ID accepted by NVIDIA NIM can be requested.
 *
 * The MODELS object above is only used when a model requires
 * special request construction.
 *
 * This is what makes the proxy future-proof.
 */

const ALLOW_UNKNOWN_MODELS =
  String(
    process.env.ALLOW_UNKNOWN_MODELS ||
      "true"
  )
    .trim()
    .toLowerCase() === "true";

/*
 * ============================================================
 * DEFAULT REASONING SETTINGS
 * ============================================================
 */

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

/*
 * ============================================================
 * PER-MODEL REASONING BUDGETS
 * ============================================================
 *
 * KEEP THESE HERE.
 *
 * These remain easy to edit.
 *
 * For models that expose reasoning_effort rather than an
 * arbitrary numeric budget, the budget acts as a LOCAL
 * policy that gets mapped to the closest supported effort.
 */

const MODEL_REASONING_BUDGETS = {
  "deepseek-ai/deepseek-v4-flash-0731": 16384,

  "deepseek-ai/deepseek-v4-pro-0813": 24576,

  "moonshotai/kimi-k3": 32768,

  "nvidia/nemotron-3-ultra-550b-a55b": 16384
};

/*
 * ============================================================
 * PER-MODEL REASONING EFFORT
 * ============================================================
 *
 * KEEP THESE HERE.
 *
 * Change these when you want to manually control a known
 * model's reasoning level.
 */

const MODEL_REASONING_EFFORTS = {
  "deepseek-ai/deepseek-v4-flash-0731":
    "high",

  "deepseek-ai/deepseek-v4-pro-0813":
    "max",

  "moonshotai/kimi-k3":
    "max",

  "nvidia/nemotron-3-ultra-550b-a55b":
    "high"
};

/*
 * ============================================================
 * GENERATION DEFAULTS
 * ============================================================
 */

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
 * ============================================================
 * TIMEOUTS
 * ============================================================
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
 * Cache the upstream model list.
 *
 * This avoids calling NVIDIA every time /v1/models is opened.
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

/*
 * Maximum amount of upstream error data retained in memory.
 */

const MAX_ERROR_BODY_SIZE =
  2 * 1024 * 1024;

/*
 * ============================================================
 * DEBUG
 * ============================================================
 */

const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY ||
      "false"
  )
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   MODEL DISCOVERY CACHE
   ============================================================ */

let modelCache = null;
let modelCacheTimestamp = 0;
let modelCachePromise = null;

/*
 * Fetch NVIDIA's actual model list.
 *
 * This is intentionally separate from MODELS.
 *
 * MODELS = special behavior.
 *
 * NVIDIA /v1/models = actual available models.
 */

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
          return Array.isArray(modelCache)
            ? modelCache
            : [];
        }

        const response =
          await axios.get(
            `${NIM_API_BASE}/models`,
            {
              headers: {
                Authorization:
                  `Bearer ${NIM_API_KEY}`,

                Accept:
                  "application/json"
              },

              timeout:
                Math.min(
                  NIM_TIMEOUT,
                  30000
                ),

              validateStatus:
                () => true
            }
          );

        if (
          response.status >= 200 &&
          response.status < 300 &&
          response.data &&
          Array.isArray(
            response.data.data
          )
        ) {
          modelCache =
            response.data.data;

          modelCacheTimestamp =
            Date.now();

          return modelCache;
        }

        console.warn(
          `NVIDIA /v1/models returned HTTP ${response.status}`
        );

        return Array.isArray(modelCache)
          ? modelCache
          : [];
      } catch (error) {
        console.warn(
          "Unable to refresh NVIDIA model list:",
          error?.message ||
            error
        );

        return Array.isArray(modelCache)
          ? modelCache
          : [];
      } finally {
        modelCachePromise =
          null;
      }
    })();

  return modelCachePromise;
}

/*
 * Determine whether the upstream model list contains a model.
 */

async function isKnownToNim(
  model
) {
  const models =
    await fetchNimModels();

  const normalized =
    normalizeModel(model);

  return models.some(
    (item) =>
      normalizeModel(
        item?.id
      ) === normalized
  );
}

/* ============================================================
   EXPRESS CONFIGURATION
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
   HELPER FUNCTIONS
   ============================================================ */

function isPlainObject(
  value
) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

function isFiniteNumber(
  value
) {
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

function normalizeModel(
  model
) {
  return String(
    model || ""
  )
    .trim()
    .toLowerCase();
}

function getModelConfig(
  model
) {
  return (
    MODELS[
      normalizeModel(model)
    ] ||
    null
  );
}

/*
 * This is deliberately NOT the old "isSupportedModel".
 *
 * Known models have special configuration.
 *
 * Unknown models are allowed when ALLOW_UNKNOWN_MODELS=true.
 */

function isSupportedModel(
  model
) {
  if (
    getModelConfig(model)
  ) {
    return true;
  }

  return ALLOW_UNKNOWN_MODELS;
}

function clampTemperature(
  value
) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_TEMPERATURE
    );

  return Math.min(
    2,
    Math.max(
      0,
      number
    )
  );
}

function clampTopP(
  value
) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_TOP_P
    );

  return Math.min(
    1,
    Math.max(
      0,
      number
    )
  );
}

function clampMaxTokens(
  value,
  modelConfig
) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_MAX_TOKENS
    );

  /*
   * For explicitly configured models, respect the known
   * maximum.
   *
   * For unknown models, do NOT artificially restrict the
   * user's requested max_tokens to 16384.
   *
   * NVIDIA will validate the actual model limit.
   */

  if (
    modelConfig?.maxOutputTokens
  ) {
    return Math.min(
      modelConfig.maxOutputTokens,
      Math.max(
        1,
        Math.floor(number)
      )
    );
  }

  return Math.max(
    1,
    Math.floor(number)
  );
}

function clampReasoningBudget(
  value
) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_REASONING_BUDGET
    );

  return Math.min(
    65536,
    Math.max(
      0,
      Math.floor(number)
    )
  );
}

/* ============================================================
   REASONING CONTROL
   ============================================================ */

function budgetToReasoningEffort(
  budget,
  modelConfig
) {
  const value =
    clampReasoningBudget(
      budget
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

  return levels[0] || "none";
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
      normalized === "off" ||
      normalized === "false" ||
      normalized === "0"
    ) {
      if (
        levels.includes("none")
      ) {
        return "none";
      }
    }

    if (
      normalized === "med" ||
      normalized === "medium"
    ) {
      if (
        levels.includes("medium")
      ) {
        return "medium";
      }
    }

    if (
      normalized === "maximum" ||
      normalized ===
        "maximum_reasoning"
    ) {
      if (
        levels.includes("max")
      ) {
        return "max";
      }
    }
  }

  return (
    modelConfig?.defaultReasoningEffort ||
    levels[0] ||
    "none"
  );
}

/*
 * Known model:
 *
 *   1. request.reasoning_effort
 *   2. request.reasoning_mode
 *   3. request.reasoning_budget
 *   4. per-model configured effort
 *   5. global effort
 *   6. model default
 *
 * Unknown model:
 *
 *   We DO NOT invent a model-specific default.
 *
 *   If the caller explicitly sends reasoning_effort,
 *   it is passed through.
 *
 *   If the caller explicitly sends reasoning_mode,
 *   it is passed through as reasoning_effort.
 *
 *   If the caller explicitly sends reasoning_budget,
 *   it is passed through as reasoning_budget.
 *
 * This prevents the proxy from sending unsupported reasoning
 * controls to models whose NIM implementation we haven't
 * explicitly configured.
 */

function getRequestedReasoningEffort(
  incoming,
  model,
  modelConfig
) {
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

  if (
    modelConfig
  ) {
    return normalizeReasoningEffort(
      DEFAULT_REASONING_EFFORT,
      modelConfig
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

  return messages
    .filter(
      (message) => {
        if (
          !isPlainObject(
            message
          )
        ) {
          return false;
        }

        if (
          typeof message.role !==
            "string" ||
          message.role.trim() === ""
        ) {
          return false;
        }

        if (
          message.content ===
            undefined ||
          message.content ===
            null
        ) {
          /*
           * Preserve assistant messages that only contain
           * tool calls or other structured content.
           */
          if (
            !Array.isArray(
              message.tool_calls
            )
          ) {
            return false;
          }
        }

        return true;
      }
    )
    .map(
      (message) => ({
        ...message,
        role:
          message.role.trim()
      })
    );
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
  const modelConfig =
    getModelConfig(
      model
    );

  const reasoningEffort =
    getRequestedReasoningEffort(
      incoming,
      model,
      modelConfig
    );

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

  /*
   * ==========================================================
   * STANDARD GENERATION PARAMETERS
   * ==========================================================
   */

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

  /*
   * ==========================================================
   * OPENAI/NIM COMPATIBLE PARAMETERS
   * ==========================================================
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

  /*
   * ==========================================================
   * PRESERVE CHAT TEMPLATE KWARGS
   * ==========================================================
   */

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    )
  ) {
    request.chat_template_kwargs = {
      ...incoming.chat_template_kwargs
    };
  }

  /*
   * ==========================================================
   * PRESERVE EXTRA BODY
   * ==========================================================
   *
   * Some NVIDIA models expose model-specific options through
   * extra_body.
   *
   * We preserve it instead of deleting it.
   */

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

  /*
   * ==========================================================
   * UNKNOWN MODEL
   * ==========================================================
   *
   * Do not guess model-specific reasoning implementation.
   *
   * Explicit reasoning_effort is forwarded.
   */

  if (
    !modelConfig
  ) {
    if (
      incoming.reasoning_effort !==
      undefined
    ) {
      request.reasoning_effort =
        incoming.reasoning_effort;
    } else if (
      incoming.reasoning_mode !==
      undefined
    ) {
      request.reasoning_effort =
        incoming.reasoning_mode;
    }

    /*
     * Only forward an explicitly supplied numeric budget.
     *
     * We do not fabricate one for an unknown model.
     */

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

  /*
   * ==========================================================
   * DEEPSEEK V4 FLASH 0731
   * ==========================================================
   */

  if (
    model ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
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

  /*
   * ==========================================================
   * DEEPSEEK V4 PRO 0813
   * ==========================================================
   */

  if (
    model ===
    "deepseek-ai/deepseek-v4-pro-0813"
  ) {
    request.reasoning_effort =
      reasoningEffort;

    return request;
  }

  /*
   * ==========================================================
   * KIMI K3
   * ==========================================================
   */

  if (
    model ===
    "moonshotai/kimi-k3"
  ) {
    request.reasoning_effort =
      reasoningEffort;

    return request;
  }

  /*
   * ==========================================================
   * NEMOTRON 3 ULTRA
   * ==========================================================
   */

  if (
    model ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    const reasoning =
      normalizeReasoningEffort(
        reasoningEffort,
        modelConfig
      );

    request.reasoning_effort =
      reasoning;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs ||
        {}),

      enable_thinking:
        reasoning !== "none"
    };

    if (
      reasoning === "medium"
    ) {
      request.chat_template_kwargs
        .medium_effort = true;
    } else {
      delete request
        .chat_template_kwargs
        .medium_effort;
    }

    /*
     * Nemotron keeps the real numeric reasoning budget.
     */

    if (
      reasoning !== "none"
    ) {
      request.reasoning_budget =
        clampReasoningBudget(
          incoming.reasoning_budget !==
            undefined
            ? incoming.reasoning_budget
            : MODEL_REASONING_BUDGETS[
                model
              ]
        );
    } else {
      delete request.reasoning_budget;
    }

    /*
     * Recommended when reasoning + tools are used.
     */

    if (
      Array.isArray(
        request.tools
      ) &&
      request.tools.length > 0 &&
      reasoning !== "none"
    ) {
      request.chat_template_kwargs
        .force_nonempty_content = true;
    }

    delete request.nvext;
    delete request.reasoning_mode;

    return request;
  }

  return request;
}

/* ============================================================
   REMOVE REASONING FROM RESPONSE
   ============================================================ */

function stripReasoning(
  parsed
) {
  if (
    !parsed ||
    typeof parsed !==
      "object"
  ) {
    return parsed;
  }

  if (
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
      typeof choice !==
        "object"
    ) {
      continue;
    }

    /*
     * Streaming response.
     */

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

    /*
     * Non-streaming response.
     */

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

  return parsed;
}

/* ============================================================
   SSE PROCESSOR
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
    const line of
      lines
  ) {
    if (
      !line.startsWith(
        "data:"
      )
    ) {
      output.push(
        line
      );

      continue;
    }

    const data =
      line
        .slice(5)
        .trim();

    if (!data) {
      output.push(
        line
      );

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
        JSON.parse(
          data
        );

      stripReasoning(
        parsed
      );

      output.push(
        "data: " +
          JSON.stringify(
            parsed
          )
      );
    } catch (
      error
    ) {
      /*
       * Preserve non-JSON SSE lines.
       */

      output.push(
        line
      );
    }
  }

  if (
    output.length === 0
  ) {
    return "";
  }

  return (
    output.join(
      "\n"
    ) +
    "\n\n"
  );
}

/* ============================================================
   READ STREAM
   ============================================================ */

function readStream(
  stream
) {
  return new Promise(
    (resolve) => {
      let output = "";

      let finished =
        false;

      const finish =
        () => {
          if (
            finished
          ) {
            return;
          }

          finished = true;

          resolve(
            output
          );
        };

      stream.on(
        "data",
        (chunk) => {
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
        finish
      );
    }
  );
}

/* ============================================================
   EXTRACT UPSTREAM ERROR
   ============================================================ */

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
      JSON.parse(
        data
      );

    return (
      parsed?.error
        ?.message ||
      parsed?.message ||
      null
    );
  } catch (
    error
  ) {
    return String(
      data
    );
  }
}

/* ============================================================
   ERROR RESPONSE
   ============================================================ */

function sendError(
  res,
  status,
  message,
  details = null
) {
  if (
    res.headersSent
  ) {
    try {
      res.end();
    } catch (
      error
    ) {}

    return;
  }

  const response = {
    error: {
      message:
        String(
          message
        )
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
    .json(
      response
    );
}

/* ============================================================
   ROOT
   ============================================================ */

app.get(
  "/",
  async (
    req,
    res
  ) => {
    const upstreamModels =
      await fetchNimModels();

    res.json({
      status: "online",

      service:
        "JanitorAI → NVIDIA NIM Proxy",

      default_model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(
          MODELS
        ),

      upstream_model_count:
        upstreamModels.length,

      unknown_models_allowed:
        ALLOW_UNKNOWN_MODELS,

      default_reasoning_effort:
        DEFAULT_REASONING_EFFORT,

      default_reasoning_budget:
        DEFAULT_REASONING_BUDGET,

      nim_api_base:
        NIM_API_BASE,

      timeout_ms:
        NIM_TIMEOUT
    });
  }
);

/* ============================================================
   HEALTH
   ============================================================ */

app.get(
  "/health",
  async (
    req,
    res
  ) => {
    const upstreamModels =
      await fetchNimModels();

    res.json({
      ok: true,

      model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(
          MODELS
        ),

      upstream_model_count:
        upstreamModels.length,

      unknown_models_allowed:
        ALLOW_UNKNOWN_MODELS,

      reasoning:
        DEFAULT_REASONING_EFFORT,

      max_tokens:
        DEFAULT_MAX_TOKENS
    });
  }
);

/* ============================================================
   MODELS ENDPOINT
   ============================================================
 *
 * Return NVIDIA's real model list.
 *
 * This is a major change from the original code:
 *
 * Your proxy is no longer limited to the MODELS object.
 */

app.get(
  "/v1/models",
  async (
    req,
    res
  ) => {
    try {
      const upstreamModels =
        await fetchNimModels();

      /*
       * If NVIDIA successfully returned models, use them.
       */

      if (
        upstreamModels.length > 0
      ) {
        return res.json({
          object:
            "list",

          data:
            upstreamModels
        });
      }

      /*
       * Fallback to the explicitly configured models if
       * upstream discovery is temporarily unavailable.
       */

      const models =
        Object.entries(
          MODELS
        ).map(
          ([id, config]) => ({
            id,

            object:
              "model",

            created:
              Math.floor(
                Date.now() / 1000
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
        object:
          "list",

        data:
          models
      });
    } catch (
      error
    ) {
      return sendError(
        res,
        502,
        "Unable to retrieve NVIDIA NIM models.",
        error?.message ||
          String(
            error
          )
      );
    }
  }
);

/* ============================================================
   FORCE MODEL CACHE REFRESH
   ============================================================ */

app.post(
  "/v1/models/refresh",
  async (
    req,
    res
  ) => {
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
    } catch (
      error
    ) {
      return sendError(
        res,
        502,
        "Unable to refresh NVIDIA NIM model list.",
        error?.message ||
          String(
            error
          )
      );
    }
  }
);

/* ============================================================
   CHAT COMPLETIONS
   ============================================================ */

app.post(
  "/v1/chat/completions",
  async (
    req,
    res
  ) => {
    try {
      /*
       * ======================================================
       * API KEY
       * ======================================================
       */

      if (
        !NIM_API_KEY ||
        typeof NIM_API_KEY !==
          "string"
      ) {
        return sendError(
          res,
          500,
          "NIM_API_KEY is not configured."
        );
      }

      /*
       * ======================================================
       * BODY
       * ======================================================
       */

      const incoming =
        isPlainObject(
          req.body
        )
          ? req.body
          : {};

      /*
       * ======================================================
       * MODEL
       * ======================================================
       */

      const requestedModel =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim()
          ? incoming.model.trim()
          : DEFAULT_MODEL;

      /*
       * Keep the actual model string for NVIDIA.
       *
       * normalizeModel() is only used for lookup.
       */

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

      /*
       * ======================================================
       * MODEL VALIDATION
       * ======================================================
       *
       * IMPORTANT:
       *
       * Unknown models are now allowed.
       *
       * We do not require the model to be in MODELS.
       *
       * If ALLOW_UNKNOWN_MODELS=true, NVIDIA gets to decide
       * whether the model exists/is usable.
       */

      if (
        !isSupportedModel(
          model
        )
      ) {
        return sendError(
          res,
          400,
          `Unsupported model: ${model}`,
          {
            explicitly_configured_models:
              Object.keys(
                MODELS
              )
          }
        );
      }

      /*
       * ======================================================
       * MESSAGES
       * ======================================================
       */

      const messages =
        normalizeMessages(
          incoming.messages
        );

      if (
        messages.length ===
        0
      ) {
        return sendError(
          res,
          400,
          "No valid messages were supplied."
        );
      }

      /*
       * ======================================================
       * STREAM
       * ======================================================
       */

      const stream =
        parseBoolean(
          incoming.stream,
          true
        );

      /*
       * ======================================================
       * BUILD NIM REQUEST
       * ======================================================
       */

      const nimRequest =
        buildNimRequest(
          incoming,
          model,
          messages,
          stream
        );

      /*
       * ======================================================
       * DEBUG
       * ======================================================
       */

      if (
        DEBUG_PROXY
      ) {
        console.log(
          "========== NIM REQUEST =========="
        );

        console.log(
          JSON.stringify(
            nimRequest,
            null,
            2
          )
        );

        console.log(
          "================================="
        );
      }

      /*
       * ======================================================
       * AXIOS CONFIG
       * ======================================================
       */

      const axiosConfig = {
        headers: {
          Authorization:
            `Bearer ${NIM_API_KEY}`,

          "Content-Type":
            "application/json",

          Accept:
            stream
              ? "text/event-stream"
              : "application/json"
        },

        timeout:
          NIM_TIMEOUT,

        validateStatus:
          () => true
      };

      /*
       * ======================================================
       * REQUEST NVIDIA NIM
       * ======================================================
       */

      const response =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          stream
            ? {
                ...axiosConfig,

                responseType:
                  "stream"
              }
            : axiosConfig
        );

      /*
       * ======================================================
       * UPSTREAM ERROR
       * ======================================================
       */

      if (
        response.status <
          200 ||
        response.status >=
          300
      ) {
        let errorBody =
          "";

        if (
          stream &&
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
          try {
            errorBody =
              JSON.stringify(
                response.data
              );
          } catch (
            error
          ) {
            errorBody =
              String(
                response.data
              );
          }
        }

        const upstreamMessage =
          extractErrorMessage(
            errorBody
          );

        console.error(
          "=========================================="
        );

        console.error(
          "NVIDIA NIM ERROR"
        );

        console.error(
          "Model:",
          normalizedModel
        );

        console.error(
          "Status:",
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
          "=========================================="
        );

        /*
         * If NVIDIA says the model doesn't exist, refresh the
         * model cache. This makes newly added/removed models
         * appear quickly without restarting Render.
         */

        if (
          response.status ===
            400 ||
          response.status ===
            404
        ) {
          await fetchNimModels(
            true
          );
        }

        const proxyStatus =
          response.status >=
          500
            ? 502
            : response.status;

        return sendError(
          res,
          proxyStatus,
          upstreamMessage ||
            `NVIDIA NIM returned HTTP ${response.status}.`,
          {
            upstream_status:
              response.status,

            upstream_body:
              errorBody
          }
        );
      }

      /*
       * ======================================================
       * NON-STREAMING
       * ======================================================
       */

      if (
        !stream
      ) {
        const cleaned =
          stripReasoning(
            response.data
          );

        return res
          .status(200)
          .json(
            cleaned
          );
      }

      /*
       * ======================================================
       * STREAM HEADERS
       * ======================================================
       */

      res.status(
        200
      );

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

      /*
       * ======================================================
       * UPSTREAM STREAM
       * ======================================================
       */

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

      let buffer =
        "";

      let ended =
        false;

      let clientDisconnected =
        false;

      /*
       * ======================================================
       * CLEAN UP UPSTREAM
       * ======================================================
       */

      const destroyUpstream =
        () => {
          try {
            if (
              upstream &&
              typeof upstream.destroy ===
                "function"
            ) {
              upstream.destroy();
            }
          } catch (
            error
          ) {}
        };

      /*
       * ======================================================
       * CLIENT DISCONNECT
       * ======================================================
       */

      req.on(
        "aborted",
        () => {
          clientDisconnected =
            true;

          destroyUpstream();
        }
      );

      res.on(
        "close",
        () => {
          if (
            !res.writableEnded
          ) {
            clientDisconnected =
              true;

            destroyUpstream();
          }
        }
      );

      /*
       * ======================================================
       * STREAM DATA
       * ======================================================
       */

      upstream.on(
        "data",
        (chunk) => {
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

          const events =
            buffer.split(
              /\r?\n\r?\n/
            );

          buffer =
            events.pop() ||
            "";

          for (
            const event of
              events
          ) {
            if (
              ended ||
              clientDisconnected
            ) {
              break;
            }

            const output =
              processSSEEvent(
                event
              );

            if (
              !output
            ) {
              continue;
            }

            try {
              res.write(
                output
              );
            } catch (
              error
            ) {
              console.error(
                "Client write error:",
                error.message
              );

              clientDisconnected =
                true;

              destroyUpstream();
            }
          }
        }
      );

      /*
       * ======================================================
       * STREAM END
       * ======================================================
       */

      upstream.on(
        "end",
        () => {
          if (
            ended
          ) {
            return;
          }

          ended =
            true;

          /*
           * Process final partial SSE event.
           */

          if (
            buffer.trim() &&
            !clientDisconnected
          ) {
            const output =
              processSSEEvent(
                buffer
              );

            if (
              output
            ) {
              try {
                res.write(
                  output
                );
              } catch (
                error
              ) {}
            }
          }

          if (
            !clientDisconnected
          ) {
            try {
              res.end();
            } catch (
              error
            ) {}
          }
        }
      );

      /*
       * ======================================================
       * STREAM ERROR
       * ======================================================
       */

      upstream.on(
        "error",
        (error) => {
          if (
            ended
          ) {
            return;
          }

          ended =
            true;

          console.error(
            "NVIDIA stream error:",
            error.message
          );

          if (
            !res.headersSent
          ) {
            return sendError(
              res,
              502,
              "NVIDIA streaming connection failed.",
              error.message
            );
          }

          try {
            res.end();
          } catch (
            endError
          ) {}
        }
      );
    } catch (
      error
    ) {
      console.error(
        "=========================================="
      );

      console.error(
        "PROXY ERROR"
      );

      console.error(
        error?.stack ||
          error?.message ||
          error
      );

      if (
        error?.response?.data
      ) {
        console.error(
          "Upstream response:",
          error.response.data
        );
      }

      console.error(
        "=========================================="
      );

      if (
        res.headersSent
      ) {
        try {
          res.end();
        } catch (
          endError
        ) {}

        return;
      }

      /*
       * Axios timeout.
       */

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

      /*
       * Connection reset.
       */

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

      return sendError(
        res,
        500,
        "Proxy request failed.",
        error?.message ||
          String(
            error
          )
      );
    }
  }
);

/* ============================================================
   404
   ============================================================ */

app.use(
  (
    req,
    res
  ) => {
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
  (
    error,
    req,
    res,
    next
  ) => {
    console.error(
      "Unhandled Express error:",
      error?.stack ||
        error?.message ||
        error
    );

    if (
      res.headersSent
    ) {
      return next(
        error
      );
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
    () => {
      console.log(
        "=========================================="
      );

      console.log(
        "🚀 JanitorAI → NVIDIA NIM Proxy"
      );

      console.log(
        `🌐 Port: ${PORT}`
      );

      console.log(
        `🤖 Default model: ${DEFAULT_MODEL}`
      );

      console.log(
        `🧠 Default reasoning effort: ${DEFAULT_REASONING_EFFORT}`
      );

      console.log(
        `💭 Default reasoning budget: ${DEFAULT_REASONING_BUDGET}`
      );

      console.log(
        `📝 Default max tokens: ${DEFAULT_MAX_TOKENS}`
      );

      console.log(
        `🌡️ Temperature: ${DEFAULT_TEMPERATURE}`
      );

      console.log(
        `🎯 Top P: ${DEFAULT_TOP_P}`
      );

      console.log(
        `⏱️ Timeout: ${NIM_TIMEOUT}ms`
      );

      console.log(
        `🔗 NVIDIA endpoint: ${NIM_API_BASE}`
      );

      console.log(
        `🌐 Unknown NIM models allowed: ${ALLOW_UNKNOWN_MODELS}`
      );

      console.log(
        "📦 Explicitly configured models:"
      );

      Object.entries(
        MODELS
      ).forEach(
        ([id, config]) => {
          console.log(
            `   - ${id}`
          );

          console.log(
            `     reasoning: ${config.reasoningLevels.join(
              ", "
            )}`
          );

          console.log(
            `     max output: ${config.maxOutputTokens}`
          );
        }
      );

      console.log(
        "=========================================="
      );

      /*
       * Warm the model cache after startup.
       *
       * Failure here does NOT stop the proxy.
       */

      fetchNimModels()
        .then(
          (models) => {
            console.log(
              `📡 NVIDIA reports ${models.length} available model(s).`
            );
          }
        )
        .catch(
          (error) => {
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
   SHUTDOWN
   ============================================================ */

function shutdown(
  signal
) {
  console.log(
    `${signal} received. Shutting down...`
  );

  server.close(
    () => {
      console.log(
        "Server closed."
      );

      process.exit(
        0
      );
    }
  );

  setTimeout(
    () => {
      console.error(
        "Forced shutdown."
      );

      process.exit(
        1
      );
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
