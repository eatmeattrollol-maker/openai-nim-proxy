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
 * Supported models
 *
 * These are NVIDIA NIM hosted model IDs.
 */
const MODELS = {
  "deepseek-ai/deepseek-v4-flash-0731": {
    name: "DeepSeek V4 Flash 0731",
    provider: "DeepSeek AI",

    /*
     * NVIDIA supports:
     * none / low / high / max
     *
     * The API examples/documentation primarily expose
     * reasoning_effort for this model.
     */
    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "low",

    /*
     * NVIDIA's current NIM endpoint documents max_tokens
     * up to 16384 for this model.
     */
    maxOutputTokens: 16384,

    /*
     * This model uses chat_template_kwargs.thinking.
     */
    thinkingParameter: "chat_template"
  },

  "deepseek-ai/deepseek-v4-pro-0813": {
    name: "DeepSeek V4 Pro 0813",
    provider: "DeepSeek AI",

    /*
     * NVIDIA documents:
     * none / high / max
     */
    reasoningLevels: [
      "none",
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "low",

    maxOutputTokens: 16384,

    thinkingParameter: "chat_template"
  },

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",

    /*
     * NVIDIA documents:
     * low / high / max
     *
     * Thinking is always enabled for Kimi K3.
     */
    reasoningLevels: [
      "low",
      "high",
      "max"
    ],

    defaultReasoningEffort: "max",

    /*
     * NVIDIA documents max_tokens:
     * 1 - 65536
     */
    maxOutputTokens: 65536,

    thinkingParameter: "native"
  },

  /*
   * Keep your existing Nemotron model available.
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
 * DEFAULT REASONING SETTINGS
 * ============================================================
 *
 * You can change these values directly.
 *
 * Example:
 *
 *   reasoningBudget: 8192
 *
 * or:
 *
 *   reasoningEffort: "max"
 *
 * IMPORTANT:
 *
 * The newer DeepSeek V4 and Kimi K3 NIM APIs expose
 * reasoning_effort rather than a universal numeric
 * reasoning_budget parameter.
 *
 * Therefore this proxy supports BOTH:
 *
 *   reasoningEffort
 *   reasoningBudget
 *
 * reasoningEffort is sent to NVIDIA.
 *
 * reasoningBudget is used by this proxy to choose an
 * appropriate reasoning level when reasoningEffort was
 * not explicitly supplied.
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
 * These are the easiest values to edit.
 *
 * NOTE:
 * NVIDIA's current DeepSeek V4/Kimi K3 hosted APIs use
 * reasoning_effort rather than accepting an arbitrary
 * numeric reasoning budget.
 *
 * The numbers below are therefore used as a policy layer
 * that maps your desired budget to low/high/max.
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
 * If you want exact control over the NVIDIA parameter,
 * edit these values.
 *
 * Valid values depend on the model.
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
 * Long-running AI requests need a generous timeout.
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
 * Debug mode should NEVER print your API key.
 */
const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY ||
      "false"
  )
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   EXPRESS CONFIGURATION
   ============================================================ */

app.disable("x-powered-by");

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
  if (
    isFiniteNumber(value)
  ) {
    return Number(value);
  }

  return fallback;
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
  return String(
    model || ""
  )
    .trim()
    .toLowerCase();
}

function getModelConfig(model) {
  const normalized =
    normalizeModel(model);

  return (
    MODELS[normalized] ||
    null
  );
}

function isSupportedModel(model) {
  return Boolean(
    getModelConfig(model)
  );
}

function clampTemperature(value) {
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

function clampTopP(value) {
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

  const modelMaximum =
    modelConfig?.maxOutputTokens ||
    16384;

  return Math.min(
    modelMaximum,
    Math.max(
      1,
      Math.floor(number)
    )
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

/*
 * Convert a numeric reasoning budget into one of the levels
 * actually supported by the model.
 *
 * This DOES NOT pretend that NVIDIA accepts arbitrary
 * reasoning token counts for DeepSeek V4 / Kimi K3.
 *
 * It is simply a convenient local control layer.
 */
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

  /*
   * Models that support "none".
   */
  if (
    levels.includes("none") &&
    value <= 0
  ) {
    return "none";
  }

  /*
   * Low.
   */
  if (
    levels.includes("low") &&
    value <= 8192
  ) {
    return "low";
  }

  /*
   * Medium.
   */
  if (
    levels.includes("medium") &&
    value <= 16384
  ) {
    return "medium";
  }

  /*
   * High.
   */
  if (
    levels.includes("high") &&
    value <= 24576
  ) {
    return "high";
  }

  /*
   * Max.
   */
  if (
    levels.includes("max")
  ) {
    return "max";
  }

  /*
   * Fall back to the highest supported level.
   */
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

    /*
     * Friendly aliases.
     */
    if (
      normalized === "off" ||
      normalized === "false" ||
      normalized === "0" ||
      normalized === "none"
    ) {
      if (
        levels.includes("none")
      ) {
        return "none";
      }
    }

    if (
      normalized === "medium" ||
      normalized === "med"
    ) {
      if (
        levels.includes("medium")
      ) {
        return "medium";
      }
    }

    if (
      normalized === "maximum" ||
      normalized === "maximum_reasoning"
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
 * Determine the actual reasoning effort for this request.
 *
 * Priority:
 *
 * 1. request.reasoning_effort
 * 2. request.reasoning_mode
 * 3. request.reasoning_budget
 * 4. per-model configured effort
 * 5. global configured effort
 * 6. model default
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

  return normalizeReasoningEffort(
    DEFAULT_REASONING_EFFORT,
    modelConfig
  );
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
           * Allow assistant messages that contain
           * tool calls/reasoning content without normal
           * textual content.
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
   * STANDARD PARAMETERS
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
   * OPTIONAL OPENAI-COMPATIBLE PARAMETERS
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
    "stream_options"
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
   * EXISTING CHAT TEMPLATE KWARGS
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
   * DEEPSEEK V4 FLASH 0731
   * ==========================================================
   *
   * NVIDIA's current example uses:
   *
   * extra_body:
   * {
   *   chat_template_kwargs: {
   *     thinking: true,
   *     reasoning_effort: "high"
   *   }
   * }
   *
   * Through the OpenAI-compatible endpoint we send the
   * corresponding chat_template_kwargs directly.
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
   *
   * NVIDIA documents reasoning_effort:
   *
   * none
   * high
   * max
   *
   * The NIM API translates this into the model's
   * chat-template configuration.
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
   *
   * NVIDIA documents:
   *
   * low
   * high
   * max
   *
   * Thinking is always enabled.
   *
   * Kimi K3 also requires preserved reasoning content
   * to be passed back on multi-turn reasoning/tool calls.
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
     * Nemotron supports a real numeric reasoning_budget.
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
     * Recommended when tools are used with reasoning.
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
       * Preserve non-JSON SSE data.
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
          output +=
            chunk.toString(
              "utf8"
            );

          /*
           * Prevent an enormous upstream error from
           * consuming too much memory.
           */
          if (
            output.length >
            2 *
              1024 *
              1024
          ) {
            output =
              output.slice(
                0,
                2 *
                  1024 *
                  1024
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
  (req, res) => {
    res.json({
      status: "online",

      service:
        "JanitorAI → NVIDIA NIM Proxy",

      default_model:
        DEFAULT_MODEL,

      supported_models:
        Object.keys(
          MODELS
        ),

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
  (req, res) => {
    res.json({
      ok: true,

      model:
        DEFAULT_MODEL,

      supported_models:
        Object.keys(
          MODELS
        ),

      reasoning:
        DEFAULT_REASONING_EFFORT,

      max_tokens:
        DEFAULT_MAX_TOKENS
    });
  }
);

/* ============================================================
   MODELS ENDPOINT
   ============================================================ */

app.get(
  "/v1/models",
  (req, res) => {
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

    res.json({
      object:
        "list",

      data:
        models
    });
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

      const model =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim()
          ? incoming.model.trim()
          : DEFAULT_MODEL;

      /*
       * ======================================================
       * MODEL VALIDATION
       * ======================================================
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
            supported_models:
              Object.keys(
                MODELS
              )
          }
        );
      }

      const normalizedModel =
        normalizeModel(
          model
        );

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
          normalizedModel,
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
       * CLIENT DISCONNECT
       * ======================================================
       */

      req.on(
        "aborted",
        () => {
          clientDisconnected =
            true;

          try {
            upstream.destroy();
          } catch (
            error
          ) {}
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

            try {
              upstream.destroy();
            } catch (
              error
            ) {}
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

              try {
                upstream.destroy();
              } catch (
                destroyError
              ) {}
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
  (req, res) => {
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
        `💭 Default reasoning budget policy: ${DEFAULT_REASONING_BUDGET}`
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
        "📦 Supported models:"
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
    NIM_TIMEOUT + 5000
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
