```js
"use strict";

/*
 * ============================================================
 * JanitorAI -> NVIDIA NIM OpenAI-Compatible Proxy
 * ============================================================
 *
 * Supported NVIDIA NIM models:
 *
 *   1. nvidia/nemotron-3-ultra-550b-a55b
 *   2. deepseek-ai/deepseek-v4-flash-0731
 *   3. deepseek-ai/deepseek-v4-pro-0813
 *   4. moonshotai/kimi-k3
 *
 * IMPORTANT:
 *
 * Reasoning configuration is centralized in MODEL_CONFIG below.
 *
 * You can change:
 *
 *   reasoningBudget
 *   reasoningEffort
 *   maxTokens
 *   temperature
 *   topP
 *
 * independently for each model.
 *
 * Example:
 *
 *   reasoningBudget: 16384
 *
 * The exact meaning of the budget depends on the model:
 *
 * - Nemotron Ultra:
 *     sends reasoning_budget directly.
 *
 * - DeepSeek V4 Flash:
 *     uses thinking + reasoning_effort.
 *
 * - DeepSeek V4 Pro:
 *     uses thinking + reasoning_effort.
 *
 * - Kimi K3:
 *     always thinks and uses reasoning_effort.
 *
 * DeepSeek/Kimi do NOT currently document a universal
 * token-level reasoning_budget parameter on their NIM
 * endpoints, so this proxy does NOT blindly send an
 * unsupported reasoning_budget to them.
 *
 * ============================================================
 */

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();


/* ============================================================
   ENVIRONMENT
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

const NIM_TIMEOUT =
  Number.isFinite(
    Number(process.env.NIM_TIMEOUT_MS)
  )
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY || "false"
  )
    .trim()
    .toLowerCase() === "true";


/* ============================================================
   GLOBAL DEFAULTS
   ============================================================ */

const DEFAULT_MAX_TOKENS =
  Number.isFinite(
    Number(process.env.DEFAULT_MAX_TOKENS)
  )
    ? Number(process.env.DEFAULT_MAX_TOKENS)
    : 32768;

const DEFAULT_TEMPERATURE =
  Number.isFinite(
    Number(process.env.DEFAULT_TEMPERATURE)
  )
    ? Number(process.env.DEFAULT_TEMPERATURE)
    : 1.0;

const DEFAULT_TOP_P =
  Number.isFinite(
    Number(process.env.DEFAULT_TOP_P)
  )
    ? Number(process.env.DEFAULT_TOP_P)
    : 0.95;

const DEFAULT_REASONING_BUDGET =
  Number.isFinite(
    Number(process.env.DEFAULT_REASONING_BUDGET)
  )
    ? Number(process.env.DEFAULT_REASONING_BUDGET)
    : 16384;

const DEFAULT_REASONING_EFFORT =
  String(
    process.env.DEFAULT_REASONING_EFFORT || "high"
  )
    .trim()
    .toLowerCase();

const DEFAULT_REPETITION_PENALTY =
  Number.isFinite(
    Number(process.env.DEFAULT_REPETITION_PENALTY)
  )
    ? Number(process.env.DEFAULT_REPETITION_PENALTY)
    : 1.0;

const DEFAULT_FREQUENCY_PENALTY =
  Number.isFinite(
    Number(process.env.DEFAULT_FREQUENCY_PENALTY)
  )
    ? Number(process.env.DEFAULT_FREQUENCY_PENALTY)
    : 0.0;

const DEFAULT_PRESENCE_PENALTY =
  Number.isFinite(
    Number(process.env.DEFAULT_PRESENCE_PENALTY)
  )
    ? Number(process.env.DEFAULT_PRESENCE_PENALTY)
    : 0.0;


/* ============================================================
   MODEL CONFIGURATION
   ============================================================
 *
 * THIS IS THE MAIN SECTION YOU WILL EDIT.
 *
 * Change reasoningBudget for each model here.
 *
 * Examples:
 *
 *   reasoningBudget: 8192
 *   reasoningBudget: 16384
 *   reasoningBudget: 32768
 *
 * For models using effort levels, the proxy converts
 * the numeric budget into:
 *
 *   low
 *   high
 *   max
 *
 * You can also directly force an effort level with:
 *
 *   reasoningEffort: "low"
 *   reasoningEffort: "high"
 *   reasoningEffort: "max"
 *
 * Set reasoningEffort to null if you want the proxy to
 * automatically derive it from reasoningBudget.
 *
 * ============================================================
 */

const MODEL_CONFIG = {

  /*
   * ----------------------------------------------------------
   * NVIDIA NEMOTRON ULTRA
   * ----------------------------------------------------------
   *
   * This model supports an actual reasoning_budget parameter.
   */

  "nvidia/nemotron-3-ultra-550b-a55b": {

    displayName:
      "NVIDIA Nemotron 3 Ultra 550B",

    type:
      "nemotron",

    reasoning:
      true,

    reasoningBudget:
      16384,

    reasoningEffort:
      "high",

    maxTokens:
      32768,

    temperature:
      1.0,

    topP:
      0.95
  },


  /*
   * ----------------------------------------------------------
   * DEEPSEEK V4 FLASH 0731
   * ----------------------------------------------------------
   *
   * NVIDIA currently exposes:
   *
   *   thinking: true/false
   *   reasoning_effort: low/high/max
   *
   * We therefore use reasoningBudget as YOUR convenient
   * configuration dial and translate it into effort.
   */

  "deepseek-ai/deepseek-v4-flash-0731": {

    displayName:
      "DeepSeek V4 Flash 0731",

    type:
      "deepseek",

    reasoning:
      true,

    reasoningBudget:
      16384,

    reasoningEffort:
      "high",

    maxTokens:
      32768,

    temperature:
      1.0,

    topP:
      0.95
  },


  /*
   * ----------------------------------------------------------
   * DEEPSEEK V4 PRO 0813
   * ----------------------------------------------------------
   */

  "deepseek-ai/deepseek-v4-pro-0813": {

    displayName:
      "DeepSeek V4 Pro 0813",

    type:
      "deepseek",

    reasoning:
      true,

    reasoningBudget:
      24576,

    reasoningEffort:
      "max",

    maxTokens:
      32768,

    temperature:
      1.0,

    topP:
      0.95
  },


  /*
   * ----------------------------------------------------------
   * KIMI K3
   * ----------------------------------------------------------
   *
   * Kimi K3 ALWAYS reasons.
   *
   * Supported effort:
   *
   *   low
   *   high
   *   max
   *
   * There is no "off" setting.
   */

  "moonshotai/kimi-k3": {

    displayName:
      "Kimi K3",

    type:
      "kimi",

    reasoning:
      true,

    reasoningBudget:
      32768,

    reasoningEffort:
      "max",

    maxTokens:
      32768,

    temperature:
      1.0,

    topP:
      0.95
  }
};


/* ============================================================
   MODEL ALIASES
   ============================================================
 *
 * These let clients use shorter model names if desired.
 *
 * Example:
 *
 *   "kimi-k3"
 *
 * instead of:
 *
 *   "moonshotai/kimi-k3"
 *
 * ============================================================
 */

const MODEL_ALIASES = {

  "nemotron":
    "nvidia/nemotron-3-ultra-550b-a55b",

  "ultra":
    "nvidia/nemotron-3-ultra-550b-a55b",

  "deepseek-flash":
    "deepseek-ai/deepseek-v4-flash-0731",

  "deepseek-v4-flash":
    "deepseek-ai/deepseek-v4-flash-0731",

  "flash":
    "deepseek-ai/deepseek-v4-flash-0731",

  "deepseek-pro":
    "deepseek-ai/deepseek-v4-pro-0813",

  "deepseek-v4-pro":
    "deepseek-ai/deepseek-v4-pro-0813",

  "pro":
    "deepseek-ai/deepseek-v4-pro-0813",

  "kimi":
    "moonshotai/kimi-k3",

  "kimi-k3":
    "moonshotai/kimi-k3"
};


/* ============================================================
   EXPRESS CONFIGURATION
   ============================================================ */

app.disable("x-powered-by");

app.use(cors());

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


/* ============================================================
   MODEL RESOLUTION
   ============================================================ */

function resolveModel(model) {

  const normalized =
    normalizeModel(model);

  if (
    MODEL_ALIASES[normalized]
  ) {
    return MODEL_ALIASES[normalized];
  }

  return (
    model ||
    DEFAULT_MODEL
  )
    .trim();
}


function getModelConfig(model) {

  const resolved =
    resolveModel(model);

  return (
    MODEL_CONFIG[resolved] ||
    null
  );
}


/* ============================================================
   REASONING HELPERS
   ============================================================ */

function normalizeReasoningEffort(
  value,
  fallback = "high"
) {

  if (
    typeof value !== "string"
  ) {
    return fallback;
  }

  const effort =
    value
      .trim()
      .toLowerCase();

  if (
    effort === "low"
  ) {
    return "low";
  }

  if (
    effort === "high"
  ) {
    return "high";
  }

  if (
    effort === "max"
  ) {
    return "max";
  }

  /*
   * Accept a few convenient aliases.
   */

  if (
    effort === "medium" ||
    effort === "med"
  ) {
    return "high";
  }

  if (
    effort === "maximum"
  ) {
    return "max";
  }

  return fallback;
}


/*
 * Converts your numeric reasoning budget into the
 * effort levels supported by DeepSeek/Kimi.
 *
 * You can change these thresholds if you want.
 */

function budgetToEffort(
  budget
) {

  const numeric =
    Math.max(
      0,
      Math.floor(
        numberOrDefault(
          budget,
          DEFAULT_REASONING_BUDGET
        )
      )
    );

  if (
    numeric <= 8192
  ) {
    return "low";
  }

  if (
    numeric <= 16384
  ) {
    return "high";
  }

  return "max";
}


function clampReasoningBudget(
  value,
  fallback
) {

  const budget =
    numberOrDefault(
      value,
      fallback
    );

  /*
   * NVIDIA documents -1 as disabling budget
   * enforcement on APIs that support reasoning_budget.
   */

  return Math.min(
    32768,
    Math.max(
      -1,
      Math.floor(budget)
    )
  );
}


function clampMaxTokens(
  value,
  fallback
) {

  const tokens =
    numberOrDefault(
      value,
      fallback
    );

  return Math.min(
    32768,
    Math.max(
      1,
      Math.floor(tokens)
    )
  );
}


function clampTemperature(
  value,
  fallback
) {

  const temperature =
    numberOrDefault(
      value,
      fallback
    );

  return Math.min(
    2,
    Math.max(
      0,
      temperature
    )
  );
}


function clampTopP(
  value,
  fallback
) {

  const topP =
    numberOrDefault(
      value,
      fallback
    );

  return Math.min(
    1,
    Math.max(
      0,
      topP
    )
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
    .filter((message) => {

      if (
        !isPlainObject(message)
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
        return false;
      }

      return true;
    })
    .map((message) => ({
      ...message,

      role:
        message.role.trim()
    }));
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

  const config =
    getModelConfig(model);

  /*
   * Unknown models are still allowed.
   *
   * This keeps the proxy compatible with additional
   * NVIDIA NIM models without requiring a code change.
   */

  const modelConfig =
    config || {

      type:
        "generic",

      reasoning:
        false,

      reasoningBudget:
        DEFAULT_REASONING_BUDGET,

      reasoningEffort:
        DEFAULT_REASONING_EFFORT,

      maxTokens:
        DEFAULT_MAX_TOKENS,

      temperature:
        DEFAULT_TEMPERATURE,

      topP:
        DEFAULT_TOP_P
    };


  /*
   * ----------------------------------------------------------
   * USER OVERRIDES
   * ----------------------------------------------------------
   *
   * Incoming request parameters override the model defaults.
   */

  const requestedMaxTokens =
    incoming.max_tokens !== undefined
      ? incoming.max_tokens
      : modelConfig.maxTokens;

  const requestedTemperature =
    incoming.temperature !== undefined
      ? incoming.temperature
      : modelConfig.temperature;

  const requestedTopP =
    incoming.top_p !== undefined
      ? incoming.top_p
      : modelConfig.topP;

  const requestedBudget =
    incoming.reasoning_budget !== undefined
      ? incoming.reasoning_budget
      : modelConfig.reasoningBudget;


  const reasoningBudget =
    clampReasoningBudget(
      requestedBudget,
      modelConfig.reasoningBudget
    );


  /*
   * Explicit effort wins.
   *
   * Otherwise derive effort from the numeric budget.
   */

  let reasoningEffort;

  if (
    incoming.reasoning_effort !==
    undefined
  ) {

    reasoningEffort =
      normalizeReasoningEffort(
        incoming.reasoning_effort,
        modelConfig.reasoningEffort
      );

  } else if (
    modelConfig.reasoningEffort
  ) {

    reasoningEffort =
      normalizeReasoningEffort(
        modelConfig.reasoningEffort,
        budgetToEffort(
          reasoningBudget
        )
      );

  } else {

    reasoningEffort =
      budgetToEffort(
        reasoningBudget
      );
  }


  const request = {

    model,

    messages,

    temperature:
      clampTemperature(
        requestedTemperature,
        DEFAULT_TEMPERATURE
      ),

    top_p:
      clampTopP(
        requestedTopP,
        DEFAULT_TOP_P
      ),

    max_tokens:
      clampMaxTokens(
        requestedMaxTokens,
        DEFAULT_MAX_TOKENS
      ),

    stream
  };


  /* ==========================================================
     STANDARD PARAMETERS
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


  /* ==========================================================
     OPTIONAL OPENAI-COMPATIBLE PARAMETERS
     ========================================================== */

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

    /*
     * Useful for long-context models.
     */

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


  /* ==========================================================
     CHAT TEMPLATE KWARGS
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
     NEMOTRON
     ========================================================== */

  if (
    modelConfig.type ===
    "nemotron"
  ) {

    /*
     * Nemotron uses reasoning_effort and
     * reasoning_budget.
     */

    request.reasoning_effort =
      reasoningEffort === "low"
        ? "medium"
        : reasoningEffort;


    /*
     * NVIDIA's hosted API accepts the
     * actual reasoning budget here.
     */

    request.reasoning_budget =
      reasoningBudget;


    request.chat_template_kwargs = {

      ...(
        request.chat_template_kwargs ||
        {}
      ),

      enable_thinking:
        true
    };


    /*
     * Medium effort compatibility.
     */

    if (
      reasoningEffort === "high"
    ) {

      request.chat_template_kwargs
        .medium_effort = true;

    } else {

      delete request
        .chat_template_kwargs
        .medium_effort;
    }


    /*
     * Tools + reasoning.
     */

    if (
      Array.isArray(
        request.tools
      ) &&
      request.tools.length > 0
    ) {

      request.chat_template_kwargs
        .force_nonempty_content = true;
    }


    delete request.nvext;

    delete request.reasoning_mode;
  }


  /* ==========================================================
     DEEPSEEK V4 FLASH / PRO
     ========================================================== */

  else if (
    modelConfig.type ===
    "deepseek"
  ) {

    /*
     * DeepSeek's NVIDIA NIM interface uses
     * chat_template_kwargs.
     *
     * Flash documents:
     *
     *   thinking: true
     *   reasoning_effort: high
     *
     * Pro documents thinking control and
     * reasoning-effort levels.
     *
     * We therefore deliberately DO NOT send
     * reasoning_budget as a top-level field.
     */

    request.chat_template_kwargs = {

      ...(
        request.chat_template_kwargs ||
        {}
      ),

      thinking:
        true,

      reasoning_effort:
        reasoningEffort
    };


    /*
     * Remove any accidentally supplied
     * Nemotron-specific reasoning controls.
     */

    delete request.reasoning_budget;

    delete request.reasoning_effort;

    delete request.reasoning_mode;

    delete request.nvext;
  }


  /* ==========================================================
     KIMI K3
     ========================================================== */

  else if (
    modelConfig.type ===
    "kimi"
  ) {

    /*
     * Kimi K3 ALWAYS thinks.
     *
     * NVIDIA documents:
     *
     *   low
     *   high
     *   max
     *
     * There is no reasoning-off mode.
     */

    request.reasoning_effort =
      reasoningEffort;


    /*
     * Do not send a fake numeric reasoning_budget.
     */

    delete request.reasoning_budget;

    delete request.reasoning_mode;

    delete request.nvext;


    /*
     * Kimi does not need:
     *
     *   enable_thinking: false
     *
     * Thinking remains enabled.
     */
  }


  /* ==========================================================
     GENERIC MODEL
     ========================================================== */

  else {

    /*
     * For models we don't have a specific configuration for,
     * don't inject model-specific reasoning parameters.
     */

    delete request.reasoning_budget;

    delete request.reasoning_effort;

    delete request.reasoning_mode;

    delete request.nvext;
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
    typeof parsed !== "object"
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
      typeof choice !== "object"
    ) {
      continue;
    }


    /*
     * Streaming.
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
     * Non-streaming.
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
      line
        .slice(5)
        .trim();


    if (
      !data
    ) {

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

      stripReasoning(parsed);

      output.push(
        "data: " +
        JSON.stringify(parsed)
      );

    } catch (_) {

      /*
       * Preserve non-JSON SSE.
       */

      output.push(line);
    }
  }


  if (
    output.length === 0
  ) {
    return "";
  }


  return (
    output.join("\n") +
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

      let finished = false;


      const finish = () => {

        if (
          finished
        ) {
          return;
        }

        finished = true;

        resolve(output);
      };


      stream.on(
        "data",
        (chunk) => {

          output +=
            chunk.toString("utf8");


          /*
           * Prevent huge upstream errors
           * from consuming memory.
           */

          if (
            output.length >
            2 * 1024 * 1024
          ) {

            output =
              output.slice(
                0,
                2 * 1024 * 1024
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
   ERROR HANDLING
   ============================================================ */

function extractErrorMessage(
  data
) {

  if (
    !data
  ) {
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

  } catch (_) {

    return String(data);
  }
}


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
    } catch (_) {}

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
   ROOT
   ============================================================ */

app.get(
  "/",
  (req, res) => {

    const config =
      getModelConfig(
        DEFAULT_MODEL
      );


    res.json({

      status:
        "online",

      service:
        "JanitorAI -> NVIDIA NIM Proxy",

      default_model:
        resolveModel(DEFAULT_MODEL),

      model_config:
        config,

      available_models:
        Object.keys(
          MODEL_CONFIG
        ),

      aliases:
        MODEL_ALIASES,

      reasoning_budget:
        config?.reasoningBudget ??
        DEFAULT_REASONING_BUDGET,

      reasoning_effort:
        config?.reasoningEffort ??
        DEFAULT_REASONING_EFFORT,

      max_tokens:
        config?.maxTokens ??
        DEFAULT_MAX_TOKENS,

      nim_api_base:
        NIM_API_BASE,

      timeout_ms:
        NIM_TIMEOUT
    });
  }
);


/* ============================================================
   MODEL LIST
   ============================================================ */

app.get(
  "/v1/models",
  (req, res) => {

    const models =
      Object.entries(
        MODEL_CONFIG
      ).map(
        ([id, config]) => ({

          id,

          object:
            "model",

          owned_by:
            config.type === "kimi"
              ? "moonshotai"
              : config.type === "deepseek"
                ? "deepseek-ai"
                : "nvidia",

          reasoning:
            config.reasoning,

          reasoning_budget:
            config.reasoningBudget,

          reasoning_effort:
            config.reasoningEffort,

          max_tokens:
            config.maxTokens
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
   HEALTH
   ============================================================ */

app.get(
  "/health",
  (req, res) => {

    const resolved =
      resolveModel(
        DEFAULT_MODEL
      );

    const config =
      getModelConfig(
        resolved
      );


    res.json({

      ok:
        true,

      model:
        resolved,

      model_type:
        config?.type ||
        "generic",

      reasoning_budget:
        config?.reasoningBudget ??
        DEFAULT_REASONING_BUDGET,

      reasoning_effort:
        config?.reasoningEffort ??
        DEFAULT_REASONING_EFFORT,

      max_tokens:
        config?.maxTokens ??
        DEFAULT_MAX_TOKENS
    });
  }
);


/* ============================================================
   CHAT COMPLETIONS
   ============================================================ */

app.post(
  "/v1/chat/completions",
  async (req, res) => {

    try {

      /* ------------------------------------------------------
         API KEY
         ------------------------------------------------------ */

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


      /* ------------------------------------------------------
         BODY
         ------------------------------------------------------ */

      const incoming =
        isPlainObject(
          req.body
        )
          ? req.body
          : {};


      /* ------------------------------------------------------
         MODEL
         ------------------------------------------------------ */

      const requestedModel =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim()
          ? incoming.model.trim()
          : DEFAULT_MODEL;


      const model =
        resolveModel(
          requestedModel
        );


      /* ------------------------------------------------------
         MESSAGES
         ------------------------------------------------------ */

      const messages =
        normalizeMessages(
          incoming.messages
        );


      if (
        messages.length === 0
      ) {

        return sendError(
          res,
          400,
          "No valid messages were supplied."
        );
      }


      /* ------------------------------------------------------
         STREAM
         ------------------------------------------------------ */

      const stream =
        parseBoolean(
          incoming.stream,
          true
        );


      /* ------------------------------------------------------
         BUILD REQUEST
         ------------------------------------------------------ */

      const nimRequest =
        buildNimRequest(
          incoming,
          model,
          messages,
          stream
        );


      /* ------------------------------------------------------
         DEBUG
         ------------------------------------------------------ */

      if (
        DEBUG_PROXY
      ) {

        /*
         * Never print the API key.
         */

        console.log(
          "\n========== NIM REQUEST =========="
        );

        console.log(
          JSON.stringify(
            nimRequest,
            null,
            2
          )
        );

        console.log(
          "=================================\n"
        );
      }


      /* ------------------------------------------------------
         AXIOS CONFIG
         ------------------------------------------------------ */

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


      /* ------------------------------------------------------
         NVIDIA REQUEST
         ------------------------------------------------------ */

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


      /* ------------------------------------------------------
         UPSTREAM ERROR
         ------------------------------------------------------ */

      if (
        response.status < 200 ||
        response.status >= 300
      ) {

        let errorBody = "";


        if (
          stream &&
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

          } catch (_) {

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
          model
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
          response.status >= 500
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


      /* ------------------------------------------------------
         NON-STREAMING
         ------------------------------------------------------ */

      if (
        !stream
      ) {

        const cleaned =
          stripReasoning(
            response.data
          );


        return res
          .status(200)
          .json(cleaned);
      }


      /* ------------------------------------------------------
         STREAM HEADERS
         ------------------------------------------------------ */

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


      /* ------------------------------------------------------
         UPSTREAM STREAM
         ------------------------------------------------------ */

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


      /* ------------------------------------------------------
         CLIENT DISCONNECT
         ------------------------------------------------------ */

      req.on(
        "aborted",
        () => {

          clientDisconnected =
            true;

          try {

            upstream.destroy();

          } catch (_) {}
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

            } catch (_) {}
          }
        }
      );


      /* ------------------------------------------------------
         STREAM DATA
         ------------------------------------------------------ */

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
            events.pop() || "";


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

            } catch (error) {

              console.error(
                "Client write error:",
                error.message
              );


              clientDisconnected =
                true;


              try {

                upstream.destroy();

              } catch (_) {}
            }
          }
        }
      );


      /* ------------------------------------------------------
         STREAM END
         ------------------------------------------------------ */

      upstream.on(
        "end",
        () => {

          if (
            ended
          ) {
            return;
          }


          ended = true;


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

              } catch (_) {}
            }
          }


          if (
            !clientDisconnected
          ) {

            try {

              res.end();

            } catch (_) {}
          }
        }
      );


      /* ------------------------------------------------------
         STREAM ERROR
         ------------------------------------------------------ */

      upstream.on(
        "error",
        (error) => {

          if (
            ended
          ) {
            return;
          }


          ended = true;


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

          } catch (_) {}
        }
      );
    }


    /* ========================================================
       TOP LEVEL ERROR
       ======================================================== */

    catch (error) {

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

      console.error(
        "=========================================="
      );


      if (
        res.headersSent
      ) {

        try {

          res.end();

        } catch (_) {}

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
          String(error)
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
    () => {

      const resolvedDefault =
        resolveModel(
          DEFAULT_MODEL
        );

      const config =
        getModelConfig(
          resolvedDefault
        );


      console.log(
        "=========================================="
      );

      console.log(
        "🚀 JanitorAI -> NVIDIA NIM Proxy"
      );

      console.log(
        `🌐 Port: ${PORT}`
      );

      console.log(
        `🤖 Default model: ${resolvedDefault}`
      );

      console.log(
        `🧠 Model type: ${config?.type || "generic"}`
      );

      console.log(
        `💭 Reasoning budget: ${
          config?.reasoningBudget ??
          DEFAULT_REASONING_BUDGET
        }`
      );

      console.log(
        `🎯 Reasoning effort: ${
          config?.reasoningEffort ??
          DEFAULT_REASONING_EFFORT
        }`
      );

      console.log(
        `📝 Max tokens: ${
          config?.maxTokens ??
          DEFAULT_MAX_TOKENS
        }`
      );

      console.log(
        `🌡️ Temperature: ${
          config?.temperature ??
          DEFAULT_TEMPERATURE
        }`
      );

      console.log(
        `🎯 Top P: ${
          config?.topP ??
          DEFAULT_TOP_P
        }`
      );

      console.log(
        `⏱️ Timeout: ${NIM_TIMEOUT}ms`
      );

      console.log(
        `🔗 NVIDIA endpoint: ${NIM_API_BASE}`
      );

      console.log(
        "=========================================="
      );


      console.log(
        "\nAvailable models:"
      );


      for (
        const [
          modelId,
          modelConfig
        ] of Object.entries(
          MODEL_CONFIG
        )
      ) {

        console.log(
          `  • ${modelId}`
        );

        console.log(
          `    reasoning budget: ${modelConfig.reasoningBudget}`
        );

        console.log(
          `    reasoning effort: ${modelConfig.reasoningEffort}`
        );
      }


      console.log(
        "\n==========================================\n"
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
    NIM_TIMEOUT + 5000
  );


/* ============================================================
   GRACEFUL SHUTDOWN
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

      process.exit(0);
    }
  );


  setTimeout(
    () => {

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
  () => shutdown("SIGTERM")
);

process.on(
  "SIGINT",
  () => shutdown("SIGINT")
);
```
