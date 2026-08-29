```js
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

/* ============================================================
   CONFIGURATION
   ============================================================ */

const PORT = Number(process.env.PORT) || 3000;

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
  Number.isFinite(Number(process.env.NIM_TIMEOUT_MS))
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const DEBUG_PROXY =
  String(process.env.DEBUG_PROXY || "false")
    .trim()
    .toLowerCase() === "true";


/* ============================================================
   MODEL CONFIGURATION
   ============================================================
   
   THIS IS THE MAIN SECTION YOU EDIT.
   
   reasoningBudget:
     Your preferred numeric "budget" for your own configuration.
   
   reasoningEffort:
     The actual setting sent to NVIDIA:
   
       low
       high
       max
   
   IMPORTANT:
   
   DeepSeek V4 and Kimi K3 do NOT use a universal numeric
   reasoning_budget parameter in the current NVIDIA examples.
   
   Therefore:
   
     reasoningBudget = YOUR CONFIGURATION VALUE
   
     reasoningEffort = ACTUAL NIM VALUE
   
   You can adjust both independently.
   
   ============================================================ */

const MODEL_CONFIG = {

  "deepseek-ai/deepseek-v4-flash-0731": {

    name:
      "DeepSeek V4 Flash 0731",

    provider:
      "deepseek-ai",

    reasoning:
      true,

    reasoningBudget:
      16384,

    reasoningEffort:
      "high",

    maxTokens:
      16384,

    temperature:
      1.0,

    topP:
      0.95
  },


  "deepseek-ai/deepseek-v4-pro-0813": {

    name:
      "DeepSeek V4 Pro 0813",

    provider:
      "deepseek-ai",

    reasoning:
      true,

    reasoningBudget:
      24576,

    reasoningEffort:
      "max",

    maxTokens:
      16384,

    temperature:
      1.0,

    topP:
      0.95
  },


  "moonshotai/kimi-k3": {

    name:
      "Kimi K3",

    provider:
      "moonshotai",

    reasoning:
      true,

    reasoningBudget:
      32768,

    reasoningEffort:
      "max",

    maxTokens:
      16384,

    temperature:
      1.0,

    topP:
      0.95
  },


  "nvidia/nemotron-3-ultra-550b-a55b": {

    name:
      "NVIDIA Nemotron 3 Ultra 550B",

    provider:
      "nvidia",

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
  }
};


/* ============================================================
   OPTIONAL MODEL ALIASES
   ============================================================ */

const MODEL_ALIASES = {

  "flash":
    "deepseek-ai/deepseek-v4-flash-0731",

  "deepseek-flash":
    "deepseek-ai/deepseek-v4-flash-0731",

  "deepseek-v4-flash":
    "deepseek-ai/deepseek-v4-flash-0731",

  "pro":
    "deepseek-ai/deepseek-v4-pro-0813",

  "deepseek-pro":
    "deepseek-ai/deepseek-v4-pro-0813",

  "deepseek-v4-pro":
    "deepseek-ai/deepseek-v4-pro-0813",

  "kimi":
    "moonshotai/kimi-k3",

  "kimi-k3":
    "moonshotai/kimi-k3",

  "nemotron":
    "nvidia/nemotron-3-ultra-550b-a55b",

  "ultra":
    "nvidia/nemotron-3-ultra-550b-a55b"
};


/* ============================================================
   EXPRESS
   ============================================================ */

app.disable("x-powered-by");

app.use(cors());

app.use(
  express.json({
    limit: "100mb"
  })
);


/* ============================================================
   HELPERS
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
    return Number.isFinite(Number(value));
  }

  return false;
}


function numberOrDefault(value, fallback) {

  if (isFiniteNumber(value)) {
    return Number(value);
  }

  return fallback;
}


function parseBoolean(value, fallback) {

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


function resolveModel(model) {

  const normalized =
    normalizeModel(model);

  return (
    MODEL_ALIASES[normalized] ||
    String(model || "").trim()
  );
}


function getModelConfig(model) {

  return MODEL_CONFIG[
    resolveModel(model)
  ] || null;
}


function clampMaxTokens(value, fallback) {

  const number =
    numberOrDefault(
      value,
      fallback
    );

  return Math.min(
    32768,
    Math.max(
      1,
      Math.floor(number)
    )
  );
}


function clampTemperature(value, fallback) {

  const number =
    numberOrDefault(
      value,
      fallback
    );

  return Math.min(
    2,
    Math.max(
      0,
      number
    )
  );
}


function clampTopP(value, fallback) {

  const number =
    numberOrDefault(
      value,
      fallback
    );

  return Math.min(
    1,
    Math.max(
      0,
      number
    )
  );
}


function normalizeReasoningEffort(
  value,
  fallback = "high"
) {

  const normalized =
    String(value || "")
      .trim()
      .toLowerCase();

  if (
    normalized === "low"
  ) {
    return "low";
  }

  if (
    normalized === "high"
  ) {
    return "high";
  }

  if (
    normalized === "max" ||
    normalized === "maximum"
  ) {
    return "max";
  }

  return fallback;
}


/* ============================================================
   MESSAGE NORMALIZATION
   ============================================================ */

function normalizeMessages(messages) {

  if (!Array.isArray(messages)) {
    return [];
  }

  return messages
    .filter((message) => {

      if (!isPlainObject(message)) {
        return false;
      }

      if (
        typeof message.role !== "string" ||
        message.role.trim() === ""
      ) {
        return false;
      }

      if (
        message.content === undefined ||
        message.content === null
      ) {
        return false;
      }

      return true;
    })
    .map((message) => ({
      ...message,
      role: message.role.trim()
    }));
}


/* ============================================================
   BUILD NVIDIA REQUEST
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
   * If somebody requests a model not listed above,
   * still allow it through as a generic NIM model.
   */

  const modelConfig =
    config || {

      name:
        model,

      provider:
        "unknown",

      reasoning:
        false,

      reasoningBudget:
        0,

      reasoningEffort:
        "high",

      maxTokens:
        16384,

      temperature:
        1.0,

      topP:
        0.95
    };


  const maxTokens =
    clampMaxTokens(
      incoming.max_tokens,
      modelConfig.maxTokens
    );


  const temperature =
    clampTemperature(
      incoming.temperature,
      modelConfig.temperature
    );


  const topP =
    clampTopP(
      incoming.top_p,
      modelConfig.topP
    );


  /*
   * User can override reasoning_effort directly.
   *
   * Example:
   *
   * {
   *   "reasoning_effort": "low"
   * }
   */

  const reasoningEffort =
    normalizeReasoningEffort(
      incoming.reasoning_effort,
      modelConfig.reasoningEffort
    );


  const request = {

    model,

    messages,

    temperature,

    top_p: topP,

    max_tokens: maxTokens,

    stream
  };


  /* ==========================================================
     STANDARD OPTIONAL PARAMETERS
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
    "stream_options"
  ];


  for (
    const parameter of optionalParameters
  ) {

    if (
      incoming[parameter] !== undefined
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
     DEEPSEEK V4 FLASH / PRO
     ========================================================== */

  if (
    modelConfig.provider ===
    "deepseek-ai"
  ) {

    /*
     * NVIDIA's current DeepSeek V4 Flash example uses:
     *
     * chat_template_kwargs:
     * {
     *   thinking: true,
     *   reasoning_effort: "high"
     * }
     *
     * We use the same structure for both V4 models.
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
     * DO NOT send:
     *
     * reasoning_budget
     * reasoning_effort at the top level
     * reasoning_mode
     * nvext
     *
     * to these models.
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
    modelConfig.provider ===
    "moonshotai"
  ) {

    /*
     * Kimi K3 always reasons.
     *
     * NVIDIA documents:
     *
     *   low
     *   high
     *   max
     *
     * We therefore send the effort level directly.
     */

    request.chat_template_kwargs = {

      ...(
        request.chat_template_kwargs ||
        {}
      ),

      reasoning_effort:
        reasoningEffort
    };


    delete request.reasoning_budget;

    delete request.reasoning_effort;

    delete request.reasoning_mode;

    delete request.nvext;
  }


  /* ==========================================================
     NEMOTRON ULTRA
     ========================================================== */

  else if (
    modelConfig.provider ===
    "nvidia"
  ) {

    /*
     * Keep the Nemotron-specific behavior isolated.
     */

    request.reasoning_effort =
      reasoningEffort;


    /*
     * Nemotron accepts a numeric reasoning budget.
     *
     * This is the ONLY model in this configuration
     * where we intentionally send reasoning_budget.
     */

    const reasoningBudget =
      Math.min(
        32768,
        Math.max(
          1,
          Math.floor(
            numberOrDefault(
              incoming.reasoning_budget,
              modelConfig.reasoningBudget
            )
          )
        )
      );


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
  }


  return request;
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

  if (res.headersSent) {

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
   UPSTREAM ERROR PARSER
   ============================================================ */

function extractErrorMessage(data) {

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

  } catch (_) {

    return String(data);
  }
}


/* ============================================================
   STREAM PROCESSING
   ============================================================ */

function processSSEEvent(event) {

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

      /*
       * We deliberately do NOT delete
       * reasoning_content here.
       *
       * It can be required for multi-turn
       * reasoning/tool conversations,
       * particularly with Kimi K3.
       */

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
   READ UPSTREAM ERROR STREAM
   ============================================================ */

function readStream(stream) {

  return new Promise(
    (resolve) => {

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
        (chunk) => {

          output +=
            chunk.toString("utf8");


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
   ROOT
   ============================================================ */

app.get(
  "/",
  (req, res) => {

    res.json({

      status:
        "online",

      service:
        "JanitorAI -> NVIDIA NIM Proxy",

      default_model:
        resolveModel(
          DEFAULT_MODEL
        ),

      models:
        Object.keys(
          MODEL_CONFIG
        ),

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

    const model =
      resolveModel(
        DEFAULT_MODEL
      );

    const config =
      getModelConfig(model);


    res.json({

      ok:
        true,

      model,

      reasoning:
        config?.reasoning || false,

      reasoning_budget:
        config?.reasoningBudget || 0,

      reasoning_effort:
        config?.reasoningEffort || null
    });
  }
);


/* ============================================================
   MODEL LIST
   ============================================================ */

app.get(
  "/v1/models",
  (req, res) => {

    const data =
      Object.entries(
        MODEL_CONFIG
      ).map(
        ([id, config]) => ({

          id,

          object:
            "model",

          owned_by:
            config.provider,

          reasoning:
            config.reasoning,

          reasoning_effort:
            config.reasoningEffort,

          reasoning_budget:
            config.reasoningBudget
        })
      );


    res.json({

      object:
        "list",

      data
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
        typeof NIM_API_KEY !== "string"
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
        isPlainObject(req.body)
          ? req.body
          : {};


      /* ------------------------------------------------------
         MODEL
         ------------------------------------------------------ */

      const requestedModel =
        typeof incoming.model === "string" &&
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
         BUILD NIM REQUEST
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

      if (DEBUG_PROXY) {

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


      /* ------------------------------------------------------
         AXIOS
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
         CALL NVIDIA
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


        const message =
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
          "HTTP:",
          response.status
        );

        console.error(
          "Message:",
          message
        );

        console.error(
          "Body:",
          errorBody
        );

        console.error(
          "=========================================="
        );


        return sendError(
          res,
          response.status >= 500
            ? 502
            : response.status,
          message ||
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

      if (!stream) {

        return res
          .status(200)
          .json(response.data);
      }


      /* ------------------------------------------------------
         STREAMING HEADERS
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
            chunk.toString("utf8");


          const events =
            buffer.split(
              /\r?\n\r?\n/
            );


          buffer =
            events.pop() || "";


          for (
            const event of events
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


            if (!output) {
              continue;
            }


            try {

              res.write(output);

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

          if (ended) {
            return;
          }


          ended = true;


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
                res.write(output);
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

          if (ended) {
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
   SERVER
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
        "JanitorAI -> NVIDIA NIM Proxy"
      );

      console.log(
        `Port: ${PORT}`
      );

      console.log(
        `Default model: ${resolveModel(DEFAULT_MODEL)}`
      );

      console.log(
        `NVIDIA endpoint: ${NIM_API_BASE}`
      );

      console.log(
        `Timeout: ${NIM_TIMEOUT}ms`
      );

      console.log(
        "=========================================="
      );

      console.log(
        "Configured models:"
      );


      for (
        const [id, config]
        of Object.entries(MODEL_CONFIG)
      ) {

        console.log(
          `- ${id}`
        );

        console.log(
          `  reasoning budget: ${config.reasoningBudget}`
        );

        console.log(
          `  reasoning effort: ${config.reasoningEffort}`
        );

        console.log(
          `  max tokens: ${config.maxTokens}`
        );
      }


      console.log(
        "=========================================="
      );
    }
  );


/* ============================================================
   LONG-RUNNING REQUESTS
   ============================================================ */

server.timeout = 0;

server.requestTimeout = 0;

server.keepAliveTimeout =
  Math.max(
    65000,
    NIM_TIMEOUT + 5000
  );


/* ============================================================
   SHUTDOWN
   ============================================================ */

function shutdown(signal) {

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
