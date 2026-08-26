const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

/* ============================================================
   CONFIG
============================================================ */

const PORT = Number(process.env.PORT) || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "nvidia/nemotron-3-ultra-550b-a55b";

const REASONING_MODE =
  String(process.env.REASONING_MODE || "on").toLowerCase();

const DEFAULT_TEMPERATURE =
  Number.isFinite(Number(process.env.DEFAULT_TEMPERATURE))
    ? Number(process.env.DEFAULT_TEMPERATURE)
    : 0.9;

const DEFAULT_TOP_P =
  Number.isFinite(Number(process.env.DEFAULT_TOP_P))
    ? Number(process.env.DEFAULT_TOP_P)
    : 0.95;

const DEFAULT_MAX_TOKENS =
  Number.isFinite(Number(process.env.DEFAULT_MAX_TOKENS))
    ? Number(process.env.DEFAULT_MAX_TOKENS)
    : 32768;const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { PassThrough } = require("stream");

const app = express();

/* ============================================================
   CONFIG
============================================================ */

const PORT =
  Number.isFinite(Number(process.env.PORT))
    ? Number(process.env.PORT)
    : 3000;

const NIM_API_BASE =
  String(
    process.env.NIM_API_BASE ||
      "https://integrate.api.nvidia.com/v1"
  ).replace(/\/+$/, "");

const NIM_API_KEY =
  process.env.NIM_API_KEY ||
  process.env.NVIDIA_API_KEY ||
  "";

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "nvidia/nemotron-3-ultra-550b-a55b";

/*
 * REASONING_MODE:
 *
 * off
 * medium
 * on / high
 *
 * For Nemotron Ultra:
 *   off    -> enable_thinking=false
 *   medium -> enable_thinking=true + medium_effort=true
 *   on     -> enable_thinking=true
 */
const REASONING_MODE =
  String(
    process.env.REASONING_MODE || "on"
  )
    .trim()
    .toLowerCase();

const DEFAULT_TEMPERATURE =
  finiteNumber(
    process.env.DEFAULT_TEMPERATURE
  )
    ? Number(process.env.DEFAULT_TEMPERATURE)
    : 1.0;

const DEFAULT_TOP_P =
  finiteNumber(
    process.env.DEFAULT_TOP_P
  )
    ? Number(process.env.DEFAULT_TOP_P)
    : 0.95;

/*
 * NVIDIA currently documents 16384 as the default
 * max_tokens for Nemotron Ultra and 32768 as the maximum.
 */
const DEFAULT_MAX_TOKENS =
  finiteNumber(
    process.env.DEFAULT_MAX_TOKENS
  )
    ? Number(process.env.DEFAULT_MAX_TOKENS)
    : 32786;

const DEFAULT_REPETITION_PENALTY =
  finiteNumber(
    process.env.DEFAULT_REPETITION_PENALTY
  )
    ? Number(process.env.DEFAULT_REPETITION_PENALTY)
    : 1.0;

const DEFAULT_FREQUENCY_PENALTY =
  finiteNumber(
    process.env.DEFAULT_FREQUENCY_PENALTY
  )
    ? Number(process.env.DEFAULT_FREQUENCY_PENALTY)
    : 0.0;

const DEFAULT_PRESENCE_PENALTY =
  finiteNumber(
    process.env.DEFAULT_PRESENCE_PENALTY
  )
    ? Number(process.env.DEFAULT_PRESENCE_PENALTY)
    : 0.0;

/*
 * This is used as NVIDIA's reasoning_budget for Ultra.
 *
 * It is intentionally separate from max_tokens.
 */
const DEFAULT_REASONING_BUDGET =
  finiteNumber(
    process.env.DEFAULT_REASONING_BUDGET
  )
    ? Number(process.env.DEFAULT_REASONING_BUDGET)
    : 8192;

const NIM_TIMEOUT =
  finiteNumber(
    process.env.NIM_TIMEOUT_MS
  )
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY || "false"
  ).toLowerCase() === "true";


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

function finiteNumber(value) {
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  ) ||
    (
      typeof value === "string" &&
      value.trim() !== "" &&
      Number.isFinite(Number(value))
    );
}


function numberOrDefault(value, fallback) {
  if (finiteNumber(value)) {
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


function isPlainObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}


function normalizeModel(model) {
  return String(model || "")
    .trim()
    .toLowerCase();
}


function isNemotronUltra(model) {
  return (
    normalizeModel(model) ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  );
}


function getReasoningMode() {
  if (
    REASONING_MODE === "off" ||
    REASONING_MODE === "false" ||
    REASONING_MODE === "0" ||
    REASONING_MODE === "none"
  ) {
    return "off";
  }

  if (
    REASONING_MODE === "medium" ||
    REASONING_MODE === "med"
  ) {
    return "medium";
  }

  return "high";
}


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

  const payload = {
    error: {
      message: String(message)
    }
  };

  if (
    details !== null &&
    details !== undefined
  ) {
    payload.error.details =
      details;
  }

  return res
    .status(status)
    .json(payload);
}


function safeJsonStringify(value) {
  try {
    return JSON.stringify(value);
  } catch (_) {
    return String(value);
  }
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
   REQUEST VALIDATION
============================================================ */

function clampMaxTokens(value) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_MAX_TOKENS
    );

  /*
   * NVIDIA documents 1..32768 for Ultra.
   *
   * Keeping this within the documented range prevents
   * accidental invalid requests from JanitorAI.
   */
  return Math.min(
    32768,
    Math.max(1, Math.floor(number))
  );
}


function clampReasoningBudget(value) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_REASONING_BUDGET
    );

  return Math.min(
    32768,
    Math.max(1, Math.floor(number))
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
    Math.max(0, number)
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
    Math.max(0, number)
  );
}


/* ============================================================
   MODEL-SPECIFIC REQUEST BUILDER
============================================================ */

function applyModelSpecificSettings(
  body,
  model,
  incoming
) {
  const lowerModel =
    normalizeModel(model);

  const reasoningMode =
    getReasoningMode();

  /*
   * ----------------------------------------------------------
   * NEMOTRON 3 ULTRA
   * ----------------------------------------------------------
   *
   * Hosted NVIDIA documentation currently exposes:
   *
   * chat_template_kwargs:
   *   enable_thinking
   *
   * and reasoning_budget for explicit reasoning budgets.
   *
   * Do NOT use nvext.max_thinking_tokens here.
   */

  if (
    lowerModel ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    const thinkingEnabled =
      reasoningMode !== "off";

    const existingTemplate =
      isPlainObject(
        incoming.chat_template_kwargs
      )
        ? incoming.chat_template_kwargs
        : {};

    body.chat_template_kwargs = {
      ...existingTemplate,
      enable_thinking:
        thinkingEnabled
    };

    /*
     * Medium effort is explicitly supported by the
     * current Ultra API documentation.
     */
    if (
      reasoningMode === "medium"
    ) {
      body.chat_template_kwargs.medium_effort =
        true;
    } else {
      delete body.chat_template_kwargs
        .medium_effort;
    }

    /*
     * Only send reasoning_budget when reasoning
     * is actually enabled.
     */
    if (thinkingEnabled) {
      body.reasoning_budget =
        clampReasoningBudget(
          incoming.reasoning_budget
        );
    } else {
      delete body.reasoning_budget;
    }

    /*
     * Ultra has a documented special requirement
     * when tools are being used.
     */
    if (
      Array.isArray(body.tools) &&
      body.tools.length > 0
    ) {
      body.chat_template_kwargs
        .force_nonempty_content = true;
    }

    /*
     * Do not send the old/ambiguous nvext reasoning
     * configuration to the hosted endpoint.
     */
    delete body.nvext;

    return;
  }


  /*
   * ----------------------------------------------------------
   * OTHER THINKING MODELS
   * ----------------------------------------------------------
   */

  const thinkingModels =
    new Set([
      "z-ai/glm-5.1",
      "z-ai/glm4.7",
      "google/gemma-4-31b-it",
      "qwen/qwen3.5-122b-a10b",
      "qwen/qwen3.5-397b-a17b",
      "qwen/qwen3-next-80b-a3b-thinking",
      "nvidia/nemotron-3-nano-30b-a3b"
    ]);

  if (
    thinkingModels.has(lowerModel)
  ) {
    const existingTemplate =
      isPlainObject(
        incoming.chat_template_kwargs
      )
        ? incoming.chat_template_kwargs
        : {};

    body.chat_template_kwargs = {
      ...existingTemplate,
      enable_thinking:
        reasoningMode !== "off"
    };

    delete body.nvext;

    return;
  }


  /*
   * ----------------------------------------------------------
   * DEEPSEEK
   * ----------------------------------------------------------
   */

  if (
    lowerModel ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
    if (
      reasoningMode === "off"
    ) {
      body.chat_template_kwargs = {
        ...(isPlainObject(
          incoming.chat_template_kwargs
        )
          ? incoming.chat_template_kwargs
          : {}),
        enable_thinking: false
      };

      delete body.reasoning_effort;
    } else {
      body.reasoning_effort =
        reasoningMode === "medium"
          ? "medium"
          : "max";
    }

    delete body.nvext;

    return;
  }


  /*
   * ----------------------------------------------------------
   * NEMOTRON 3 SUPER
   * ----------------------------------------------------------
   */

  if (
    lowerModel ===
    "nvidia/nemotron-3-super-120b-a12b"
  ) {
    body.reasoning_effort =
      reasoningMode === "off"
        ? "none"
        : reasoningMode === "medium"
          ? "medium"
          : "high";

    delete body.nvext;

    return;
  }


  /*
   * ----------------------------------------------------------
   * GENERIC MODELS
   * ----------------------------------------------------------
   *
   * Do not inject Ultra-specific parameters into
   * unrelated models.
   */

  delete body.nvext;
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
        incoming.max_tokens
      ),

    stream
  };


  /*
   * Penalties
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
   * Safe OpenAI-compatible parameters.
   *
   * We intentionally do not blindly forward every property
   * from JanitorAI.
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
    "parallel_tool_calls"
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
   * Preserve client chat template kwargs.
   * Model-specific logic may modify these afterward.
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
   * Some clients may explicitly provide reasoning_effort.
   * Preserve it initially; model-specific logic decides
   * whether it is appropriate.
   */

  if (
    incoming.reasoning_effort !==
    undefined
  ) {
    request.reasoning_effort =
      incoming.reasoning_effort;
  }


  /*
   * Apply model-specific behavior.
   */

  applyModelSpecificSettings(
    request,
    model,
    incoming
  );


  return request;
}


/* ============================================================
   REASONING STRIPPING
============================================================ */

function stripReasoningFromObject(
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
     * Streaming responses normally use delta.
     */
    if (
      choice.delta &&
      typeof choice.delta === "object"
    ) {
      delete choice.delta.reasoning_content;
      delete choice.delta.reasoning;
      delete choice.delta.thinking;
      delete choice.delta.reasoning_tokens;
    }

    /*
     * Non-streaming responses normally use message.
     */
    if (
      choice.message &&
      typeof choice.message === "object"
    ) {
      delete choice.message.reasoning_content;
      delete choice.message.reasoning;
      delete choice.message.thinking;
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
    /*
     * Preserve SSE comments and metadata.
     */
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

      stripReasoningFromObject(
        parsed
      );

      output.push(
        `data: ${JSON.stringify(parsed)}`
      );
    } catch (_) {
      /*
       * Never destroy an upstream SSE event simply
       * because it isn't JSON.
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
   STREAM READER
============================================================ */

function readStream(
  stream,
  maxBytes = 2 * 1024 * 1024
) {
  return new Promise(
    (resolve) => {
      let output = "";
      let finished = false;

      function done() {
        if (finished) {
          return;
        }

        finished = true;
        resolve(output);
      }

      stream.on(
        "data",
        (chunk) => {
          if (
            output.length >=
            maxBytes
          ) {
            return;
          }

          output +=
            chunk.toString("utf8");

          if (
            output.length >=
            maxBytes
          ) {
            output =
              output.slice(
                0,
                maxBytes
              );
          }
        }
      );

      stream.on(
        "end",
        done
      );

      stream.on(
        "close",
        done
      );

      stream.on(
        "error",
        done
      );
    }
  );
}


/* ============================================================
   UPSTREAM ERROR EXTRACTION
============================================================ */

async function getUpstreamErrorBody(
  response,
  stream
) {
  if (
    !response
  ) {
    return null;
  }

  if (
    stream &&
    response.data &&
    typeof response.data.on ===
      "function"
  ) {
    return readStream(
      response.data
    );
  }

  if (
    typeof response.data ===
    "string"
  ) {
    return response.data;
  }

  return safeJsonStringify(
    response.data
  );
}


function extractErrorMessage(
  body
) {
  if (
    !body
  ) {
    return null;
  }

  if (
    typeof body === "object"
  ) {
    return (
      body.error?.message ||
      body.message ||
      null
    );
  }

  try {
    const parsed =
      JSON.parse(body);

    return (
      parsed?.error?.message ||
      parsed?.message ||
      null
    );
  } catch (_) {
    return String(body);
  }
}


/* ============================================================
   HEALTH CHECK
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

      reasoning:
        getReasoningMode(),

      nim_api_base:
        NIM_API_BASE,

      defaults: {
        temperature:
          DEFAULT_TEMPERATURE,

        top_p:
          DEFAULT_TOP_P,

        max_tokens:
          DEFAULT_MAX_TOKENS,

        repetition_penalty:
          DEFAULT_REPETITION_PENALTY,

        frequency_penalty:
          DEFAULT_FREQUENCY_PENALTY,

        presence_penalty:
          DEFAULT_PRESENCE_PENALTY,

        reasoning_budget:
          DEFAULT_REASONING_BUDGET,

        timeout_ms:
          NIM_TIMEOUT
      }
    });
  }
);


/* ============================================================
   OPTIONAL MODEL HEALTH ENDPOINT
============================================================ */

app.get(
  "/health",
  (req, res) => {
    res.json({
      ok: true,
      model: DEFAULT_MODEL,
      reasoning:
        getReasoningMode()
    });
  }
);


/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post(
  "/v1/chat/completions",
  async (req, res) => {
    let response = null;
    let upstream = null;
    let clientDisconnected = false;

    try {
      /*
       * ------------------------------------------------------
       * API KEY
       * ------------------------------------------------------
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
       * ------------------------------------------------------
       * REQUEST BODY
       * ------------------------------------------------------
       */

      const incoming =
        isPlainObject(req.body)
          ? req.body
          : {};


      /*
       * ------------------------------------------------------
       * MODEL
       * ------------------------------------------------------
       */

      const model =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim() !== ""
          ? incoming.model.trim()
          : DEFAULT_MODEL;


      /*
       * ------------------------------------------------------
       * MESSAGES
       * ------------------------------------------------------
       */

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


      /*
       * ------------------------------------------------------
       * STREAM
       * ------------------------------------------------------
       */

      const stream =
        parseBoolean(
          incoming.stream,
          true
        );


      /*
       * ------------------------------------------------------
       * BUILD REQUEST
       * ------------------------------------------------------
       */

      const nimRequest =
        buildNimRequest(
          incoming,
          model,
          messages,
          stream
        );


      /*
       * ------------------------------------------------------
       * DEBUG
       * ------------------------------------------------------
       */

      if (DEBUG_PROXY) {
        /*
         * Never log the API key.
         */
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
       * ------------------------------------------------------
       * AXIOS CONFIG
       * ------------------------------------------------------
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

        /*
         * We handle upstream HTTP errors ourselves so
         * we can return the actual NVIDIA response body.
         */
        validateStatus:
          () => true
      };


      /*
       * ------------------------------------------------------
       * SEND TO NVIDIA
       * ------------------------------------------------------
       */

      response =
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
       * ------------------------------------------------------
       * UPSTREAM HTTP ERROR
       * ------------------------------------------------------
       */

      if (
        response.status < 200 ||
        response.status >= 300
      ) {
        const errorBody =
          await getUpstreamErrorBody(
            response,
            stream
          );

        const upstreamMessage =
          extractErrorMessage(
            errorBody
          );

        console.error(
          "========== NVIDIA ERROR =========="
        );

        console.error(
          "HTTP status:",
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
          "=================================="
        );


        /*
         * Return a useful gateway error to JanitorAI.
         *
         * 4xx from NVIDIA remains 4xx.
         *
         * 5xx from NVIDIA becomes 502 because our proxy
         * successfully contacted NVIDIA but NVIDIA failed.
         */
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

            upstream:
              errorBody
          }
        );
      }


      /*
       * ------------------------------------------------------
       * NON-STREAM RESPONSE
       * ------------------------------------------------------
       */

      if (!stream) {
        return res
          .status(response.status)
          .json(
            stripReasoningFromObject(
              response.data
            )
          );
      }


      /*
       * ------------------------------------------------------
       * STREAM RESPONSE HEADERS
       * ------------------------------------------------------
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

      /*
       * Useful for proxies/load balancers.
       */
      res.setHeader(
        "Transfer-Encoding",
        "chunked"
      );

      if (
        typeof res.flushHeaders ===
        "function"
      ) {
        res.flushHeaders();
      }


      /*
       * ------------------------------------------------------
       * UPSTREAM STREAM
       * ------------------------------------------------------
       */

      upstream =
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
       * ------------------------------------------------------
       * CLIENT DISCONNECT
       * ------------------------------------------------------
       */

      const abortUpstream =
        () => {
          if (
            clientDisconnected
          ) {
            return;
          }

          clientDisconnected =
            true;

          try {
            upstream.destroy();
          } catch (_) {}
        };

      req.on(
        "aborted",
        abortUpstream
      );

      res.on(
        "close",
        () => {
          /*
           * res.close is expected when the client goes away.
           * Do not destroy the stream after normal completion.
           */
          if (
            !res.writableEnded
          ) {
            abortUpstream();
          }
        }
      );


      /*
       * ------------------------------------------------------
       * SSE BUFFER
       * ------------------------------------------------------
       */

      let buffer = "";

      let streamEnded = false;

      /*
       * Prevent multiple drain handlers from piling up.
       */
      let waitingForDrain =
        false;


      /*
       * ------------------------------------------------------
       * UPSTREAM DATA
       * ------------------------------------------------------
       */

      upstream.on(
        "data",
        (chunk) => {
          if (
            clientDisconnected ||
            streamEnded
          ) {
            return;
          }

          buffer +=
            Buffer
              .from(chunk)
              .toString("utf8");

          /*
           * SSE events end in a blank line.
           */
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
              clientDisconnected ||
              streamEnded
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
              const canContinue =
                res.write(
                  output
                );

              if (
                !canContinue &&
                !waitingForDrain
              ) {
                waitingForDrain =
                  true;

                upstream.pause();

                res.once(
                  "drain",
                  () => {
                    waitingForDrain =
                      false;

                    if (
                      !clientDisconnected &&
                      !streamEnded
                    ) {
                      upstream.resume();
                    }
                  }
                );
              }
            } catch (error) {
              console.error(
                "Response write error:",
                error.message
              );

              abortUpstream();
            }
          }
        }
      );


      /*
       * ------------------------------------------------------
       * UPSTREAM END
       * ------------------------------------------------------
       */

      upstream.on(
        "end",
        () => {
          if (
            streamEnded
          ) {
            return;
          }

          streamEnded =
            true;

          /*
           * Process an incomplete final SSE event.
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


      /*
       * ------------------------------------------------------
       * UPSTREAM ERROR
       * ------------------------------------------------------
       */

      upstream.on(
        "error",
        (error) => {
          if (
            streamEnded
          ) {
            return;
          }

          streamEnded =
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
          } catch (_) {}
        }
      );

    } catch (error) {
      /*
       * ------------------------------------------------------
       * TOP-LEVEL ERROR
       * ------------------------------------------------------
       */

      console.error(
        "========== PROXY ERROR =========="
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
        "================================="
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
       * Axios timeout
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
       * Axios connection failure
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
   404 HANDLER
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
        `🤖 Model: ${DEFAULT_MODEL}`
      );

      console.log(
        `🧠 Reasoning: ${getReasoningMode()}`
      );

      console.log(
        `💭 Reasoning budget: ${DEFAULT_REASONING_BUDGET}`
      );

      console.log(
        `📝 Max output tokens: ${DEFAULT_MAX_TOKENS}`
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
        `🔗 NIM endpoint: ${NIM_API_BASE}`
      );

      console.log(
        "=========================================="
      );
    }
  );


/* ============================================================
   SERVER TIMEOUTS
============================================================ */

/*
 * Nemotron Ultra can legitimately take a long time when
 * reasoning is enabled.
 *
 * Keep Node's server timeout disabled here and let Axios's
 * upstream timeout control the NIM request.
 */
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

const DEFAULT_REPETITION_PENALTY =
  Number.isFinite(Number(process.env.DEFAULT_REPETITION_PENALTY))
    ? Number(process.env.DEFAULT_REPETITION_PENALTY)
    : 1.0;

const DEFAULT_FREQUENCY_PENALTY =
  Number.isFinite(Number(process.env.DEFAULT_FREQUENCY_PENALTY))
    ? Number(process.env.DEFAULT_FREQUENCY_PENALTY)
    : 0.0;

const DEFAULT_PRESENCE_PENALTY =
  Number.isFinite(Number(process.env.DEFAULT_PRESENCE_PENALTY))
    ? Number(process.env.DEFAULT_PRESENCE_PENALTY)
    : 0.0;

const DEFAULT_THINKING_TOKENS =
  Number.isFinite(Number(process.env.DEFAULT_THINKING_TOKENS))
    ? Number(process.env.DEFAULT_THINKING_TOKENS)
    : 8192;

const NIM_TIMEOUT =
  Number.isFinite(Number(process.env.NIM_TIMEOUT_MS))
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;


/* ============================================================
   EXPRESS
============================================================ */

app.use(cors());

app.use(
  express.json({
    limit: "100mb"
  })
);


/* ============================================================
   HELPERS
============================================================ */

/**
 * Returns true only for finite JavaScript numbers.
 */
function isNumber(value) {
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  );
}


/**
 * Converts a value to a number if possible.
 * Otherwise returns the supplied fallback.
 */
function numberOrDefault(value, fallback) {
  if (isNumber(value)) {
    return value;
  }

  if (
    typeof value === "string" &&
    value.trim() !== ""
  ) {
    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}


/**
 * Parses booleans safely.
 *
 * Handles:
 * true
 * false
 * "true"
 * "false"
 * 1
 * 0
 */
function parseBoolean(value, fallback) {
  if (value === undefined || value === null) {
    return fallback;
  }

  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return value !== 0;
  }

  if (typeof value === "string") {
    const normalized = value
      .trim()
      .toLowerCase();

    if (
      normalized === "true" ||
      normalized === "1" ||
      normalized === "yes"
    ) {
      return true;
    }

    if (
      normalized === "false" ||
      normalized === "0" ||
      normalized === "no"
    ) {
      return false;
    }
  }

  return fallback;
}


/**
 * Sends a consistent JSON error.
 */
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

  const payload = {
    error: {
      message: String(message)
    }
  };

  if (
    details !== null &&
    details !== undefined
  ) {
    payload.error.details = details;
  }

  return res
    .status(status)
    .json(payload);
}


/* ============================================================
   MESSAGE NORMALIZATION
============================================================ */

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.filter((message) => {
    if (
      !message ||
      typeof message !== "object"
    ) {
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
  });
}


/* ============================================================
   MODEL-SPECIFIC SETTINGS
============================================================ */

function addModelSpecificSettings(
  body,
  model
) {
  const lowerModel =
    String(model || "").toLowerCase();

  const thinkingEnabled =
    REASONING_MODE !== "off" &&
    REASONING_MODE !== "false" &&
    REASONING_MODE !== "0";


  /* ----------------------------------------------------------
     NEMOTRON 3 ULTRA
  ---------------------------------------------------------- */

  if (
    lowerModel ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: thinkingEnabled
    };

    if (thinkingEnabled) {
      body.nvext = {
        ...(body.nvext || {}),
        max_thinking_tokens:
          DEFAULT_THINKING_TOKENS
      };
    }

    return;
  }


  /* ----------------------------------------------------------
     OTHER THINKING MODELS
  ---------------------------------------------------------- */

  const thinkingModels = new Set([
    "z-ai/glm-5.1",
    "z-ai/glm4.7",
    "google/gemma-4-31b-it",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-next-80b-a3b-thinking",
    "nvidia/nemotron-3-nano-30b-a3b"
  ]);

  if (thinkingModels.has(lowerModel)) {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: thinkingEnabled
    };

    return;
  }


  /* ----------------------------------------------------------
     DEEPSEEK
  ---------------------------------------------------------- */

  if (
    lowerModel ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
    if (!thinkingEnabled) {
      body.chat_template_kwargs = {
        ...(body.chat_template_kwargs || {}),
        enable_thinking: false
      };
    } else {
      body.reasoning_effort = "max";
    }

    return;
  }


  /* ----------------------------------------------------------
     NEMOTRON 3 SUPER
  ---------------------------------------------------------- */

  if (
    lowerModel ===
    "nvidia/nemotron-3-super-120b-a12b"
  ) {
    body.reasoning_effort =
      thinkingEnabled
        ? "high"
        : "none";

    return;
  }
}


/* ============================================================
   REMOVE REASONING FROM STREAM
============================================================ */

function stripReasoning(parsed) {
  if (
    !parsed ||
    typeof parsed !== "object" ||
    !Array.isArray(parsed.choices)
  ) {
    return parsed;
  }

  for (const choice of parsed.choices) {
    if (
      !choice ||
      typeof choice !== "object" ||
      !choice.delta ||
      typeof choice.delta !== "object"
    ) {
      continue;
    }

    delete choice.delta.reasoning_content;
    delete choice.delta.reasoning;
    delete choice.delta.thinking;
  }

  return parsed;
}


/* ============================================================
   SSE EVENT PROCESSOR
============================================================ */

function processSSEEvent(event) {
  if (!event || !event.trim()) {
    return "";
  }

  const lines =
    event.split(/\r?\n/);

  const outputLines = [];

  for (const line of lines) {
    if (!line.startsWith("data:")) {
      outputLines.push(line);
      continue;
    }

    const data =
      line.slice(5).trim();

    if (!data) {
      outputLines.push(line);
      continue;
    }

    if (data === "[DONE]") {
      outputLines.push(
        "data: [DONE]"
      );

      continue;
    }

    try {
      const parsed =
        JSON.parse(data);

      stripReasoning(parsed);

      outputLines.push(
        `data: ${JSON.stringify(parsed)}`
      );
    } catch (_) {
      /*
       * If NVIDIA sends something that isn't JSON,
       * preserve it instead of destroying the stream.
       */

      outputLines.push(line);
    }
  }

  if (outputLines.length === 0) {
    return "";
  }

  return (
    outputLines.join("\n") +
    "\n\n"
  );
}


/* ============================================================
   HEALTH CHECK
============================================================ */

app.get("/", (req, res) => {
  res.json({
    status: "online",
    service: "JanitorAI → NVIDIA NIM Proxy",

    default_model:
      DEFAULT_MODEL,

    reasoning:
      REASONING_MODE,

    defaults: {
      temperature:
        DEFAULT_TEMPERATURE,

      top_p:
        DEFAULT_TOP_P,

      max_tokens:
        DEFAULT_MAX_TOKENS,

      repetition_penalty:
        DEFAULT_REPETITION_PENALTY,

      frequency_penalty:
        DEFAULT_FREQUENCY_PENALTY,

      presence_penalty:
        DEFAULT_PRESENCE_PENALTY,

      thinking_tokens:
        DEFAULT_THINKING_TOKENS
    }
  });
});


/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post(
  "/v1/chat/completions",
  async (req, res) => {
    try {

      /* --------------------------------------------------------
         API KEY
      -------------------------------------------------------- */

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


      /* --------------------------------------------------------
         REQUEST BODY
      -------------------------------------------------------- */

      const incoming =
        req.body &&
        typeof req.body === "object"
          ? req.body
          : {};


      /* --------------------------------------------------------
         MODEL
      -------------------------------------------------------- */

      const model =
        typeof incoming.model === "string" &&
        incoming.model.trim() !== ""
          ? incoming.model
          : DEFAULT_MODEL;


      /* --------------------------------------------------------
         MESSAGES
      -------------------------------------------------------- */

      const messages =
        normalizeMessages(
          incoming.messages
        );

      if (messages.length === 0) {
        return sendError(
          res,
          400,
          "No valid messages were supplied."
        );
      }


      /* --------------------------------------------------------
         STREAM
      -------------------------------------------------------- */

      const stream =
        parseBoolean(
          incoming.stream,
          true
        );


      /* --------------------------------------------------------
         BASE NIM REQUEST
      -------------------------------------------------------- */

      const nimRequest = {
        model,
        messages,

        temperature:
          numberOrDefault(
            incoming.temperature,
            DEFAULT_TEMPERATURE
          ),

        top_p:
          numberOrDefault(
            incoming.top_p,
            DEFAULT_TOP_P
          ),

        max_tokens:
          numberOrDefault(
            incoming.max_tokens,
            DEFAULT_MAX_TOKENS
          ),

        stream
      };


      /* --------------------------------------------------------
         PENALTIES
      -------------------------------------------------------- */

      if (
        incoming.repetition_penalty !==
        undefined
      ) {
        nimRequest.repetition_penalty =
          numberOrDefault(
            incoming.repetition_penalty,
            DEFAULT_REPETITION_PENALTY
          );
      }

      if (
        incoming.frequency_penalty !==
        undefined
      ) {
        nimRequest.frequency_penalty =
          numberOrDefault(
            incoming.frequency_penalty,
            DEFAULT_FREQUENCY_PENALTY
          );
      }

      if (
        incoming.presence_penalty !==
        undefined
      ) {
        nimRequest.presence_penalty =
          numberOrDefault(
            incoming.presence_penalty,
            DEFAULT_PRESENCE_PENALTY
          );
      }


      /* --------------------------------------------------------
         OPTIONAL PARAMETERS
      -------------------------------------------------------- */

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
        "parallel_tool_calls"
      ];

      for (
        const parameter
        of optionalParameters
      ) {
        if (
          incoming[parameter] !==
          undefined
        ) {
          nimRequest[parameter] =
            incoming[parameter];
        }
      }


      /* --------------------------------------------------------
         CHAT TEMPLATE KWARGS
      -------------------------------------------------------- */

      if (
        incoming.chat_template_kwargs &&
        typeof incoming.chat_template_kwargs ===
          "object" &&
        !Array.isArray(
          incoming.chat_template_kwargs
        )
      ) {
        nimRequest.chat_template_kwargs = {
          ...incoming.chat_template_kwargs
        };
      }


      /* --------------------------------------------------------
         NVEXT
      -------------------------------------------------------- */

      if (
        incoming.nvext &&
        typeof incoming.nvext === "object" &&
        !Array.isArray(incoming.nvext)
      ) {
        nimRequest.nvext = {
          ...incoming.nvext
        };
      }


      /* --------------------------------------------------------
         MODEL-SPECIFIC SETTINGS
      -------------------------------------------------------- */

      addModelSpecificSettings(
        nimRequest,
        model
      );


      /* --------------------------------------------------------
         DEBUG
      -------------------------------------------------------- */

      if (
        process.env.DEBUG_PROXY === "true"
      ) {
        console.log(
          "NIM request settings:",
          {
            model:
              nimRequest.model,

            temperature:
              nimRequest.temperature,

            top_p:
              nimRequest.top_p,

            max_tokens:
              nimRequest.max_tokens,

            repetition_penalty:
              nimRequest.repetition_penalty,

            frequency_penalty:
              nimRequest.frequency_penalty,

            presence_penalty:
              nimRequest.presence_penalty,

            stream:
              nimRequest.stream,

            chat_template_kwargs:
              nimRequest.chat_template_kwargs,

            nvext:
              nimRequest.nvext
          }
        );
      }


      /* --------------------------------------------------------
         AXIOS CONFIG
      -------------------------------------------------------- */

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


      /* --------------------------------------------------------
         SEND REQUEST TO NVIDIA
      -------------------------------------------------------- */

      const response =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          stream
            ? {
                ...axiosConfig,
                responseType: "stream"
              }
            : axiosConfig
        );


      /* --------------------------------------------------------
         NVIDIA ERROR
      -------------------------------------------------------- */

      if (
        response.status < 200 ||
        response.status >= 300
      ) {

        let errorBody = "";

        if (
          stream &&
          response.data
        ) {
          errorBody =
            await readStream(
              response.data
            );
        } else {
          errorBody =
            typeof response.data ===
            "string"
              ? response.data
              : JSON.stringify(
                  response.data
                );
        }

        console.error(
          `NVIDIA returned ${response.status}:`,
          errorBody
        );

        return sendError(
          res,
          response.status,
          `NVIDIA NIM returned HTTP ${response.status}.`,
          errorBody
        );
      }


      /* --------------------------------------------------------
         NON-STREAMING RESPONSE
      -------------------------------------------------------- */

      if (!stream) {
        return res
          .status(response.status)
          .json(response.data);
      }


      /* --------------------------------------------------------
         STREAM HEADERS
      -------------------------------------------------------- */

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


      /* --------------------------------------------------------
         UPSTREAM STREAM
      -------------------------------------------------------- */

      const upstream =
        response.data;

      let buffer = "";
      let clientDisconnected = false;

      req.on(
        "close",
        () => {
          clientDisconnected = true;

          try {
            upstream.destroy();
          } catch (_) {}
        }
      );


      /* --------------------------------------------------------
         STREAM DATA
      -------------------------------------------------------- */

      upstream.on(
        "data",
        (chunk) => {
          if (clientDisconnected) {
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
            const event
            of events
          ) {
            const output =
              processSSEEvent(event);

            if (!output) {
              continue;
            }

            const canContinue =
              res.write(output);

            if (!canContinue) {
              upstream.pause();

              res.once(
                "drain",
                () => {
                  if (
                    !clientDisconnected
                  ) {
                    upstream.resume();
                  }
                }
              );
            }
          }
        }
      );


      /* --------------------------------------------------------
         STREAM END
      -------------------------------------------------------- */

      upstream.on(
        "end",
        () => {
          if (clientDisconnected) {
            return;
          }

          /*
           * NVIDIA normally terminates SSE events with
           * a blank line, but process anything remaining
           * just in case.
           */

          if (buffer.trim()) {
            const output =
              processSSEEvent(buffer);

            if (output) {
              try {
                res.write(output);
              } catch (_) {}
            }
          }

          try {
            res.end();
          } catch (_) {}
        }
      );


      /* --------------------------------------------------------
         STREAM ERROR
      -------------------------------------------------------- */

      upstream.on(
        "error",
        (error) => {
          console.error(
            "NVIDIA stream error:",
            error.message
          );

          if (!res.headersSent) {
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

    } catch (error) {

      console.error(
        "Proxy error:",
        error.response?.data ||
          error.message ||
          error
      );

      if (res.headersSent) {
        try {
          res.end();
        } catch (_) {}

        return;
      }

      return sendError(
        res,
        error.response?.status || 500,
        error.message ||
          "Proxy request failed.",
        error.response?.data || null
      );
    }
  }
);


/* ============================================================
   READ STREAM HELPER
============================================================ */

function readStream(stream) {
  return new Promise(
    (resolve) => {
      let output = "";

      stream.on(
        "data",
        (chunk) => {
          output +=
            chunk.toString();
        }
      );

      stream.on(
        "end",
        () => {
          resolve(output);
        }
      );

      stream.on(
        "error",
        () => {
          resolve(output);
        }
      );
    }
  );
}


/* ============================================================
   START SERVER
============================================================ */

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
      `🤖 Model: ${DEFAULT_MODEL}`
    );

    console.log(
      `🧠 Reasoning: ${REASONING_MODE}`
    );

    console.log(
      `💭 Thinking tokens: ${DEFAULT_THINKING_TOKENS}`
    );

    console.log(
      `📝 Max output tokens: ${DEFAULT_MAX_TOKENS}`
    );

    console.log(
      `⏱️ Timeout: ${NIM_TIMEOUT}ms`
    );

    console.log(
      "=========================================="
    );
  }
);
