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
  "nvidia/nemotron-3-ultra-550b-a55b";

/*
 * Reasoning modes:
 *
 * off
 * medium
 * high
 *
 * "on" is treated as "high".
 */
const REASONING_MODE =
  String(
    process.env.REASONING_MODE || "high"
  )
    .trim()
    .toLowerCase();

/*
 * You specifically requested 32768.
 *
 * NVIDIA currently documents:
 *   max_tokens: 1 - 32768
 */
const DEFAULT_MAX_TOKENS = 65536;

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

/*
 * Reasoning budget is separate from max_tokens.
 *
 * 16384 gives Ultra a substantial reasoning budget while
 * leaving room for the final answer inside a 32768-token
 * generation ceiling.
 */
const DEFAULT_REASONING_BUDGET =
  Number.isFinite(
    Number(process.env.DEFAULT_REASONING_BUDGET)
  )
    ? Number(process.env.DEFAULT_REASONING_BUDGET)
    : 16384;

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

const NIM_TIMEOUT =
  Number.isFinite(
    Number(process.env.NIM_TIMEOUT_MS)
  )
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const DEBUG_PROXY =
  String(
    process.env.DEBUG_PROXY || "false"
  ).toLowerCase() === "true";


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


function isNemotronUltra(model) {
  return (
    normalizeModel(model) ===
    "nvidia/nemotron-3-ultra-550b-a55b"
  );
}


function getReasoningMode() {
  if (
    REASONING_MODE === "off" ||
    REASONING_MODE === "none" ||
    REASONING_MODE === "false" ||
    REASONING_MODE === "0"
  ) {
    return "none";
  }

  if (
    REASONING_MODE === "medium" ||
    REASONING_MODE === "med"
  ) {
    return "medium";
  }

  return "high";
}


function clampMaxTokens(value) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_MAX_TOKENS
    );

  return Math.min(
    32768,
    Math.max(
      1,
      Math.floor(number)
    )
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
    Math.max(
      -1,
      Math.floor(number)
    )
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
      message: String(message)
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
   BUILD REQUEST
============================================================ */

function buildNimRequest(
  incoming,
  model,
  messages,
  stream
) {
  const request = {
    model: model,

    messages: messages,

    temperature:
      clampTemperature(
        incoming.temperature
      ),

    top_p:
      clampTopP(
        incoming.top_p
      ),

    /*
     * ALWAYS allow the full 32768 requested ceiling.
     */
    max_tokens:
      clampMaxTokens(
        incoming.max_tokens
      ),

    stream: stream
  };


  /* ----------------------------------------------------------
     STANDARD PENALTIES
  ---------------------------------------------------------- */

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


  /* ----------------------------------------------------------
     OPTIONAL OPENAI-COMPATIBLE PARAMETERS
  ---------------------------------------------------------- */

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


  /* ----------------------------------------------------------
     CHAT TEMPLATE KWARGS
  ---------------------------------------------------------- */

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    )
  ) {
    request.chat_template_kwargs = {
      ...incoming.chat_template_kwargs
    };
  }


  /* ----------------------------------------------------------
     NEMOTRON ULTRA
  ---------------------------------------------------------- */

  if (
    isNemotronUltra(model)
  ) {
    const reasoning =
      getReasoningMode();

    /*
     * NVIDIA's current Ultra API supports:
     *
     * none
     * medium
     * high
     *
     * through reasoning_effort.
     */
    request.reasoning_effort =
      reasoning;


    /*
     * Keep chat_template_kwargs compatible with
     * the model's documented interface.
     */
    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),

      enable_thinking:
        reasoning !== "none"
    };


    /*
     * Medium effort requires medium_effort=true.
     */
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
     * Explicit reasoning budget for high reasoning.
     *
     * Don't send it for "none".
     */
    if (
      reasoning === "high"
    ) {
      request.reasoning_budget =
        clampReasoningBudget(
          incoming.reasoning_budget
        );
    } else {
      delete request.reasoning_budget;
    }


    /*
     * NVIDIA recommends force_nonempty_content
     * when tools are used with reasoning.
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


    /*
     * Do NOT send nvext.max_thinking_tokens.
     *
     * That was the problematic part of the old
     * configuration for the hosted API.
     */
    delete request.nvext;

    /*
     * reasoning_effort is the clean hosted API
     * control for Ultra.
     */
    delete request.reasoning_mode;
  }


  /* ----------------------------------------------------------
     OTHER MODELS
  ---------------------------------------------------------- */

  else {
    /*
     * Do not contaminate unrelated models with
     * Nemotron-specific parameters.
     */
    delete request.reasoning_budget;
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
      typeof choice !==
        "object"
    ) {
      continue;
    }


    /* Streaming response */

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


    /* Non-streaming response */

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
          JSON.stringify(
            parsed
          )
      );
    } catch (_) {
      /*
       * Preserve non-JSON SSE data.
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
        if (finished) {
          return;
        }

        finished = true;

        resolve(output);
      };

      stream.on(
        "data",
        (chunk) => {
          output +=
            chunk.toString(
              "utf8"
            );

          /*
           * Don't allow an enormous upstream error
           * to consume memory.
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
   EXTRACT UPSTREAM ERROR
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
  } catch (_) {
    return String(data);
  }
}


/* ============================================================
   ROOT / HEALTH
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

      max_tokens:
        DEFAULT_MAX_TOKENS,

      reasoning_budget:
        DEFAULT_REASONING_BUDGET,

      nim_api_base:
        NIM_API_BASE,

      timeout_ms:
        NIM_TIMEOUT
    });
  }
);


app.get(
  "/health",
  (req, res) => {
    res.json({
      ok: true,

      model:
        DEFAULT_MODEL,

      reasoning:
        getReasoningMode(),

      max_tokens:
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

      const model =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim()
          ? incoming.model.trim()
          : DEFAULT_MODEL;


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
         REQUEST NVIDIA
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
         * If NVIDIA returns 500, report 502 from our
         * proxy. That clearly means "upstream failed".
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
         DATA
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

            if (!output) {
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
         END
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
           * Process any final partial SSE event.
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
       TOP-LEVEL ERROR
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
       * Connection reset
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
        `📝 Max tokens: ${DEFAULT_MAX_TOKENS}`
      );

      console.log(
        `💭 Reasoning budget: ${DEFAULT_REASONING_BUDGET}`
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
        "=========================================="
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
