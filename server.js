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
    : 32768;

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
