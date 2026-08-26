const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

// ============================================================
// DEFAULT SETTINGS
// ============================================================
//
// These are ONLY fallbacks.
//
// If JanitorAI sends a parameter, we preserve it.
// This means you can control temperature, top_p, penalties,
// max tokens, etc. directly from JanitorAI.
//
// ============================================================

const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL ||
  "nvidia/nemotron-3-ultra-550b-a55b";

// ------------------------------------------------------------
// Sampling
// ------------------------------------------------------------

const DEFAULT_TEMPERATURE = Number(
  process.env.DEFAULT_TEMPERATURE || 0.9
);

const DEFAULT_TOP_P = Number(
  process.env.DEFAULT_TOP_P || 0.95
);

// ------------------------------------------------------------
// Output length
// ------------------------------------------------------------
//
// 32768 is a much more sensible ceiling than 384k.
//
// This is maximum GENERATED tokens, not context size.
//
// NVIDIA's own Ultra configurations use 16384 and 32768
// for long-form generation.
//
// ------------------------------------------------------------

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 32768
);

// ------------------------------------------------------------
// Penalties
// ------------------------------------------------------------
//
// These are defaults only.
//
// JanitorAI's values override them if supplied.
//
// repetition_penalty:
//   1.0 = neutral
//
// frequency_penalty:
//   0.0 = neutral
//
// presence_penalty:
//   0.0 = neutral
//
// For RP, I recommend starting neutral and tuning from
// JanitorAI rather than forcing aggressive penalties.
//
// ------------------------------------------------------------

const DEFAULT_REPETITION_PENALTY = Number(
  process.env.DEFAULT_REPETITION_PENALTY || 1.0
);

const DEFAULT_FREQUENCY_PENALTY = Number(
  process.env.DEFAULT_FREQUENCY_PENALTY || 0.0
);

const DEFAULT_PRESENCE_PENALTY = Number(
  process.env.DEFAULT_PRESENCE_PENALTY || 0.0
);

// ============================================================
// REASONING
// ============================================================

const REASONING_MODE =
  process.env.REASONING_MODE || "on";

// ------------------------------------------------------------
// Nemotron Ultra thinking budget
// ------------------------------------------------------------
//
// This controls the maximum thinking tokens separately from
// the final response.
//
// 8192 is a good starting point for RP.
//
// Increase this if you want more difficult reasoning,
// continuity checking, planning, etc.
//
// Decrease it if responses become unnecessarily slow.
//
// ------------------------------------------------------------

const DEFAULT_THINKING_TOKENS = Number(
  process.env.DEFAULT_THINKING_TOKENS || 8192
);

// ============================================================
// EXPRESS
// ============================================================

app.use(cors());

app.use(
  express.json({
    limit: "100mb"
  })
);

// ============================================================
// HEALTH CHECK
// ============================================================

app.get("/", (req, res) => {
  res.json({
    status: "online",
    service: "JanitorAI → NVIDIA NIM Proxy",

    default_model: DEFAULT_MODEL,

    reasoning: REASONING_MODE,

    defaults: {
      temperature: DEFAULT_TEMPERATURE,
      top_p: DEFAULT_TOP_P,
      max_tokens: DEFAULT_MAX_TOKENS,
      repetition_penalty: DEFAULT_REPETITION_PENALTY,
      frequency_penalty: DEFAULT_FREQUENCY_PENALTY,
      presence_penalty: DEFAULT_PRESENCE_PENALTY,
      thinking_tokens: DEFAULT_THINKING_TOKENS
    }
  });
});

// ============================================================
// MODEL-SPECIFIC SETTINGS
// ============================================================

function addModelSpecificSettings(body, model) {
  const lower = String(model).toLowerCase();

  // ==========================================================
  // NEMOTRON 3 ULTRA
  // ==========================================================

  if (
    lower === "nvidia/nemotron-3-ultra-550b-a55b"
  ) {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),

      enable_thinking:
        REASONING_MODE !== "off"
    };

    // --------------------------------------------------------
    // Thinking budget
    // --------------------------------------------------------
    //
    // NVIDIA's current Ultra documentation uses:
    //
    // nvext.max_thinking_tokens
    //
    // rather than relying on reasoning_effort.
    //
    // --------------------------------------------------------

    if (REASONING_MODE !== "off") {
      body.nvext = {
        ...(body.nvext || {}),

        max_thinking_tokens:
          DEFAULT_THINKING_TOKENS
      };
    }

    return;
  }

  // ==========================================================
  // GLM / OTHER THINKING MODELS
  // ==========================================================

  if (
    lower === "z-ai/glm-5.1" ||
    lower === "z-ai/glm4.7" ||
    lower === "google/gemma-4-31b-it" ||
    lower === "qwen/qwen3.5-122b-a10b" ||
    lower === "qwen/qwen3.5-397b-a17b" ||
    lower === "qwen/qwen3-next-80b-a3b-thinking" ||
    lower === "nvidia/nemotron-3-nano-30b-a3b"
  ) {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),

      enable_thinking:
        REASONING_MODE !== "off"
    };

    return;
  }

  // ==========================================================
  // DEEPSEEK
  // ==========================================================

  if (
    lower ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
    if (REASONING_MODE === "off") {
      body.chat_template_kwargs = {
        ...(body.chat_template_kwargs || {}),
        enable_thinking: false
      };
    } else {
      body.reasoning_effort = "max";
    }

    return;
  }

  // ==========================================================
  // NEMOTRON 3 SUPER
  // ==========================================================

  if (
    lower ===
    "nvidia/nemotron-3-super-120b-a12b"
  ) {
    body.reasoning_effort =
      REASONING_MODE === "off"
        ? "none"
        : "high";

    return;
  }

  // ==========================================================
  // UNKNOWN MODEL
  // ==========================================================
  //
  // Do nothing.
  //
  // This prevents us from guessing how an unknown model
  // handles reasoning.
  //
}

// ============================================================
// NORMALIZE / PRESERVE MESSAGES
// ============================================================
//
// IMPORTANT:
//
// We no longer rebuild messages from scratch.
//
// This preserves additional message fields JanitorAI may send.
//
// The previous version could silently discard useful fields.
//
// ============================================================

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.filter((message) => {
    if (!message) return false;
    if (!message.role) return false;

    if (
      message.content === undefined ||
      message.content === null
    ) {
      return false;
    }

    return true;
  });
}

// ============================================================
// NUMERIC PARAMETER HELPER
// ============================================================

function getNumberOrDefault(
  value,
  fallback
) {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value
    : fallback;
}

// ============================================================
// ERROR HELPER
// ============================================================

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

  res.status(status).json({
    error: {
      message,

      ...(details
        ? { details }
        : {})
    }
  });
}

// ============================================================
// STRIP REASONING FROM A STREAMING CHUNK
// ============================================================
//
// The model can think internally.
//
// JanitorAI should only receive the actual response.
//
// We modify JSON deltas rather than regex-filtering raw text.
//
// ============================================================

function stripReasoning(parsed) {
  if (
    !parsed ||
    !Array.isArray(parsed.choices)
  ) {
    return parsed;
  }

  for (const choice of parsed.choices) {
    if (!choice || !choice.delta) {
      continue;
    }

    const delta = choice.delta;

    // Common reasoning field.
    delete delta.reasoning_content;

    // Alternate reasoning field.
    delete delta.reasoning;

    // Some implementations can expose thinking
    // under additional fields.
    delete delta.thinking;
  }

  return parsed;
}

// ============================================================
// MAIN CHAT COMPLETIONS ROUTE
// ============================================================

app.post(
  "/v1/chat/completions",
  async (req, res) => {
    try {
      // ======================================================
      // API KEY
      // ======================================================

      if (!NIM_API_KEY) {
        return sendError(
          res,
          500,
          "NIM_API_KEY is not configured on Render."
        );
      }

      const incoming =
        req.body || {};

      // ======================================================
      // MODEL
      // ======================================================

      const model =
        incoming.model ||
        DEFAULT_MODEL;

      // ======================================================
      // MESSAGES
      // ======================================================

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

      // ======================================================
      // STREAM
      // ======================================================
      //
      // Don't use Boolean(value).
      //
      // Boolean("false") === true in JavaScript.
      //
      // ======================================================

      const stream =
        incoming.stream === undefined
          ? true
          : incoming.stream === true;

      // ======================================================
      // GENERATION PARAMETERS
      // ======================================================

      const nimRequest = {
        model,
        messages,

        temperature:
          getNumberOrDefault(
            incoming.temperature,
            DEFAULT_TEMPERATURE
          ),

        top_p:
          getNumberOrDefault(
            incoming.top_p,
            DEFAULT_TOP_P
          ),

        max_tokens:
          getNumberOrDefault(
            incoming.max_tokens,
            DEFAULT_MAX_TOKENS
          ),

        // ----------------------------------------------------
        // IMPORTANT:
        //
        // JanitorAI's values are forwarded if supplied.
        //
        // ----------------------------------------------------

       // Only forward penalties if JanitorAI actually supplied them.

if (typeof incoming.repetition_penalty === "number") {
  nimRequest.repetition_penalty =
    incoming.repetition_penalty;
}

if (typeof incoming.frequency_penalty === "number") {
  nimRequest.frequency_penalty =
    incoming.frequency_penalty;
}

if (typeof incoming.presence_penalty === "number") {
  nimRequest.presence_penalty =
    incoming.presence_penalty;
}

        stream
      };

      // ======================================================
      // OPTIONAL PARAMETERS
      // ======================================================

      if (
        incoming.stop !== undefined
      ) {
        nimRequest.stop =
          incoming.stop;
      }

      if (
        incoming.seed !== undefined
      ) {
        nimRequest.seed =
          incoming.seed;
      }

      if (
        incoming.tools !== undefined
      ) {
        nimRequest.tools =
          incoming.tools;
      }

      if (
        incoming.tool_choice !== undefined
      ) {
        nimRequest.tool_choice =
          incoming.tool_choice;
      }

      if (
        incoming.response_format !== undefined
      ) {
        nimRequest.response_format =
          incoming.response_format;
      }

      if (
        incoming.logprobs !== undefined
      ) {
        nimRequest.logprobs =
          incoming.logprobs;
      }

      if (
        incoming.top_k !== undefined
      ) {
        nimRequest.top_k =
          incoming.top_k;
      }

      if (
        incoming.min_tokens !== undefined
      ) {
        nimRequest.min_tokens =
          incoming.min_tokens;
      }

      if (
        incoming.ignore_eos !== undefined
      ) {
        nimRequest.ignore_eos =
          incoming.ignore_eos;
      }

      if (
        incoming.logit_bias !== undefined
      ) {
        nimRequest.logit_bias =
          incoming.logit_bias;
      }

      if (
        incoming.user !== undefined
      ) {
        nimRequest.user =
          incoming.user;
      }

      if (
        incoming.parallel_tool_calls !== undefined
      ) {
        nimRequest.parallel_tool_calls =
          incoming.parallel_tool_calls;
      }

      // ======================================================
      // CLIENT CHAT TEMPLATE SETTINGS
      // ======================================================

      if (
        incoming.chat_template_kwargs &&
        typeof incoming.chat_template_kwargs ===
          "object"
      ) {
        nimRequest.chat_template_kwargs = {
          ...incoming.chat_template_kwargs
        };
      }

      // ======================================================
      // CLIENT nvext SETTINGS
      // ======================================================
      //
      // Preserve any Janitor/client-supplied nvext values.
      //
      // Our Ultra thinking budget is applied below.
      //
      // ======================================================

      if (
        incoming.nvext &&
        typeof incoming.nvext === "object"
      ) {
        nimRequest.nvext = {
          ...incoming.nvext
        };
      }

      // ======================================================
      // MODEL-SPECIFIC SETTINGS
      // ======================================================

      addModelSpecificSettings(
        nimRequest,
        model
      );

      // ======================================================
      // OPTIONAL DEBUG LOGGING
      // ======================================================
      //
      // Set DEBUG_PROXY=true on Render if you want to see
      // exactly what is being sent to NVIDIA.
      //
      // We intentionally DO NOT log messages because your
      // Janitor conversation may contain private content.
      //
      // ======================================================

      if (
        process.env.DEBUG_PROXY === "true"
      ) {
        console.log(
          "NIM request settings:",
          {
            model: nimRequest.model,
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

      // ======================================================
      // NVIDIA REQUEST CONFIG
      // ======================================================

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

        // ----------------------------------------------------
        // Long RP generations can legitimately take a while.
        //
        // 15 minutes is safer than infinite.
        // ----------------------------------------------------

        timeout:
          Number(
            process.env.NIM_TIMEOUT_MS ||
            900000
          ),

        validateStatus:
          () => true
      };

      // ======================================================
      // SEND TO NVIDIA
      // ======================================================

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

      // ======================================================
      // NVIDIA ERROR
      // ======================================================

      if (
        response.status < 200 ||
        response.status >= 300
      ) {
        let errorBody = "";

        if (
          stream &&
          response.data
        ) {
          await new Promise(
            (resolve) => {
              response.data.on(
                "data",
                (chunk) => {
                  errorBody +=
                    chunk.toString();
                }
              );

              response.data.on(
                "end",
                resolve
              );

              response.data.on(
                "error",
                resolve
              );
            }
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

      // ======================================================
      // NON-STREAMING
      // ======================================================

      if (!stream) {
        return res
          .status(response.status)
          .json(response.data);
      }

      // ======================================================
      // STREAMING HEADERS
      // ======================================================

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

      // ======================================================
      // UPSTREAM STREAM
      // ======================================================

      const upstream =
        response.data;

      let clientDisconnected =
        false;

      req.on(
        "close",
        () => {
          clientDisconnected =
            true;

          try {
            upstream.destroy();
          } catch (_) {}
        }
      );

      // ======================================================
      // SSE PARSER
      // ======================================================
      //
      // NVIDIA uses SSE.
      //
      // We process complete events and preserve normal
      // content while removing reasoning fields.
      //
      // ======================================================

      let buffer = "";

      upstream.on(
        "data",
        (chunk) => {
          if (
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
              !event.trim()
            ) {
              continue;
            }

            const lines =
              event.split(
                /\r?\n/
              );

            const outputLines = [];

            for (
              const line of lines
            ) {
              if (
                !line.startsWith(
                  "data:"
                )
              ) {
                // Preserve other SSE fields.
                outputLines.push(
                  line
                );

                continue;
              }

              const data =
                line
                  .slice(5)
                  .trim();

              if (!data) {
                outputLines.push(
                  line
                );

                continue;
              }

              // ------------------------------------------------
              // DONE
              // ------------------------------------------------

              if (
                data === "[DONE]"
              ) {
                outputLines.push(
                  "data: [DONE]"
                );

                continue;
              }

              // ------------------------------------------------
              // JSON DATA
              // ------------------------------------------------

              try {
                const parsed =
                  JSON.parse(data);

                stripReasoning(
                  parsed
                );

                outputLines.push(
                  `data: ${JSON.stringify(parsed)}`
                );
              } catch (_) {
                // Preserve unexpected non-JSON data.
                outputLines.push(
                  line
                );
              }
            }

            if (
              outputLines.length > 0
            ) {
              const output =
                outputLines.join(
                  "\n"
                ) +
                "\n\n";

              // ------------------------------------------------
              // Backpressure handling
              // ------------------------------------------------

              const canContinue =
                res.write(output);

              if (
                !canContinue
              ) {
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
        }
      );

      // ======================================================
      // UPSTREAM END
      // ======================================================

      upstream.on(
        "end",
        () => {
          if (
            clientDisconnected
          ) {
            return;
          }

          // --------------------------------------------------
          // Process any remaining buffered SSE data.
          // --------------------------------------------------

          if (
            buffer.trim()
          ) {
            const lines =
              buffer.split(
                /\r?\n/
              );

            const outputLines =
              [];

            for (
              const line of lines
            ) {
              if (
                !line.startsWith(
                  "data:"
                )
              ) {
                outputLines.push(
                  line
                );

                continue;
              }

              const data =
                line
                  .slice(5)
                  .trim();

              if (!data) {
                continue;
              }

              if (
                data === "[DONE]"
              ) {
                outputLines.push(
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

                outputLines.push(
                  `data: ${JSON.stringify(parsed)}`
                );
              } catch (_) {
                outputLines.push(
                  line
                );
              }
            }

            if (
              outputLines.length > 0
            ) {
              res.write(
                outputLines.join(
                  "\n"
                ) +
                  "\n\n"
              );
            }
          }

          res.end();
        }
      );

      // ======================================================
      // UPSTREAM ERROR
      // ======================================================

      upstream.on(
        "error",
        (error) => {
          console.error(
            "NVIDIA stream error:",
            error.message
          );

          if (
            !res.headersSent
          ) {
            sendError(
              res,
              502,
              "NVIDIA streaming connection failed.",
              error.message
            );
          } else {
            try {
              res.end();
            } catch (_) {}
          }
        }
      );
    } catch (error) {
      console.error(
        "Proxy error:",
        error.response?.data ||
          error.message
      );

      if (
        res.headersSent
      ) {
        try {
          res.end();
        } catch (_) {}

        return;
      }

      sendError(
        res,
        error.response?.status ||
          500,

        error.message ||
          "Proxy request failed.",

        error.response?.data ||
          null
      );
    }
  }
);

// ============================================================
// START SERVER
// ============================================================

app.listen(
  PORT,
  "0.0.0.0",
  () => {
    console.log(
      `🚀 JanitorAI → NVIDIA NIM proxy running on port ${PORT}`
    );

    console.log(
      `🤖 Default model: ${DEFAULT_MODEL}`
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
  }
);
