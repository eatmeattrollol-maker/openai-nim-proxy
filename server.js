const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY = process.env.NIM_API_KEY;

// ============================================================
// DEFAULT SETTINGS
// ============================================================

// Your default model.
// You can change this in Render without touching this file.
const DEFAULT_MODEL =
  process.env.DEFAULT_MODEL || "z-ai/glm-5.2";

// Default generation settings.
// JanitorAI can override these when it sends them.
const DEFAULT_TEMPERATURE = Number(
  process.env.DEFAULT_TEMPERATURE || 1.0
);

const DEFAULT_TOP_P = Number(
  process.env.DEFAULT_TOP_P || 0.95
);

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 16384
);

// Reasoning behavior:
//
// "on"  = use reasoning when supported
// "off" = disable reasoning when supported
//
// For your use case, keep this ON.
const REASONING_MODE =
  process.env.REASONING_MODE || "on";

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
// BASIC HEALTH CHECK
// ============================================================

app.get("/", (req, res) => {
  res.json({
    status: "online",
    service: "JanitorAI → NVIDIA NIM Proxy",
    default_model: DEFAULT_MODEL,
    reasoning: REASONING_MODE
  });
});

// ============================================================
// MODEL SETTINGS
// ============================================================
//
// Different NIM models expose reasoning differently.
//
// We DO NOT inject fake prompts into the conversation.
//
// Instead, when the model has a known NIM reasoning control,
// we send the appropriate API parameter.
//
// ============================================================

function addModelSpecificSettings(body, model) {
  const lower = model.toLowerCase();

  // ----------------------------------------------------------
  // GLM-5.2
  // ----------------------------------------------------------
  //
  // NVIDIA's GLM-5.2 endpoint supports reasoning traces.
  //
  // The current NIM implementation accepts the OpenAI-style
  // reasoning_effort control.
  //
  // "max" = deepest reasoning
  // "high" = high reasoning
  //
  // We use max by default because your goal is maximum
  // narrative/world reasoning rather than minimum latency.
  //
  if (lower === "z-ai/glm-5.2") {
    if (REASONING_MODE === "off") {
      body.chat_template_kwargs = {
        ...(body.chat_template_kwargs || {}),
        enable_thinking: false
      };
    } else {
      body.reasoning_effort = "medium";
    }

    return;
  }

  // ----------------------------------------------------------
  // GLM-5.1
  // ----------------------------------------------------------

  if (lower === "z-ai/glm-5.1") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off",
      clear_thinking: false
    };

    return;
  }

  // ----------------------------------------------------------
  // GLM-4.7
  // ----------------------------------------------------------

  if (lower === "z-ai/glm4.7") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off",
      clear_thinking: false
    };

    return;
  }

  // ----------------------------------------------------------
  // GEMMA 4 31B
  // ----------------------------------------------------------

  if (lower === "google/gemma-4-31b-it") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off"
    };

    return;
  }

  // ----------------------------------------------------------
  // QWEN 3.5 122B
  // ----------------------------------------------------------

  if (lower === "qwen/qwen3.5-122b-a10b") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off"
    };

    return;
  }

  // ----------------------------------------------------------
  // QWEN 3.5 397B
  // ----------------------------------------------------------

  if (lower === "qwen/qwen3.5-397b-a17b") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off"
    };

    return;
  }

  // ----------------------------------------------------------
  // QWEN 3 NEXT THINKING
  // ----------------------------------------------------------

  if (lower === "qwen/qwen3-next-80b-a3b-thinking") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off"
    };

    return;
  }

  // ----------------------------------------------------------
  // NEMOTRON 3 ULTRA
  // ----------------------------------------------------------
  //
  // NVIDIA documents:
  //
  // none   = no reasoning
  // medium = efficient reasoning
  // high   = full reasoning
  //
  // We want maximum reasoning.
  //

  if (lower === "nvidia/nemotron-3-ultra-550b-a55b") {
    body.reasoning_effort =
      REASONING_MODE === "off" ? "none" : "high";

    return;
  }

  // ----------------------------------------------------------
  // NEMOTRON 3 SUPER
  // ----------------------------------------------------------
  //
  // NVIDIA documents:
  //
  // none / low / high
  //

  if (lower === "nvidia/nemotron-3-super-120b-a12b") {
    body.reasoning_effort =
      REASONING_MODE === "off" ? "none" : "high";

    return;
  }

  // ----------------------------------------------------------
  // NEMOTRON 3 NANO
  // ----------------------------------------------------------

  if (lower === "nvidia/nemotron-3-nano-30b-a3b") {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: REASONING_MODE !== "off"
    };

    return;
  }

  // ----------------------------------------------------------
  // FALLBACK
  // ----------------------------------------------------------
  //
  // If the model isn't in our special list, we don't modify
  // its reasoning settings.
  //
  // This is intentional.
  //
  // Unknown models should receive the normal OpenAI-compatible
  // request instead of us guessing at parameters.
  // ----------------------------------------------------------
}

// ============================================================
// CLEAN INCOMING MESSAGES
// ============================================================

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages
    .filter((message) => {
      if (!message) return false;
      if (!message.role) return false;
      if (message.content === undefined || message.content === null) {
        return false;
      }

      return true;
    })
    .map((message) => {
      // Preserve the original role.
      //
      // IMPORTANT:
      // We do NOT convert system → user.
      //
      return {
        role: message.role,
        content: message.content,
        ...(message.name ? { name: message.name } : {})
      };
    });
}

// ============================================================
// ERROR RESPONSE HELPER
// ============================================================

function sendError(res, status, message, details = null) {
  if (res.headersSent) {
    try {
      res.end();
    } catch (_) {}
    return;
  }

  res.status(status).json({
    error: {
      message,
      ...(details ? { details } : {})
    }
  });
}

// ============================================================
// MAIN CHAT COMPLETIONS ROUTE
// ============================================================

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return sendError(
        res,
        500,
        "NIM_API_KEY is not configured on Render."
      );
    }

    const incoming = req.body || {};

    // --------------------------------------------------------
    // MODEL
    // --------------------------------------------------------

    const model =
      incoming.model ||
      DEFAULT_MODEL;

    // --------------------------------------------------------
    // MESSAGES
    // --------------------------------------------------------

    const messages = normalizeMessages(
      incoming.messages
    );

    if (messages.length === 0) {
      return sendError(
        res,
        400,
        "No valid messages were supplied."
      );
    }

    // --------------------------------------------------------
    // STREAMING
    // --------------------------------------------------------

    const stream =
      incoming.stream !== undefined
        ? Boolean(incoming.stream)
        : true;

    // --------------------------------------------------------
    // GENERATION PARAMETERS
    // --------------------------------------------------------

    const nimRequest = {
      model,
      messages,

      temperature:
        typeof incoming.temperature === "number"
          ? incoming.temperature
          : DEFAULT_TEMPERATURE,

      top_p:
        typeof incoming.top_p === "number"
          ? incoming.top_p
          : DEFAULT_TOP_P,

      max_tokens:
        typeof incoming.max_tokens === "number"
          ? incoming.max_tokens
          : DEFAULT_MAX_TOKENS,

      stream
    };

    // --------------------------------------------------------
    // PRESERVE OPTIONAL PARAMETERS
    // --------------------------------------------------------
    //
    // These can be useful with certain models or clients.
    //

    if (incoming.stop !== undefined) {
      nimRequest.stop = incoming.stop;
    }

    if (incoming.seed !== undefined) {
      nimRequest.seed = incoming.seed;
    }

    if (incoming.tools !== undefined) {
      nimRequest.tools = incoming.tools;
    }

    if (incoming.tool_choice !== undefined) {
      nimRequest.tool_choice = incoming.tool_choice;
    }

    if (incoming.response_format !== undefined) {
      nimRequest.response_format =
        incoming.response_format;
    }

    // --------------------------------------------------------
    // COPY CLIENT-SUPPLIED CHAT TEMPLATE SETTINGS
    // --------------------------------------------------------
    //
    // This allows advanced clients to provide their own
    // chat_template_kwargs.
    //
    // Our model-specific settings below can override them
    // when necessary.
    //

    if (incoming.chat_template_kwargs) {
      nimRequest.chat_template_kwargs =
        incoming.chat_template_kwargs;
    }

    // --------------------------------------------------------
    // MODEL-SPECIFIC REASONING
    // --------------------------------------------------------

    addModelSpecificSettings(
      nimRequest,
      model
    );

    // --------------------------------------------------------
    // NVIDIA REQUEST
    // --------------------------------------------------------

    const axiosConfig = {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        "Content-Type": "application/json",
        Accept: stream
          ? "text/event-stream"
          : "application/json"
      },

      timeout: 0,

      validateStatus: () => true
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      stream
        ? {
            ...axiosConfig,
            responseType: "stream"
          }
        : axiosConfig
    );

    // --------------------------------------------------------
    // NVIDIA ERROR
    // --------------------------------------------------------

    if (response.status < 200 || response.status >= 300) {
      let errorBody = "";

      if (stream && response.data) {
        await new Promise((resolve) => {
          response.data.on("data", (chunk) => {
            errorBody += chunk.toString();
          });

          response.data.on("end", resolve);
          response.data.on("error", resolve);
        });
      } else {
        errorBody =
          typeof response.data === "string"
            ? response.data
            : JSON.stringify(response.data);
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

    // ========================================================
    // NON-STREAMING RESPONSE
    // ========================================================

    if (!stream) {
      return res.status(response.status).json(
        response.data
      );
    }

    // ========================================================
    // STREAMING RESPONSE
    // ========================================================

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

    // Flush headers immediately if supported.
    if (typeof res.flushHeaders === "function") {
      res.flushHeaders();
    }

    const upstream = response.data;

    let clientDisconnected = false;

    req.on("close", () => {
      clientDisconnected = true;

      try {
        upstream.destroy();
      } catch (_) {}
    });

    // --------------------------------------------------------
    // IMPORTANT:
    //
    // We DO NOT run a regex filter over the stream.
    //
    // NVIDIA's reasoning-capable endpoints send reasoning
    // separately from normal `content` in their SSE deltas.
    //
    // We strip `reasoning_content` from the SSE JSON while
    // preserving normal `content`.
    //
    // This lets the model THINK while JanitorAI only receives
    // the actual answer.
    // --------------------------------------------------------

    let buffer = "";

    upstream.on("data", (chunk) => {
      if (clientDisconnected) {
        return;
      }

      buffer += chunk.toString("utf8");

      // SSE events are separated by blank lines.
      const events = buffer.split(/\r?\n\r?\n/);

      // Keep incomplete event for next chunk.
      buffer = events.pop() || "";

      for (const event of events) {
        if (!event.trim()) {
          continue;
        }

        const lines = event.split(/\r?\n/);

        let shouldSend = false;
        let outputData = null;

        for (const line of lines) {
          if (!line.startsWith("data:")) {
            continue;
          }

          const data = line.slice(5).trim();

          if (!data) {
            continue;
          }

          // NVIDIA/OpenAI stream termination.
          if (data === "[DONE]") {
            outputData = "[DONE]";
            shouldSend = true;
            break;
          }

          try {
            const parsed = JSON.parse(data);

            if (
              parsed.choices &&
              Array.isArray(parsed.choices)
            ) {
              for (const choice of parsed.choices) {
                if (!choice.delta) {
                  continue;
                }

                // Remove reasoning from the outgoing delta.
                if (
                  Object.prototype.hasOwnProperty.call(
                    choice.delta,
                    "reasoning_content"
                  )
                ) {
                  delete choice.delta.reasoning_content;
                }

                // Some reasoning implementations use
                // `reasoning` instead.
                if (
                  Object.prototype.hasOwnProperty.call(
                    choice.delta,
                    "reasoning"
                  )
                ) {
                  delete choice.delta.reasoning;
                }
              }
            }

            outputData = JSON.stringify(parsed);
            shouldSend = true;
          } catch (_) {
            // If NVIDIA sends an unusual non-JSON SSE field,
            // preserve it rather than crashing the stream.
            outputData = data;
            shouldSend = true;
          }
        }

        if (shouldSend && outputData !== null) {
          res.write(`data: ${outputData}\n\n`);
        }
      }
    });

    upstream.on("end", () => {
      if (clientDisconnected) {
        return;
      }

      // Flush any final complete event that remained buffered.
      if (buffer.trim()) {
        const lines = buffer.split(/\r?\n/);

        for (const line of lines) {
          if (!line.startsWith("data:")) {
            continue;
          }

          const data = line.slice(5).trim();

          if (!data) {
            continue;
          }

          if (data === "[DONE]") {
            res.write("data: [DONE]\n\n");
            continue;
          }

          try {
            const parsed = JSON.parse(data);

            if (
              parsed.choices &&
              Array.isArray(parsed.choices)
            ) {
              for (const choice of parsed.choices) {
                if (!choice.delta) {
                  continue;
                }

                delete choice.delta.reasoning_content;
                delete choice.delta.reasoning;
              }
            }

            res.write(
              `data: ${JSON.stringify(parsed)}\n\n`
            );
          } catch (_) {
            res.write(`data: ${data}\n\n`);
          }
        }
      }

      res.end();
    });

    upstream.on("error", (error) => {
      console.error(
        "NVIDIA stream error:",
        error.message
      );

      if (!res.headersSent) {
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
    });

  } catch (error) {
    console.error(
      "Proxy error:",
      error.response?.data || error.message
    );

    if (res.headersSent) {
      try {
        res.end();
      } catch (_) {}

      return;
    }

    sendError(
      res,
      error.response?.status || 500,
      error.message || "Proxy request failed.",
      error.response?.data || null
    );
  }
});

// ============================================================
// START SERVER
// ============================================================

app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `🚀 JanitorAI → NVIDIA NIM proxy running on port ${PORT}`
  );

  console.log(
    `🤖 Default model: ${DEFAULT_MODEL}`
  );

  console.log(
    `🧠 Reasoning: ${REASONING_MODE}`
  );
});
