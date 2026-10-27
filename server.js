/* ============================================================
   MODEL CACHE
============================================================ */

let modelCache = {
  timestamp: 0,
  data: null
};

async function fetchRemoteModels() {
  const now = Date.now();

  if (
    modelCache.data &&
    now - modelCache.timestamp < MODEL_CACHE_TTL_MS
  ) {
    return modelCache.data;
  }

  try {
    const response = await axios.get(
      `${NIM_API_BASE}/models`,
      {
        headers: buildHeaders(false),
        timeout: 30000,
        httpAgent,
        httpsAgent,
        validateStatus: () => true
      }
    );

    if (
      response.status >= 200 &&
      response.status < 300 &&
      isPlainObject(response.data)
    ) {
      modelCache = {
        timestamp: now,
        data: response.data
      };

      return response.data;
    }
  } catch {
    // Fall back to the proxy's configured local model list.
  }

  return null;
}

function localModelList() {
  const created = Math.floor(Date.now() / 1000);
  const result = [];

  for (const [id, config] of Object.entries(MODELS)) {
    result.push({
      id,
      object: "model",
      created,
      owned_by: config.provider,
      name: config.name
    });
  }

  for (const [id, profile] of Object.entries(PROFILES)) {
    const config = MODELS[profile.baseModel];

    result.push({
      id,
      object: "model",
      created,
      owned_by: config?.provider || "proxy",
      name: `${config?.name || profile.baseModel} - ${profile.label}`
    });
  }

  return result;
}

/* ============================================================
   OPTIONAL PROXY AUTHENTICATION
============================================================ */

function proxyAuth(req, res, next) {
  // Authentication is disabled unless PROXY_API_KEY is configured.
  if (!PROXY_API_KEY) {
    return next();
  }

  const authorization = req.get("authorization") || "";
  const match = authorization.match(/^Bearer\s+(.+)$/i);

  if (!match) {
    return res.status(401).json({
      error: {
        message: "Missing or invalid proxy API key",
        type: "authentication_error",
        code: "unauthorized"
      }
    });
  }

  const supplied = Buffer.from(match[1], "utf8");
  const expected = Buffer.from(PROXY_API_KEY, "utf8");

  // timingSafeEqual requires buffers of equal length.
  if (
    supplied.length !== expected.length ||
    !crypto.timingSafeEqual(supplied, expected)
  ) {
    return res.status(401).json({
      error: {
        message: "Missing or invalid proxy API key",
        type: "authentication_error",
        code: "unauthorized"
      }
    });
  }

  return next();
}

/* ============================================================
   CORS / BODY PARSER
============================================================ */

app.use(
  cors({
    origin: true,
    credentials: true
  })
);

app.use(
  express.json({
    limit: process.env.JSON_BODY_LIMIT || "10mb"
  })
);

// Protect /v1 endpoints if PROXY_API_KEY is set.
// Root and health endpoints remain public.
app.use("/v1", proxyAuth);

/* ============================================================
   ROOT
============================================================ */

app.get("/", (req, res) => {
  res.json({
    name: "NVIDIA NIM OpenAI-Compatible Proxy",
    status: "ok",
    default_model: DEFAULT_MODEL,
    upstream: NIM_API_BASE,
    profile_count: Object.keys(PROFILES).length
  });
});

/* ============================================================
   HEALTH
============================================================ */

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    upstream: NIM_API_BASE,
    default_model: DEFAULT_MODEL,
    api_key_configured: Boolean(NIM_API_KEY),
    proxy_auth_enabled: Boolean(PROXY_API_KEY)
  });
});

/* ============================================================
   MODELS
============================================================ */

app.get("/v1/models", async (req, res) => {
  const remote = await fetchRemoteModels();
  const local = localModelList();

  if (
    isPlainObject(remote) &&
    Array.isArray(remote.data)
  ) {
    const ids = new Set(
      remote.data
        .filter(isPlainObject)
        .map((model) => model.id)
    );

    const merged = [...remote.data];

    for (const model of local) {
      if (!ids.has(model.id)) {
        merged.push(model);
      }
    }

    return res.json({
      object: "list",
      data: merged
    });
  }

  return res.json({
    object: "list",
    data: local
  });
});

app.post("/v1/models/refresh", async (req, res) => {
  modelCache = {
    timestamp: 0,
    data: null
  };

  const remote = await fetchRemoteModels();

  res.json({
    object: "list",
    data: remote?.data || localModelList()
  });
});

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post("/v1/chat/completions", async (req, res) => {
  const startedAt = Date.now();

  const body = isPlainObject(req.body)
    ? req.body
    : {};

  const requestedModel = normalizeModel(
    body.model || DEFAULT_MODEL
  );

  const resolution = resolveModel(requestedModel);

  if (!resolution) {
    return res.status(400).json({
      error: {
        message: `Unknown model: ${requestedModel}`,
        type: "invalid_request_error",
        code: "unknown_model"
      }
    });
  }

  const nimRequest = buildNimRequest(body, resolution);
  const streaming = Boolean(nimRequest.stream);

  debugLog(
    "request",
    requestMetadata(body, resolution)
  );

  debugLog("upstream_request_metadata", {
    model: nimRequest.model,
    stream: Boolean(nimRequest.stream),
    max_tokens: nimRequest.max_tokens,
    temperature: nimRequest.temperature,
    has_tools:
      Array.isArray(nimRequest.tools) &&
      nimRequest.tools.length > 0,
    chat_template_kwargs:
      nimRequest.chat_template_kwargs || null,
    thinking_token_budget:
      nimRequest.thinking_token_budget || null,
    reasoning_effort:
      nimRequest.reasoning_effort || null
  });

  try {
    // Exactly one upstream request; no automatic retries.
    const upstream = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: buildHeaders(streaming),
        timeout: NIM_TIMEOUT_MS,
        httpAgent,
        httpsAgent,
        responseType: streaming ? "stream" : "json",
        validateStatus: () => true,
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      }
    );

    const elapsedMs = Date.now() - startedAt;

    /* UPSTREAM ERROR */

    if (
      upstream.status < 200 ||
      upstream.status >= 300
    ) {
      let errorData = upstream.data;

      if (
        errorData &&
        typeof errorData.pipe === "function"
      ) {
        errorData = await streamToBuffer(errorData);
      }

      const errorObject = makeProxyError(
        {
          response: {
            status: upstream.status,
            data: errorData
          }
        },
        resolution,
        elapsedMs
      );

      console.error(
        `[proxy] NIM ${upstream.status} ` +
        `${resolution.model} ` +
        `profile=${resolution.profile?.label || "default"}`
      );

      if (DEBUG_PROXY) {
        debugLog("upstream_error", {
          status: upstream.status,
          model: resolution.model,
          profile: resolution.profile?.label || null,
          adapter: resolution.config?.adapter || null,
          upstream_message:
            extractErrorBody(errorData)?.error?.message ||
            extractErrorBody(errorData)?.message ||
            null
        });
      }

      return res.status(upstream.status).json(errorObject);
    }

    /* NON-STREAMING RESPONSE */

    if (!streaming) {
      const cleaned = STRIP_REASONING
        ? cleanResponseObject(upstream.data)
        : upstream.data;

      return res.status(upstream.status).json(cleaned);
    }

    /* STREAMING RESPONSE */

    res.status(upstream.status);

    res.setHeader(
      "Content-Type",
      "text/event-stream; charset=utf-8"
    );

    res.setHeader(
      "Cache-Control",
      "no-cache, no-transform"
    );

    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();

    /*
     * If reasoning stripping is disabled, relay the upstream
     * stream directly without parsing or changing its events.
     */
    if (!STRIP_REASONING) {
      upstream.data.on("data", (chunk) => {
        if (!res.writableEnded) {
          res.write(chunk);
        }
      });

      upstream.data.on("end", () => {
        if (!res.writableEnded) {
          res.end();
        }
      });

      upstream.data.on("error", (streamError) => {
        console.error(
          `[proxy] upstream stream error: ` +
          `${streamError?.message || "unknown error"}`
        );

        if (!res.writableEnded) {
          res.end();
        }
      });

      res.on("close", () => {
        // Abort upstream only when the client disconnects early.
        if (
          !res.writableEnded &&
          upstream.data &&
          typeof upstream.data.destroy === "function"
        ) {
          upstream.data.destroy();
        }
      });

      return;
    }

    const thinkingFilter = new ThinkingStreamFilter();

    const parser = new SSEParser((event) => {
      if (res.writableEnded) {
        return;
      }

      if (event.data === "[DONE]") {
        thinkingFilter.flush();

        writeSSEEvent(res, {
          event: event.event,
          id: event.id,
          data: "[DONE]"
        });

        return;
      }

      const parsed = safeJsonParse(event.data);

      if (!parsed) {
        writeSSEEvent(res, event);
        return;
      }

      const cleaned = cleanSSEJson(parsed, thinkingFilter);

      writeSSEEvent(res, {
        event: event.event,
        id: event.id,
        data: JSON.stringify(cleaned)
      });
    });

    upstream.data.on("data", (chunk) => {
      if (!res.writableEnded) {
        parser.push(chunk.toString("utf8"));
      }
    });

    upstream.data.on("end", () => {
      if (res.writableEnded) {
        return;
      }

      parser.end();
      thinkingFilter.flush();
      res.end();
    });

    upstream.data.on("error", (streamError) => {
      console.error(
        `[proxy] upstream stream error for ${resolution.model}: ` +
        `${streamError?.message || "unknown error"}`
      );

      if (!res.writableEnded) {
        res.end();
      }
    });

    res.on("close", () => {
      // req.close is not used: it can fire after request-body parsing.
      if (
        !res.writableEnded &&
        upstream.data &&
        typeof upstream.data.destroy === "function"
      ) {
        upstream.data.destroy();
      }
    });
  } catch (error) {
    const elapsedMs = Date.now() - startedAt;

    console.error(
      `[proxy] request failed for ${resolution.model}: ` +
      `${error?.message || "unknown error"}`
    );

    if (res.headersSent) {
      return res.end();
    }

    return res
      .status(Number(error?.response?.status) || 502)
      .json(
        makeProxyError(
          error,
          resolution,
          elapsedMs
        )
      );
  }
});

/* ============================================================
   404
============================================================ */

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Not found",
      type: "not_found"
    }
  });
});

/* ============================================================
   GLOBAL ERROR HANDLER
============================================================ */

app.use((error, req, res, next) => {
  console.error(
    `[proxy] unhandled error: ` +
    `${error?.message || "unknown error"}`
  );

  if (res.headersSent) {
    return next(error);
  }

  res.status(500).json({
    error: {
      message: error?.message || "Internal proxy error",
      type: "internal_error"
    }
  });
});

/* ============================================================
   START SERVER
============================================================ */

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`[proxy] listening on port ${PORT}`);
  console.log(`[proxy] upstream=${NIM_API_BASE}`);
  console.log(`[proxy] default_model=${DEFAULT_MODEL}`);
  console.log(`[proxy] api_key_configured=${Boolean(NIM_API_KEY)}`);
  console.log(`[proxy] proxy_auth_enabled=${Boolean(PROXY_API_KEY)}`);
  console.log(`[proxy] debug=${DEBUG_PROXY}`);
  console.log(`[proxy] reasoning_stripping=${STRIP_REASONING}`);
  console.log(`[proxy] profiles=${Object.keys(PROFILES).length}`);
});

/* ============================================================
   GRACEFUL SHUTDOWN
============================================================ */

function shutdown(signal) {
  console.log(`[proxy] received ${signal}; shutting down`);

  server.close(() => {
    httpAgent.destroy();
    httpsAgent.destroy();
    process.exit(0);
  });

  setTimeout(() => {
    process.exit(1);
  }, 10000).unref();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
