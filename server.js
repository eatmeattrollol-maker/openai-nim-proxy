"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const crypto = require("crypto");
const http = require("http");
const https = require("https");

const app = express();

const PORT = Number(process.env.PORT) || 10000;

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
  "moonshotai/kimi-k3";

/* ============================================================
   MODEL CONFIGURATION
============================================================ */

const MODELS = {
  "deepseek-ai/deepseek-v4-flash-0731": {
    name: "DeepSeek V4 Flash 0731",
    provider: "DeepSeek AI",
    reasoningLevels: ["none", "low", "high", "max"],
    defaultReasoningEffort: "low",
    maxOutputTokens: 22768
  },

  "deepseek-ai/deepseek-v4-pro-0813": {
    name: "DeepSeek V4 Pro 0813",
    provider: "DeepSeek AI",
    reasoningLevels: ["none", "low", "high", "max"],
    defaultReasoningEffort: "max",
    maxOutputTokens: 16384
  },

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",
    reasoningLevels: ["low", "high", "max"],
    defaultReasoningEffort: "max",
    maxOutputTokens: 1048576
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",
    reasoningLevels: ["none", "medium", "high"],
    defaultReasoningEffort: "high",
    maxOutputTokens: 32768
  }
};

const ALLOW_UNKNOWN_MODELS =
  String(process.env.ALLOW_UNKNOWN_MODELS || "true")
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   REASONING
   IMPORTANT: VALUES PRESERVED FROM ORIGINAL
============================================================ */

const DEFAULT_REASONING_EFFORT =
  String(process.env.DEFAULT_REASONING_EFFORT || "high")
    .trim()
    .toLowerCase();

const DEFAULT_REASONING_BUDGET =
  Number.isFinite(Number(process.env.DEFAULT_REASONING_BUDGET))
    ? Number(process.env.DEFAULT_REASONING_BUDGET)
    : 16384;

const MODEL_REASONING_BUDGETS = {
  "deepseek-ai/deepseek-v4-flash-0731": 16384,
  "deepseek-ai/deepseek-v4-pro-0813": 24576,
  "moonshotai/kimi-k3": 32768,
  "nvidia/nemotron-3-ultra-550b-a55b": 16384
};

const MODEL_REASONING_EFFORTS = {
  "deepseek-ai/deepseek-v4-flash-0731": "low",
  "deepseek-ai/deepseek-v4-pro-0813": "max",
  "moonshotai/kimi-k3": "max",
  "nvidia/nemotron-3-ultra-550b-a55b": "high"
};

const DEFAULT_MAX_TOKENS =
  Number.isFinite(Number(process.env.DEFAULT_MAX_TOKENS))
    ? Number(process.env.DEFAULT_MAX_TOKENS)
    : 16384;

const DEFAULT_TEMPERATURE =
  Number.isFinite(Number(process.env.DEFAULT_TEMPERATURE))
    ? Number(process.env.DEFAULT_TEMPERATURE)
    : 1.0;

const DEFAULT_TOP_P =
  Number.isFinite(Number(process.env.DEFAULT_TOP_P))
    ? Number(process.env.DEFAULT_TOP_P)
    : 0.95;

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

/* ============================================================
   NETWORK
============================================================ */

const NIM_TIMEOUT =
  Number.isFinite(Number(process.env.NIM_TIMEOUT_MS))
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const MAX_ERROR_BODY_SIZE = 2 * 1024 * 1024;

const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 64,
  maxFreeSockets: 16,
  scheduling: "lifo"
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 64,
  maxFreeSockets: 16,
  scheduling: "lifo"
});

/* ============================================================
   DEBUG
============================================================ */

const DEBUG_PROXY =
  String(process.env.DEBUG_PROXY || "false")
    .trim()
    .toLowerCase() === "true";

/* ============================================================
   KIMI MEMORY CONFIGURATION
============================================================ */

/*const KIMI_MEMORY_ENABLED =
  String(process.env.KIMI_MEMORY_ENABLED || "true")
    .trim()
    .toLowerCase() === "true";

const KIMI_CONTEXT_BUDGET =
  Number.isFinite(Number(process.env.KIMI_CONTEXT_BUDGET))
    ? Math.min(
        1048576,
        Math.max(
          32768,
          Math.floor(Number(process.env.KIMI_CONTEXT_BUDGET))
        )
      )
    : 1000000;

const KIMI_CHARS_PER_TOKEN =
  Number.isFinite(Number(process.env.KIMI_CHARS_PER_TOKEN))
    ? Math.max(2, Number(process.env.KIMI_CHARS_PER_TOKEN))
    : 3.5;

const KIMI_MAX_STORED_MESSAGES =
  Number.isFinite(Number(process.env.KIMI_MAX_STORED_MESSAGES))
    ? Math.max(
        100,
        Math.floor(Number(process.env.KIMI_MAX_STORED_MESSAGES))
      )
    : 10000;

const KIMI_MEMORY_TTL_MS =
  Number.isFinite(Number(process.env.KIMI_MEMORY_TTL_MS))
    ? Math.max(60000, Number(process.env.KIMI_MEMORY_TTL_MS))
    : 7 * 24 * 60 * 60 * 1000;

const KIMI_MAX_CONVERSATIONS =
  Number.isFinite(Number(process.env.KIMI_MAX_CONVERSATIONS))
    ? Math.max(
        10,
        Math.floor(Number(process.env.KIMI_MAX_CONVERSATIONS))
      )
    : 2000;

/* ============================================================
   STATE
============================================================ */

/*
 * Conversation store:
 *
 * conversationId -> {
 *   messages: [],
 *   updatedAt,
 *   systemFingerprint,
 *   firstUserFingerprint
 * }
 */
/*const kimiConversationStore = new Map();

/*
 * Fast indexes used to avoid scanning every conversation.
 *
 * fingerprint -> Set(conversationId)
 */
/*const kimiFirstUserIndex = new Map();
const kimiSystemIndex = new Map();

let modelCache = null;
let modelCacheTimestamp = 0;
let modelCachePromise = null;

/* ============================================================
   BASIC HELPERS
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

  if (typeof value === "string" && value.trim() !== "") {
    return Number.isFinite(Number(value));
  }

  return false;
}

function numberOrDefault(value, fallback) {
  return isFiniteNumber(value)
    ? Number(value)
    : fallback;
}

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
    const normalized = value.trim().toLowerCase();

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

function getModelConfig(model) {
  return MODELS[normalizeModel(model)] || null;
}

function isSupportedModel(model) {
  return !!getModelConfig(model) || ALLOW_UNKNOWN_MODELS;
}

function clampTemperature(value) {
  return Math.min(
    2,
    Math.max(
      0,
      numberOrDefault(
        value,
        DEFAULT_TEMPERATURE
      )
    )
  );
}

function clampTopP(value) {
  return Math.min(
    1,
    Math.max(
      0,
      numberOrDefault(
        value,
        DEFAULT_TOP_P
      )
    )
  );
}

function clampMaxTokens(value, modelConfig) {
  const number = Math.max(
    1,
    Math.floor(
      numberOrDefault(
        value,
        DEFAULT_MAX_TOKENS
      )
    )
  );

  return modelConfig?.maxOutputTokens
    ? Math.min(
        modelConfig.maxOutputTokens,
        number
      )
    : number;
}

function clampReasoningBudget(value) {
  return Math.min(
    65536,
    Math.max(
      0,
      Math.floor(
        numberOrDefault(
          value,
          DEFAULT_REASONING_BUDGET
        )
      )
    )
  );
}

/* ============================================================
   REASONING
   EXACT BEHAVIOR PRESERVED
============================================================ */

function budgetToReasoningEffort(budget, modelConfig) {
  const value = clampReasoningBudget(budget);
  const levels = modelConfig?.reasoningLevels || [];

  if (levels.includes("none") && value <= 0) {
    return "none";
  }

  if (levels.includes("low") && value <= 8192) {
    return "low";
  }

  if (levels.includes("medium") && value <= 16384) {
    return "medium";
  }

  if (levels.includes("high") && value <= 24576) {
    return "high";
  }

  if (levels.includes("max")) {
    return "max";
  }

  if (levels.includes("high")) {
    return "high";
  }

  if (levels.includes("medium")) {
    return "medium";
  }

  if (levels.includes("low")) {
    return "low";
  }

  return levels[0] || "none";
}

function normalizeReasoningEffort(value, modelConfig) {
  const levels = modelConfig?.reasoningLevels || [];

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();

    if (levels.includes(normalized)) {
      return normalized;
    }

    if (
      (
        normalized === "off" ||
        normalized === "false" ||
        normalized === "0"
      ) &&
      levels.includes("none")
    ) {
      return "none";
    }

    if (
      (
        normalized === "med" ||
        normalized === "medium"
      ) &&
      levels.includes("medium")
    ) {
      return "medium";
    }

    if (
      (
        normalized === "maximum" ||
        normalized === "maximum_reasoning"
      ) &&
      levels.includes("max")
    ) {
      return "max";
    }
  }

  return (
    modelConfig?.defaultReasoningEffort ||
    levels[0] ||
    "none"
  );
}

function getRequestedReasoningEffort(
  incoming,
  model,
  modelConfig
) {
  if (incoming.reasoning_effort !== undefined) {
    return modelConfig
      ? normalizeReasoningEffort(
          incoming.reasoning_effort,
          modelConfig
        )
      : String(
          incoming.reasoning_effort
        ).trim();
  }

  if (incoming.reasoning_mode !== undefined) {
    return modelConfig
      ? normalizeReasoningEffort(
          incoming.reasoning_mode,
          modelConfig
        )
      : String(
          incoming.reasoning_mode
        ).trim();
  }

  if (incoming.reasoning_budget !== undefined) {
    return modelConfig
      ? budgetToReasoningEffort(
          incoming.reasoning_budget,
          modelConfig
        )
      : null;
  }

  if (MODEL_REASONING_EFFORTS[model]) {
    return normalizeReasoningEffort(
      MODEL_REASONING_EFFORTS[model],
      modelConfig
    );
  }

  return modelConfig
    ? normalizeReasoningEffort(
        DEFAULT_REASONING_EFFORT,
        modelConfig
      )
    : null;
}

/* ============================================================
   KIMI MEMORY HELPERS
============================================================ */

function estimateKimiTokens(value) {
  if (value === undefined || value === null) {
    return 0;
  }

  if (typeof value === "string") {
    return Math.ceil(
      value.length / KIMI_CHARS_PER_TOKEN
    );
  }

  try {
    return Math.ceil(
      JSON.stringify(value).length /
        KIMI_CHARS_PER_TOKEN
    );
  } catch {
    return 0;
  }
}

function estimateKimiMessageTokens(message) {
  return (
    8 +
    estimateKimiTokens(message?.content) +
    estimateKimiTokens(message?.reasoning_content) +
    estimateKimiTokens(message?.tool_calls) +
    estimateKimiTokens(message?.function_call) +
    estimateKimiTokens(message?.name)
  );
}

function estimateKimiMessagesTokens(messages) {
  let total = 0;

  for (const message of messages) {
    total += estimateKimiMessageTokens(message);
  }

  return total;
}

function hashKimiValue(value) {
  return crypto
    .createHash("sha256")
    .update(String(value || ""), "utf8")
    .digest("hex");
}

/*
 * Cache the expensive signature directly on the object.
 *
 * This avoids repeatedly doing:
 *
 * JSON.stringify()
 * SHA-256
 *
 * for the same message.
 */
function kimiMessageSignature(message) {
  if (
    message &&
    typeof message === "object" &&
    typeof message.__kimiSignature === "string"
  ) {
    return message.__kimiSignature;
  }

  const signature = hashKimiValue(
    JSON.stringify({
      role: message?.role || "",
      content: message?.content ?? null,
      reasoning_content:
        message?.reasoning_content ?? null,
      name: message?.name ?? null,
      tool_calls:
        message?.tool_calls ?? null,
      function_call:
        message?.function_call ?? null
    })
  );

  if (
    message &&
    typeof message === "object"
  ) {
    Object.defineProperty(
      message,
      "__kimiSignature",
      {
        value: signature,
        enumerable: false,
        configurable: true
      }
    );
  }

  return signature;
}

function getExplicitKimiConversationId(
  req,
  incoming
) {
  const candidates = [
    incoming?.conversation_id,
    incoming?.conversationId,
    incoming?.session_id,
    incoming?.sessionId,
    incoming?.chat_id,
    incoming?.chatId,
    req?.headers?.["x-conversation-id"],
    req?.headers?.["x-session-id"],
    req?.headers?.["x-chat-id"]
  ];

  for (const candidate of candidates) {
    if (
      typeof candidate === "string" &&
      candidate.trim()
    ) {
      return candidate.trim();
    }

    if (
      typeof candidate === "number" &&
      Number.isFinite(candidate)
    ) {
      return String(candidate);
    }
  }

  return null;
}

function getKimiStableFingerprint(messages) {
  let system = "";
  let firstUser = "";
  let systemCount = 0;

  for (const message of messages) {
    if (
      (
        message?.role === "system" ||
        message?.role === "developer"
      ) &&
      systemCount < 3
    ) {
      system += JSON.stringify({
        role: message.role,
        content: message.content
      });

      systemCount++;

      continue;
    }

    if (
      !firstUser &&
      message?.role === "user"
    ) {
      firstUser = JSON.stringify({
        role: message.role,
        content: message.content
      });
    }

    if (
      systemCount >= 3 &&
      firstUser
    ) {
      break;
    }
  }

  return {
    systemFingerprint:
      hashKimiValue(system),

    firstUserFingerprint:
      firstUser
        ? hashKimiValue(firstUser)
        : null
  };
}

/* ============================================================
   KIMI INDEX MANAGEMENT
============================================================ */

function addToIndex(
  index,
  fingerprint,
  conversationId
) {
  if (!fingerprint) {
    return;
  }

  let set = index.get(fingerprint);

  if (!set) {
    set = new Set();
    index.set(fingerprint, set);
  }

  set.add(conversationId);
}

function removeFromIndex(
  index,
  fingerprint,
  conversationId
) {
  if (!fingerprint) {
    return;
  }

  const set = index.get(fingerprint);

  if (!set) {
    return;
  }

  set.delete(conversationId);

  if (set.size === 0) {
    index.delete(fingerprint);
  }
}

function indexConversation(
  conversationId,
  entry
) {
  addToIndex(
    kimiFirstUserIndex,
    entry.firstUserFingerprint,
    conversationId
  );

  addToIndex(
    kimiSystemIndex,
    entry.systemFingerprint,
    conversationId
  );
}

function unindexConversation(
  conversationId,
  entry
) {
  removeFromIndex(
    kimiFirstUserIndex,
    entry.firstUserFingerprint,
    conversationId
  );

  removeFromIndex(
    kimiSystemIndex,
    entry.systemFingerprint,
    conversationId
  );
}

function deleteKimiConversation(
  conversationId
) {
  const entry =
    kimiConversationStore.get(
      conversationId
    );

  if (!entry) {
    return;
  }

  unindexConversation(
    conversationId,
    entry
  );

  kimiConversationStore.delete(
    conversationId
  );
}

function touchKimiConversation(
  conversationId
) {
  const entry =
    kimiConversationStore.get(
      conversationId
    );

  if (!entry) {
    return;
  }

  entry.updatedAt = Date.now();

  /*
   * Map insertion order is used as a cheap LRU.
   */
  kimiConversationStore.delete(
    conversationId
  );

  kimiConversationStore.set(
    conversationId,
    entry
  );
}

/* ============================================================
   KIMI MEMORY CLEANUP
============================================================ */

function cleanupKimiMemory() {
  const cutoff =
    Date.now() -
    KIMI_MEMORY_TTL_MS;

  for (
    const [
      conversationId,
      entry
    ] of kimiConversationStore
  ) {
    if (
      entry.updatedAt < cutoff
    ) {
      deleteKimiConversation(
        conversationId
      );
    }
  }
}

setInterval(
  cleanupKimiMemory,
  Math.min(
    KIMI_MEMORY_TTL_MS,
    60 * 60 * 1000
  )
).unref();

/* ============================================================
   CREATE CONVERSATION
============================================================ */

function createKimiConversation(
  conversationId,
  messages
) {
  const fingerprints =
    getKimiStableFingerprint(
      messages
    );

  const entry = {
    messages: [],
    updatedAt: Date.now(),
    systemFingerprint:
      fingerprints.systemFingerprint,
    firstUserFingerprint:
      fingerprints.firstUserFingerprint
  };

  kimiConversationStore.set(
    conversationId,
    entry
  );

  indexConversation(
    conversationId,
    entry
  );

  /*
   * Cheap LRU eviction.
   */
  while (
    kimiConversationStore.size >
    KIMI_MAX_CONVERSATIONS
  ) {
    const oldestKey =
      kimiConversationStore
        .keys()
        .next()
        .value;

    if (
      oldestKey === undefined
    ) {
      break;
    }

    deleteKimiConversation(
      oldestKey
    );
  }

  return entry;
}

/* ============================================================
   FAST HISTORY MATCHING
============================================================ */

function findKimiConversationByHistory(
  messages
) {
  if (
    !Array.isArray(messages) ||
    messages.length === 0
  ) {
    return null;
  }

  const fingerprints =
    getKimiStableFingerprint(
      messages
    );

  /*
   * Instead of scanning all 2,000 conversations,
   * start with conversations having the same first
   * user message.
   */
  let candidateIds = null;

  if (
    fingerprints.firstUserFingerprint
  ) {
    candidateIds =
      kimiFirstUserIndex.get(
        fingerprints.firstUserFingerprint
      );
  }

  /*
   * If there isn't a first-user match,
   * use the system fingerprint.
   */
  if (
    !candidateIds &&
    fingerprints.systemFingerprint
  ) {
    candidateIds =
      kimiSystemIndex.get(
        fingerprints.systemFingerprint
      );
  }

  /*
   * No indexed candidate.
   *
   * Preserve the old behavior as a fallback,
   * but this should be uncommon.
   */
  if (!candidateIds) {
    return null;
  }

  const incomingSignatures =
    new Set();

  for (
    const message of messages
  ) {
    incomingSignatures.add(
      kimiMessageSignature(message)
    );
  }

  let bestMatch = null;
  let bestScore = 0;

  for (
    const conversationId of candidateIds
  ) {
    const entry =
      kimiConversationStore.get(
        conversationId
      );

    if (!entry) {
      continue;
    }

    let score = 0;

    if (
      fingerprints.firstUserFingerprint &&
      entry.firstUserFingerprint ===
        fingerprints.firstUserFingerprint
    ) {
      score += 1000;
    }

    if (
      fingerprints.systemFingerprint &&
      entry.systemFingerprint ===
        fingerprints.systemFingerprint
    ) {
      score += 100;
    }

    if (entry.messages.length) {
      let overlap = 0;

      for (
        const message of entry.messages
      ) {
        if (
          incomingSignatures.has(
            kimiMessageSignature(message)
          )
        ) {
          overlap++;

          if (overlap >= 50) {
            break;
          }
        }
      }

      score += Math.min(
        overlap * 10,
        500
      );
    }

    if (score > bestScore) {
      bestScore = score;

      bestMatch = {
        conversationId,
        entry
      };

      if (score >= 1600) {
        break;
      }
    }
  }

  return bestScore >= 10
    ? bestMatch
    : null;
}

/* ============================================================
   GET OR CREATE KIMI CONVERSATION
============================================================ */

function getOrCreateKimiConversation(
  req,
  incoming,
  messages
) {
  const explicitId =
    getExplicitKimiConversationId(
      req,
      incoming
    );

  if (explicitId) {
    const conversationId =
      "explicit:" +
      hashKimiValue(explicitId);

    let entry =
      kimiConversationStore.get(
        conversationId
      );

    if (!entry) {
      entry =
        createKimiConversation(
          conversationId,
          messages
        );
    }

    touchKimiConversation(
      conversationId
    );

    return {
      conversationId,
      entry
    };
  }

  const matched =
    findKimiConversationByHistory(
      messages
    );

  if (matched) {
    touchKimiConversation(
      matched.conversationId
    );

    return matched;
  }

  const fingerprints =
    getKimiStableFingerprint(
      messages
    );

  const seed =
    fingerprints.systemFingerprint +
    "|" +
    (
      fingerprints.firstUserFingerprint ||
      ""
    ) +
    "|" +
    Date.now() +
    "|" +
    Math.random();

  const conversationId =
    "derived:" +
    hashKimiValue(seed);

  const entry =
    createKimiConversation(
      conversationId,
      messages
    );

  return {
    conversationId,
    entry
  };
}

/* ============================================================
   MERGE KIMI MESSAGES
============================================================ */

function mergeKimiMessages(
  storedMessages,
  incomingMessages
) {
  const result = [];
  const seen = new Set();

  for (
    const message of storedMessages
  ) {
    const signature =
      kimiMessageSignature(message);

    if (!seen.has(signature)) {
      seen.add(signature);
      result.push(message);
    }
  }

  for (
    const message of incomingMessages
  ) {
    const signature =
      kimiMessageSignature(message);

    if (!seen.has(signature)) {
      seen.add(signature);
      result.push(message);
    }
  }

  return result;
}

/* ============================================================
   FIT KIMI CONTEXT
============================================================ */

function fitKimiContext(messages) {
  if (
    !Array.isArray(messages) ||
    messages.length === 0
  ) {
    return [];
  }

  const pinned = [];
  const normal = [];

  for (
    const message of messages
  ) {
    if (
      message?.role === "system" ||
      message?.role === "developer"
    ) {
      pinned.push(message);
    } else {
      normal.push(message);
    }
  }

  let used =
    estimateKimiMessagesTokens(
      pinned
    );

  if (
    used >= KIMI_CONTEXT_BUDGET
  ) {
    return pinned;
  }

  const selected = [];

  /*
   * Walk backwards so the most recent
   * messages get priority.
   */
  for (
    let index = normal.length - 1;
    index >= 0;
    index--
  ) {
    const message =
      normal[index];

    const cost =
      estimateKimiMessageTokens(
        message
      );

    if (
      used + cost >
      KIMI_CONTEXT_BUDGET
    ) {
      continue;
    }

    selected.push(message);
    used += cost;
  }

  selected.reverse();

  return pinned.concat(
    selected
  );
}

/* ============================================================
   UPDATE KIMI MEMORY
============================================================ */

function updateKimiMemory(
  conversationId,
  incomingMessages
) {
  let memory =
    kimiConversationStore.get(
      conversationId
    );

  if (!memory) {
    memory =
      createKimiConversation(
        conversationId,
        incomingMessages
      );
  }

  const merged =
    mergeKimiMessages(
      memory.messages,
      incomingMessages
    );

  const fitted =
    fitKimiContext(merged);

  memory.messages =
    fitted.slice(
      -KIMI_MAX_STORED_MESSAGES
    );

  const fingerprints =
    getKimiStableFingerprint(
      memory.messages
    );

  /*
   * Update indexes if fingerprints changed.
   */
  if (
    memory.systemFingerprint !==
    fingerprints.systemFingerprint
  ) {
    removeFromIndex(
      kimiSystemIndex,
      memory.systemFingerprint,
      conversationId
    );

    memory.systemFingerprint =
      fingerprints.systemFingerprint;

    addToIndex(
      kimiSystemIndex,
      memory.systemFingerprint,
      conversationId
    );
  }

  if (
    fingerprints.firstUserFingerprint &&
    memory.firstUserFingerprint !==
      fingerprints.firstUserFingerprint
  ) {
    removeFromIndex(
      kimiFirstUserIndex,
      memory.firstUserFingerprint,
      conversationId
    );

    memory.firstUserFingerprint =
      fingerprints.firstUserFingerprint;

    addToIndex(
      kimiFirstUserIndex,
      memory.firstUserFingerprint,
      conversationId
    );
  }

  memory.updatedAt = Date.now();

  touchKimiConversation(
    conversationId
  );

  return memory.messages;
}

function buildKimiMemoryContext(
  req,
  incoming,
  incomingMessages
) {
  if (!KIMI_MEMORY_ENABLED) {
    return {
      conversationId: null,
      messages: incomingMessages
    };
  }

  const conversation =
    getOrCreateKimiConversation(
      req,
      incoming,
      incomingMessages
    );

  return {
    conversationId:
      conversation.conversationId,

    messages:
      updateKimiMemory(
        conversation.conversationId,
        incomingMessages
      )
  };
}

/* ============================================================
   KIMI ASSISTANT MEMORY
============================================================ */

function appendKimiAssistantMessage(
  conversationId,
  assistantMessage
) {
  if (
    !KIMI_MEMORY_ENABLED ||
    !conversationId ||
    !assistantMessage
  ) {
    return;
  }

  const memory =
    kimiConversationStore.get(
      conversationId
    );

  if (!memory) {
    return;
  }

  const clean = {
    role:
      assistantMessage.role ||
      "assistant"
  };

  if (
    assistantMessage.content !==
    undefined
  ) {
    clean.content =
      assistantMessage.content;
  }

  if (
    assistantMessage.reasoning_content !==
    undefined
  ) {
    clean.reasoning_content =
      assistantMessage.reasoning_content;
  }

  if (
    Array.isArray(
      assistantMessage.tool_calls
    )
  ) {
    clean.tool_calls =
      assistantMessage.tool_calls;
  }

  if (
    assistantMessage.function_call !==
    undefined
  ) {
    clean.function_call =
      assistantMessage.function_call;
  }

  if (
    assistantMessage.name !==
    undefined
  ) {
    clean.name =
      assistantMessage.name;
  }

  const signature =
    kimiMessageSignature(clean);

  /*
   * Since signatures are cached,
   * this check is cheaper than the original.
   */
  for (
    const message of memory.messages
  ) {
    if (
      kimiMessageSignature(message) ===
      signature
    ) {
      return;
    }
  }

  memory.messages.push(clean);

  memory.messages =
    fitKimiContext(
      memory.messages
    ).slice(
      -KIMI_MAX_STORED_MESSAGES
    );

  memory.updatedAt = Date.now();

  touchKimiConversation(
    conversationId
  );
}

function extractKimiAssistantMessage(
  parsed
) {
  const message =
    parsed?.choices?.[0]?.message;

  if (!message) {
    return null;
  }

  return {
    role:
      message.role ||
      "assistant",

    content:
      message.content,

    reasoning_content:
      message.reasoning_content,

    tool_calls:
      message.tool_calls,

    function_call:
      message.function_call,

    name:
      message.name
  };
}

/* ============================================================
   KIMI STREAM ACCUMULATOR
============================================================ */

function createKimiStreamAccumulator() {
  return {
    role: "assistant",
    contentParts: [],
    reasoningParts: [],
    toolCalls: [],
    functionCall: null,
    name: undefined
  };
}

function mergeKimiToolCallDelta(
  accumulator,
  toolCall
) {
  if (
    !toolCall ||
    typeof toolCall !== "object"
  ) {
    return;
  }

  const index =
    Number.isFinite(
      Number(toolCall.index)
    )
      ? Number(toolCall.index)
      : accumulator.toolCalls.length;

  while (
    accumulator.toolCalls.length <=
    index
  ) {
    accumulator.toolCalls.push({});
  }

  const target =
    accumulator.toolCalls[index];

  if (
    toolCall.id !== undefined
  ) {
    target.id =
      toolCall.id;
  }

  if (
    toolCall.type !== undefined
  ) {
    target.type =
      toolCall.type;
  }

  if (toolCall.function) {
    if (!target.function) {
      target.function = {};
    }

    if (
      toolCall.function.name !==
      undefined
    ) {
      target.function.name =
        toolCall.function.name;
    }

    if (
      toolCall.function.arguments !==
      undefined
    ) {
      target.function.arguments =
        (
          target.function.arguments ||
          ""
        ) +
        toolCall.function.arguments;
    }
  }

  for (
    const key of Object.keys(toolCall)
  ) {
    if (
      key !== "index" &&
      key !== "id" &&
      key !== "type" &&
      key !== "function"
    ) {
      target[key] =
        toolCall[key];
    }
  }
}

function addKimiSSEToAccumulator(
  accumulator,
  event
) {
  if (
    !event ||
    !event.trim()
  ) {
    return;
  }

  const lines =
    event.split(/\r?\n/);

  for (
    const line of lines
  ) {
    if (
      !line.startsWith("data:")
    ) {
      continue;
    }

    const data =
      line.slice(5).trim();

    if (
      !data ||
      data === "[DONE]"
    ) {
      continue;
    }

    let parsed;

    try {
      parsed =
        JSON.parse(data);
    } catch {
      continue;
    }

    const delta =
      parsed?.choices?.[0]?.delta;

    if (!delta) {
      continue;
    }

    if (delta.role) {
      accumulator.role =
        delta.role;
    }

    if (
      typeof delta.content ===
      "string"
    ) {
      accumulator.contentParts.push(
        delta.content
      );
    }

    if (
      typeof delta.reasoning_content ===
      "string"
    ) {
      accumulator.reasoningParts.push(
        delta.reasoning_content
      );
    }

    if (
      typeof delta.reasoning ===
      "string"
    ) {
      accumulator.reasoningParts.push(
        delta.reasoning
      );
    }

    if (
      Array.isArray(
        delta.tool_calls
      )
    ) {
      for (
        const toolCall of
          delta.tool_calls
      ) {
        mergeKimiToolCallDelta(
          accumulator,
          toolCall
        );
      }
    }

    if (
      delta.function_call
    ) {
      if (
        !accumulator.functionCall
      ) {
        accumulator.functionCall = {};
      }

      if (
        delta.function_call.name !==
        undefined
      ) {
        accumulator.functionCall.name =
          delta.function_call.name;
      }

      if (
        delta.function_call.arguments !==
        undefined
      ) {
        accumulator.functionCall.arguments =
          (
            accumulator.functionCall
              .arguments || ""
          ) +
          delta.function_call.arguments;
      }
    }

    if (
      delta.name !== undefined
    ) {
      accumulator.name =
        delta.name;
    }
  }
}

function finalizeKimiStreamAccumulator(
  accumulator
) {
  const message = {
    role:
      accumulator.role ||
      "assistant",

    content:
      accumulator.contentParts.join("")
  };

  const reasoning =
    accumulator.reasoningParts.join("");

  if (reasoning) {
    message.reasoning_content =
      reasoning;
  }

  const toolCalls =
    accumulator.toolCalls.filter(
      toolCall =>
        Object.keys(toolCall)
          .length > 0
    );

  if (toolCalls.length) {
    message.tool_calls =
      toolCalls;
  }

  if (accumulator.functionCall) {
    message.function_call =
      accumulator.functionCall;
  }

  if (
    accumulator.name !==
    undefined
  ) {
    message.name =
      accumulator.name;
  }

  return message;
}

/* ============================================================
   NVIDIA MODEL CACHE
============================================================ */

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

async function fetchNimModels(
  forceRefresh = false
) {
  const refresh =
    forceRefresh === true;

  const now = Date.now();

  if (
    !refresh &&
    Array.isArray(modelCache) &&
    now - modelCacheTimestamp <
      MODEL_CACHE_TTL
  ) {
    return modelCache;
  }

  /*
   * Prevent multiple simultaneous refreshes.
   */
  if (
    modelCachePromise &&
    !refresh
  ) {
    return modelCachePromise;
  }

  modelCachePromise =
    (async () => {
      try {
        if (!NIM_API_KEY) {
          return Array.isArray(
            modelCache
          )
            ? modelCache
            : [];
        }

        const response =
          await axios.get(
            NIM_API_BASE +
              "/models",
            {
              headers: {
                Authorization:
                  "Bearer " +
                  NIM_API_KEY,

                Accept:
                  "application/json"
              },

              timeout:
                Math.min(
                  NIM_TIMEOUT,
                  30000
                ),

              httpAgent,
              httpsAgent,

              validateStatus:
                () => true
            }
          );

        if (
          response.status >= 200 &&
          response.status < 300 &&
          Array.isArray(
            response.data?.data
          )
        ) {
          modelCache =
            response.data.data;

          modelCacheTimestamp =
            Date.now();

          return modelCache;
        }

        return Array.isArray(
          modelCache
        )
          ? modelCache
          : [];
      } catch (error) {
        console.warn(
          "Unable to refresh NVIDIA model list:",
          error?.message ||
            error
        );

        return Array.isArray(
          modelCache
        )
          ? modelCache
          : [];
      } finally {
        modelCachePromise =
          null;
      }
    })();

  return modelCachePromise;
}

/* ============================================================
   MESSAGE NORMALIZATION
============================================================ */

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  const result = [];

  for (
    const message of messages
  ) {
    if (
      !isPlainObject(message)
    ) {
      continue;
    }

    if (
      typeof message.role !==
        "string" ||
      !message.role.trim()
    ) {
      continue;
    }

    if (
      (
        message.content ===
          undefined ||
        message.content === null
      ) &&
      !Array.isArray(
        message.tool_calls
      ) &&
      message.reasoning_content ===
        undefined &&
      message.reasoning ===
        undefined
    ) {
      continue;
    }

    result.push({
      ...message,
      role:
        message.role.trim()
    });
  }

  return result;
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
  const modelConfig =
    getModelConfig(model);

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

  if (
    isPlainObject(
      incoming.chat_template_kwargs
    )
  ) {
    request.chat_template_kwargs = {
      ...incoming.chat_template_kwargs
    };
  }

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
   * Unknown models preserve the incoming
   * reasoning parameters.
   */
  if (!modelConfig) {
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

  switch (model) {
    case "deepseek-ai/deepseek-v4-flash-0731": {
      request.reasoning_effort =
        reasoningEffort;

      request.chat_template_kwargs = {
        ...(request.chat_template_kwargs ||
          {}),

        thinking:
          reasoningEffort !== "none",

        reasoning_effort:
          reasoningEffort
      };

      break;
    }

    case "deepseek-ai/deepseek-v4-pro-0813":
    case "moonshotai/kimi-k3": {
      request.reasoning_effort =
        reasoningEffort;

      break;
    }

    case "nvidia/nemotron-3-ultra-550b-a55b": {
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
        delete request
          .reasoning_budget;
      }

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

      break;
    }
  }

  return request;
}

/* ============================================================
   RESPONSE REASONING REMOVAL
============================================================ */

function stripReasoning(parsed) {
  if (
    !parsed ||
    typeof parsed !==
      "object" ||
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

    if (choice.delta) {
      delete choice.delta
        .reasoning_content;

      delete choice.delta
        .reasoning;

      delete choice.delta
        .thinking;
    }

    if (choice.message) {
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
   SSE PROCESSING
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

    if (data === "[DONE]") {
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
    } catch {
      /*
       * Preserve malformed/non-JSON SSE
       * instead of destroying the stream.
       */
      output.push(line);
    }
  }

  return output.length
    ? output.join("\n") +
        "\n\n"
    : "";
}

/* ============================================================
   ERROR HANDLING
============================================================ */

function readStream(stream) {
  return new Promise(resolve => {
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
      chunk => {
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
  });
}

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
      JSON.parse(data);

    return (
      parsed?.error?.message ||
      parsed?.message ||
      null
    );
  } catch {
    return String(data);
  }
}

function sendError(
  res,
  status,
  message,
  details
) {
  if (res.headersSent) {
    try {
      res.end();
    } catch {}

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
   EXPRESS
============================================================ */

app.disable(
  "x-powered-by"
);

app.use(cors());

app.use(
  express.json({
    limit: "100mb"
  })
);

/* ============================================================
   ROOT
============================================================ */

app.get(
  "/",
  function (req, res) {
    res.json({
      status: "online",

      service:
        "JanitorAI -> NVIDIA NIM Proxy",

      default_model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      upstream_model_count:
        Array.isArray(modelCache)
          ? modelCache.length
          : 0,

      unknown_models_allowed:
        ALLOW_UNKNOWN_MODELS,

      default_reasoning_effort:
        DEFAULT_REASONING_EFFORT,

      default_reasoning_budget:
        DEFAULT_REASONING_BUDGET,

      nim_api_base:
        NIM_API_BASE,

      timeout_ms:
        NIM_TIMEOUT,

      kimi_memory:
        KIMI_MEMORY_ENABLED,

      kimi_context_budget:
        KIMI_CONTEXT_BUDGET,

      kimi_active_conversations:
        kimiConversationStore.size
    });
  }
);

/* ============================================================
   HEALTH
============================================================ */

app.get(
  "/health",
  function (req, res) {
    res.json({
      ok: true,

      model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      upstream_model_count:
        Array.isArray(modelCache)
          ? modelCache.length
          : 0,

      unknown_models_allowed:
        ALLOW_UNKNOWN_MODELS,

      reasoning:
        DEFAULT_REASONING_EFFORT,

      max_tokens:
        DEFAULT_MAX_TOKENS,

      kimi_memory:
        KIMI_MEMORY_ENABLED,

      kimi_context_budget:
        KIMI_CONTEXT_BUDGET,

      kimi_active_conversations:
        kimiConversationStore.size
    });
  }
);

/* ============================================================
   NVIDIA MODEL LIST
============================================================ */

app.get(
  "/v1/models",
  async function (req, res) {
    try {
      const upstreamModels =
        await fetchNimModels();

      if (
        upstreamModels.length > 0
      ) {
        return res.json({
          object: "list",
          data: upstreamModels
        });
      }

      const models =
        Object.entries(
          MODELS
        ).map(
          ([id, config]) => ({
            id,
            object: "model",

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
        object: "list",
        data: models
      });
    } catch (error) {
      return sendError(
        res,
        502,
        "Unable to retrieve NVIDIA NIM models.",
        error?.message ||
          String(error)
      );
    }
  }
);

/* ============================================================
   FORCE MODEL CACHE REFRESH
============================================================ */

app.post(
  "/v1/models/refresh",
  async function (req, res) {
    try {
      const models =
        await fetchNimModels(
          true
        );

      return res.json({
        object: "list",
        data: models,
        count: models.length
      });
    } catch (error) {
      return sendError(
        res,
        502,
        "Unable to refresh NVIDIA NIM model list.",
        error?.message ||
          String(error)
      );
    }
  }
);

/* ============================================================
   CHAT COMPLETIONS
============================================================ */

app.post(
  "/v1/chat/completions",
  async function (req, res) {
    let kimiConversationId =
      null;

    let kimiStreamAccumulator =
      null;

    try {
      if (!NIM_API_KEY) {
        return sendError(
          res,
          500,
          "NIM_API_KEY is not configured."
        );
      }

      const incoming =
        isPlainObject(req.body)
          ? req.body
          : {};

      const requestedModel =
        typeof incoming.model ===
          "string" &&
        incoming.model.trim()
          ? incoming.model.trim()
          : DEFAULT_MODEL;

      const model =
        requestedModel;

      const normalizedModel =
        normalizeModel(model);

      const modelConfig =
        getModelConfig(model);

      if (
        !isSupportedModel(model)
      ) {
        return sendError(
          res,
          400,
          "Unsupported model: " +
            model,
          {
            explicitly_configured_models:
              Object.keys(MODELS)
          }
        );
      }

      let messages =
        normalizeMessages(
          incoming.messages
        );

      if (
        !messages.length
      ) {
        return sendError(
          res,
          400,
          "No valid messages were supplied."
        );
      }

      /* ======================================================
         KIMI MEMORY
      ====================================================== */

      if (
        normalizedModel ===
          "moonshotai/kimi-k3" &&
        KIMI_MEMORY_ENABLED
      ) {
        const kimiContext =
          buildKimiMemoryContext(
            req,
            incoming,
            messages
          );

        kimiConversationId =
          kimiContext.conversationId;

        messages =
          kimiContext.messages;

        if (DEBUG_PROXY) {
          console.log(
            "KIMI MEMORY:",
            kimiConversationId,
            "messages:",
            messages.length,
            "estimated tokens:",
            estimateKimiMessagesTokens(
              messages
            )
          );
        }
      }

      /* ======================================================
         STREAM
      ====================================================== */

      const stream =
        parseBoolean(
          incoming.stream,
          true
        );

      const nimRequest =
        buildNimRequest(
          incoming,
          model,
          messages,
          stream
        );

      if (DEBUG_PROXY) {
        console.log(
          "NIM REQUEST:",
          JSON.stringify(
            nimRequest
          )
        );
      }

      /* ======================================================
         NVIDIA REQUEST
      ====================================================== */

      const axiosConfig = {
        headers: {
          Authorization:
            "Bearer " +
            NIM_API_KEY,

          "Content-Type":
            "application/json",

          Accept:
            stream
              ? "text/event-stream"
              : "application/json"
        },

        timeout:
          NIM_TIMEOUT,

        httpAgent,
        httpsAgent,

        validateStatus:
          () => true
      };

      const response =
        await axios.post(
          NIM_API_BASE +
            "/chat/completions",

          nimRequest,

          stream
            ? {
                ...axiosConfig,
                responseType:
                  "stream"
              }
            : axiosConfig
        );

      /* ======================================================
         UPSTREAM ERROR
      ====================================================== */

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
          } catch {
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
          "NVIDIA NIM ERROR",
          normalizedModel,
          response.status,
          upstreamMessage
        );

        /*
         * Refresh model cache asynchronously.
         * Don't make the user wait for it.
         */
        if (
          response.status ===
            400 ||
          response.status ===
            404
        ) {
          fetchNimModels(
            true
          ).catch(() => {});
        }

        const proxyStatus =
          response.status >= 500
            ? 502
            : response.status;

        return sendError(
          res,
          proxyStatus,
          upstreamMessage ||
            "NVIDIA NIM returned HTTP " +
              response.status +
              ".",
          {
            upstream_status:
              response.status,

            upstream_body:
              errorBody
          }
        );
      }

      /* ======================================================
         NON-STREAMING RESPONSE
      ====================================================== */

      if (!stream) {
        if (
          normalizedModel ===
            "moonshotai/kimi-k3" &&
          kimiConversationId &&
          KIMI_MEMORY_ENABLED
        ) {
          const assistantMessage =
            extractKimiAssistantMessage(
              response.data
            );

          if (
            assistantMessage
          ) {
            appendKimiAssistantMessage(
              kimiConversationId,
              assistantMessage
            );
          }
        }

        return res
          .status(200)
          .json(
            stripReasoning(
              response.data
            )
          );
      }

      /* ======================================================
         STREAMING RESPONSE
      ====================================================== */

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

      if (
        normalizedModel ===
          "moonshotai/kimi-k3" &&
        kimiConversationId &&
        KIMI_MEMORY_ENABLED
      ) {
        kimiStreamAccumulator =
          createKimiStreamAccumulator();
      }

      let buffer = "";
      let ended = false;
      let clientDisconnected =
        false;

      function destroyUpstream() {
        try {
          upstream.destroy();
        } catch {}
      }

      function disconnect() {
        if (
          clientDisconnected
        ) {
          return;
        }

        clientDisconnected =
          true;

        destroyUpstream();
      }

      req.on(
        "aborted",
        disconnect
      );

      res.on(
        "close",
        function () {
          if (
            !res.writableEnded
          ) {
            disconnect();
          }
        }
      );

      upstream.on(
        "data",
        function (chunk) {
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

          let separatorIndex;

          while (
            (
              separatorIndex =
                buffer.search(
                  /\r?\n\r?\n/
                )
            ) !== -1
          ) {
            const event =
              buffer.slice(
                0,
                separatorIndex
              );

            /*
             * Determine whether separator
             * was \n\n or \r\n\r\n.
             */
            const separatorLength =
              buffer[
                separatorIndex
              ] === "\r"
                ? 4
                : 2;

            buffer =
              buffer.slice(
                separatorIndex +
                  separatorLength
              );

            /*
             * Accumulate Kimi before stripping
             * reasoning from what the client sees.
             */
            if (
              kimiStreamAccumulator
            ) {
              addKimiSSEToAccumulator(
                kimiStreamAccumulator,
                event
              );
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
            } catch {
              disconnect();
              break;
            }
          }
        }
      );

      upstream.on(
        "end",
        function () {
          if (ended) {
            return;
          }

          ended = true;

          /*
           * Process final partial event.
           */
          if (
            buffer.trim() &&
            !clientDisconnected
          ) {
            if (
              kimiStreamAccumulator
            ) {
              addKimiSSEToAccumulator(
                kimiStreamAccumulator,
                buffer
              );
            }

            const output =
              processSSEEvent(
                buffer
              );

            if (output) {
              try {
                res.write(
                  output
                );
              } catch {}
            }
          }

          /*
           * Save complete Kimi assistant
           * response after stream completion.
           */
          if (
            kimiStreamAccumulator &&
            kimiConversationId &&
            KIMI_MEMORY_ENABLED
          ) {
            appendKimiAssistantMessage(
              kimiConversationId,
              finalizeKimiStreamAccumulator(
                kimiStreamAccumulator
              )
            );
          }

          if (
            !clientDisconnected
          ) {
            try {
              res.end();
            } catch {}
          }
        }
      );

      upstream.on(
        "error",
        function (error) {
          if (ended) {
            return;
          }

          ended = true;

          console.error(
            "NVIDIA stream error:",
            error?.message ||
              error
          );

          if (
            !res.headersSent
          ) {
            return sendError(
              res,
              502,
              "NVIDIA streaming connection failed.",
              error?.message ||
                String(error)
            );
          }

          try {
            res.end();
          } catch {}
        }
      );
    } catch (error) {
      console.error(
        "PROXY ERROR:",
        error?.stack ||
          error?.message ||
          error
      );

      if (
        res.headersSent
      ) {
        try {
          res.end();
        } catch {}

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
  function (req, res) {
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
  function (
    error,
    req,
    res,
    next
  ) {
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
    function () {
      console.log(
        "JanitorAI -> NVIDIA NIM Proxy listening on 0.0.0.0:" +
          PORT
      );

      console.log(
        "Default model:",
        DEFAULT_MODEL
      );

     /* console.log(
        "Kimi memory:",
        KIMI_MEMORY_ENABLED
      ); */

      console.log(
        "NIM endpoint:",
        NIM_API_BASE
      );

      fetchNimModels()
        .then(models => {
          console.log(
            "NVIDIA reports",
            models.length,
            "available model(s)."
          );
        })
        .catch(error => {
          console.warn(
            "Initial NVIDIA model discovery failed:",
            error?.message ||
              error
          );
        });
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
   GRACEFUL SHUTDOWN
============================================================ */

function shutdown(signal) {
  console.log(
    signal +
      " received. Shutting down..."
  );

  server.close(
    function () {
      console.log(
        "Server closed."
      );

      process.exit(0);
    }
  );

  setTimeout(
    function () {
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
