```js
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const crypto = require("crypto");

const app = express();

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

const MODELS = {
  "deepseek-ai/deepseek-v4-flash-0731": {
    name: "DeepSeek V4 Flash 0731",
    provider: "DeepSeek AI",
    reasoningLevels: ["none", "low", "high", "max"],
    defaultReasoningEffort: "high",
    maxOutputTokens: 16384,
    thinkingParameter: "chat_template"
  },

  "deepseek-ai/deepseek-v4-pro-0813": {
    name: "DeepSeek V4 Pro 0813",
    provider: "DeepSeek AI",
    reasoningLevels: ["none", "low", "high", "max"],
    defaultReasoningEffort: "max",
    maxOutputTokens: 16384,
    thinkingParameter: "chat_template"
  },

  "moonshotai/kimi-k3": {
    name: "Kimi K3",
    provider: "Moonshot AI",
    reasoningLevels: ["low", "high", "max"],
    defaultReasoningEffort: "max",
    maxOutputTokens: 1048576,
    thinkingParameter: "native"
  },

  "nvidia/nemotron-3-ultra-550b-a55b": {
    name: "NVIDIA Nemotron 3 Ultra 550B",
    provider: "NVIDIA",
    reasoningLevels: ["none", "medium", "high"],
    defaultReasoningEffort: "high",
    maxOutputTokens: 32768,
    thinkingParameter: "nemotron"
  }
};

const ALLOW_UNKNOWN_MODELS =
  String(process.env.ALLOW_UNKNOWN_MODELS || "true")
    .trim()
    .toLowerCase() === "true";

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
  "deepseek-ai/deepseek-v4-flash-0731": "high",
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

const NIM_TIMEOUT =
  Number.isFinite(Number(process.env.NIM_TIMEOUT_MS))
    ? Number(process.env.NIM_TIMEOUT_MS)
    : 900000;

const MODEL_CACHE_TTL =
  Number.isFinite(Number(process.env.MODEL_CACHE_TTL_MS))
    ? Number(process.env.MODEL_CACHE_TTL_MS)
    : 300000;

const MAX_ERROR_BODY_SIZE = 2 * 1024 * 1024;

const DEBUG_PROXY =
  String(process.env.DEBUG_PROXY || "false")
    .trim()
    .toLowerCase() === "true";

const KIMI_MEMORY_ENABLED =
  String(process.env.KIMI_MEMORY_ENABLED || "true")
    .trim()
    .toLowerCase() === "true";

/*
 * NVIDIA currently documents 1,048,576 input tokens for K3.
 *
 * Default:
 *
 *     1,000,000
 *
 * You can lower this through Render environment variables if
 * necessary.
 */
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

/*
 * Approximate token calculation.
 *
 * This is deliberately conservative and is NOT intended to
 * replace Kimi/NVIDIA's actual tokenizer.
 *
 * 3.5 characters/token is a practical approximation for mixed
 * conversational English text.
 */
const KIMI_CHARS_PER_TOKEN =
  Number.isFinite(Number(process.env.KIMI_CHARS_PER_TOKEN))
    ? Math.max(2, Number(process.env.KIMI_CHARS_PER_TOKEN))
    : 3.5;

/*
 * Maximum number of messages kept in the server memory.
 *
 * The token budget is the real context limiter.
 *
 * This simply prevents pathological numbers of tiny messages
 * from consuming unlimited Node.js memory.
 */
const KIMI_MAX_STORED_MESSAGES =
  Number.isFinite(Number(process.env.KIMI_MAX_STORED_MESSAGES))
    ? Math.max(
        100,
        Math.floor(Number(process.env.KIMI_MAX_STORED_MESSAGES))
      )
    : 10000;

/*
 * Remove completely inactive conversations after this period.
 *
 * Default: 7 days.
 */
const KIMI_MEMORY_TTL_MS =
  Number.isFinite(Number(process.env.KIMI_MEMORY_TTL_MS))
    ? Math.max(60000, Number(process.env.KIMI_MEMORY_TTL_MS))
    : 7 * 24 * 60 * 60 * 1000;

/*
 * Maximum simultaneous conversations held in RAM.
 */
const KIMI_MAX_CONVERSATIONS =
  Number.isFinite(Number(process.env.KIMI_MAX_CONVERSATIONS))
    ? Math.max(
        10,
        Math.floor(Number(process.env.KIMI_MAX_CONVERSATIONS))
      )
    : 2000;

/*
 * Map:
 *
 * conversationId -> {
 *   messages: [],
 *   updatedAt,
 *   systemFingerprint,
 *   firstUserFingerprint
 * }
 */
const kimiConversationStore = new Map();

/*
 * ============================================================
 * KIMI TOKEN ESTIMATION
 * ============================================================
 */

function estimateKimiTokens(value) {
  if (value === undefined || value === null) {
    return 0;
  }

  if (typeof value === "string") {
    return Math.ceil(value.length / KIMI_CHARS_PER_TOKEN);
  }

  try {
    return Math.ceil(
      JSON.stringify(value).length / KIMI_CHARS_PER_TOKEN
    );
  } catch (error) {
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
  return messages.reduce(
    (total, message) =>
      total + estimateKimiMessageTokens(message),
    0
  );
}

/*
 * ============================================================
 * KIMI MEMORY HASHING
 * ============================================================
 */

function hashKimiValue(value) {
  return crypto
    .createHash("sha256")
    .update(String(value || ""), "utf8")
    .digest("hex");
}

/*
 * ============================================================
 * KIMI CONVERSATION ID
 * ============================================================
 *
 * Best case:
 *
 * JanitorAI/client gives us a conversation/session ID.
 *
 * We accept several common names.
 *
 * If no ID is supplied, we derive one from stable conversation
 * information and also use overlap matching below.
 */

function getExplicitKimiConversationId(req, incoming) {
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
      candidate.trim() !== ""
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

function getKimiSystemMessages(messages) {
  return messages.filter(
    (message) =>
      message?.role === "system" ||
      message?.role === "developer"
  );
}

function getKimiUserMessages(messages) {
  return messages.filter(
    (message) => message?.role === "user"
  );
}

function getKimiStableFingerprint(messages) {
  const systemMessages =
    getKimiSystemMessages(messages);

  const userMessages =
    getKimiUserMessages(messages);

  const systemFingerprint = hashKimiValue(
    JSON.stringify(
      systemMessages.slice(0, 3).map((message) => ({
        role: message.role,
        content: message.content
      }))
    )
  );

  const firstUser = userMessages[0];

  const firstUserFingerprint = firstUser
    ? hashKimiValue(
        JSON.stringify({
          role: firstUser.role,
          content: firstUser.content
        })
      )
    : null;

  return {
    systemFingerprint,
    firstUserFingerprint
  };
}

/*
 * ============================================================
 * KIMI MESSAGE SIGNATURE
 * ============================================================
 *
 * Includes reasoning_content and tool calls.
 *
 * This matters because NVIDIA specifically says K3 multi-turn
 * conversations/tool calls should preserve the complete prior
 * assistant message.
 */

function kimiMessageSignature(message) {
  return hashKimiValue(
    JSON.stringify({
      role: message?.role || "",
      content: message?.content ?? null,
      reasoning_content:
        message?.reasoning_content ?? null,
      name: message?.name ?? null,
      tool_calls: message?.tool_calls ?? null,
      function_call: message?.function_call ?? null
    })
  );
}

/*
 * ============================================================
 * KIMI MEMORY TOUCH / CLEANUP
 * ============================================================
 */

function touchKimiConversation(conversationId) {
  const entry =
    kimiConversationStore.get(conversationId);

  if (!entry) {
    return;
  }

  entry.updatedAt = Date.now();

  /*
   * Reinsert so Map iteration order makes the oldest entry
   * the least recently used entry.
   */
  kimiConversationStore.delete(conversationId);
  kimiConversationStore.set(conversationId, entry);
}

function cleanupKimiMemory() {
  const cutoff =
    Date.now() - KIMI_MEMORY_TTL_MS;

  for (const [conversationId, entry] of kimiConversationStore) {
    if (entry.updatedAt < cutoff) {
      kimiConversationStore.delete(conversationId);
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

/*
 * ============================================================
 * KIMI CONVERSATION LOOKUP
 * ============================================================
 */

function createKimiConversation(
  conversationId,
  messages
) {
  const fingerprints =
    getKimiStableFingerprint(messages);

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

  /*
   * LRU cleanup.
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

    if (oldestKey === undefined) {
      break;
    }

    kimiConversationStore.delete(oldestKey);
  }

  return entry;
}

/*
 * If an explicit ID isn't available, try to identify an existing
 * conversation from messages already known to the server.
 *
 * This is intentionally conservative.
 *
 * It prefers exact first-user matches, then system-prompt
 * matches with message overlap.
 */

function findKimiConversationByHistory(messages) {
  if (
    !Array.isArray(messages) ||
    messages.length === 0
  ) {
    return null;
  }

  const fingerprints =
    getKimiStableFingerprint(messages);

  let bestMatch = null;
  let bestScore = 0;

  const incomingSignatures = new Set(
    messages.map(kimiMessageSignature)
  );

  for (
    const [conversationId, entry] of
    kimiConversationStore
  ) {
    let score = 0;

    /*
     * Strongest signal:
     * same first user message.
     */
    if (
      fingerprints.firstUserFingerprint &&
      entry.firstUserFingerprint &&
      fingerprints.firstUserFingerprint ===
        entry.firstUserFingerprint
    ) {
      score += 1000;
    }

    /*
     * System prompt match is useful for character chats.
     */
    if (
      fingerprints.systemFingerprint &&
      entry.systemFingerprint &&
      fingerprints.systemFingerprint ===
        entry.systemFingerprint
    ) {
      score += 100;
    }

    /*
     * Message overlap is the most useful fallback when
     * JanitorAI sends a shortened recent history.
     */
    if (Array.isArray(entry.messages)) {
      let overlap = 0;

      for (const message of entry.messages) {
        if (
          incomingSignatures.has(
            kimiMessageSignature(message)
          )
        ) {
          overlap++;
        }
      }

      score += Math.min(overlap * 10, 500);
    }

    /*
     * Don't match completely unrelated conversations.
     */
    if (score > bestScore) {
      bestScore = score;

      bestMatch = {
        conversationId,
        entry
      };
    }
  }

  /*
   * Require an actual meaningful match.
   */
  if (!bestMatch || bestScore < 10) {
    return null;
  }

  return bestMatch;
}

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
      `explicit:${hashKimiValue(explicitId)}`;

    let entry =
      kimiConversationStore.get(
        conversationId
      );

    if (!entry) {
      entry = createKimiConversation(
        conversationId,
        messages
      );
    }

    touchKimiConversation(conversationId);

    return {
      conversationId,
      entry
    };
  }

  /*
   * No explicit ID.
   *
   * Try to reconnect the request to a conversation that already
   * exists in RAM.
   */
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

  /*
   * Last resort:
   * create a new conversation.
   *
   * This is only used when the request contains no usable
   * conversation identifier and no previous history overlaps.
   */
  const fingerprints =
    getKimiStableFingerprint(messages);

  const seed = JSON.stringify({
    system: fingerprints.systemFingerprint,
    firstUser: fingerprints.firstUserFingerprint,
    time: Date.now(),
    random: Math.random()
  });

  const conversationId =
    `derived:${hashKimiValue(seed)}`;

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

/*
 * ============================================================
 * KIMI MESSAGE MERGING
 * ============================================================
 *
 * JanitorAI may send only part of the conversation.
 *
 * We merge:
 *
 *     server memory
 *          +
 *     JanitorAI's incoming messages
 *
 * without duplicating messages.
 */

function mergeKimiMessages(
  storedMessages,
  incomingMessages
) {
  const result = [];
  const seen = new Set();

  const addMessage = (message) => {
    const signature =
      kimiMessageSignature(message);

    if (seen.has(signature)) {
      return;
    }

    seen.add(signature);
    result.push(message);
  };

  for (const message of storedMessages) {
    addMessage(message);
  }

  for (const message of incomingMessages) {
    addMessage(message);
  }

  return result;
}

/*
 * ============================================================
 * KIMI CONTEXT FITTING
 * ============================================================
 *
 * System/developer messages are always preserved.
 *
 * For the remaining messages, take the newest messages that fit
 * the context budget.
 *
 * Because K3 supports a 1M input context, this allows vastly
 * more history than the 20-message example in the Kimi docs.
 */

function fitKimiContext(messages) {
  if (
    !Array.isArray(messages) ||
    messages.length === 0
  ) {
    return [];
  }

  const pinned = [];
  const normal = [];

  for (const message of messages) {
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
    estimateKimiMessagesTokens(pinned);

  /*
   * If system messages alone exceed the configured budget,
   * return them rather than corrupting them.
   */
  if (used >= KIMI_CONTEXT_BUDGET) {
    return pinned;
  }

  const selected = [];

  /*
   * Walk backward so the newest messages are preferred.
   */
  for (
    let index = normal.length - 1;
    index >= 0;
    index--
  ) {
    const message = normal[index];

    const cost =
      estimateKimiMessageTokens(message);

    if (
      used + cost >
      KIMI_CONTEXT_BUDGET
    ) {
      /*
       * Skip an oversized individual message rather than
       * abandoning the entire history.
       */
      continue;
    }

    selected.push(message);
    used += cost;
  }

  selected.reverse();

  return [
    ...pinned,
    ...selected
  ];
}

/*
 * ============================================================
 * KIMI MEMORY UPDATE
 * ============================================================
 */

function updateKimiMemory(
  conversationId,
  incomingMessages
) {
  const memory =
    kimiConversationStore.get(
      conversationId
    ) ||
    createKimiConversation(
      conversationId,
      incomingMessages
    );

  /*
   * Preserve everything the server already knows, then merge
   * the history JanitorAI just sent us.
   */
  const merged =
    mergeKimiMessages(
      memory.messages,
      incomingMessages
    );

  /*
   * Keep the maximum useful context that fits K3's budget.
   */
  memory.messages =
    fitKimiContext(merged).slice(
      -KIMI_MAX_STORED_MESSAGES
    );

  /*
   * Refresh stable fingerprints as the conversation grows.
   */
  const fingerprints =
    getKimiStableFingerprint(
      memory.messages
    );

  if (fingerprints.systemFingerprint) {
    memory.systemFingerprint =
      fingerprints.systemFingerprint;
  }

  if (fingerprints.firstUserFingerprint) {
    memory.firstUserFingerprint =
      fingerprints.firstUserFingerprint;
  }

  memory.updatedAt = Date.now();

  touchKimiConversation(
    conversationId
  );

  return memory.messages;
}

/*
 * ============================================================
 * KIMI MEMORY PREPARATION
 * ============================================================
 */

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

  const messages =
    updateKimiMemory(
      conversation.conversationId,
      incomingMessages
    );

  return {
    conversationId:
      conversation.conversationId,
    messages
  };
}

/*
 * ============================================================
 * STORE KIMI ASSISTANT RESPONSE
 * ============================================================
 *
 * This stores the COMPLETE assistant message internally.
 *
 * In particular:
 *
 *     content
 *     reasoning_content
 *     tool_calls
 *     function_call
 *
 * are retained.
 *
 * stripReasoning() is only applied to the response going back
 * to JanitorAI, NOT to this internal memory.
 */

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

  const cleanAssistantMessage = {
    role:
      assistantMessage.role ||
      "assistant"
  };

  if (
    assistantMessage.content !==
    undefined
  ) {
    cleanAssistantMessage.content =
      assistantMessage.content;
  }

  if (
    assistantMessage.reasoning_content !==
    undefined
  ) {
    cleanAssistantMessage.reasoning_content =
      assistantMessage.reasoning_content;
  }

  if (
    Array.isArray(
      assistantMessage.tool_calls
    )
  ) {
    cleanAssistantMessage.tool_calls =
      assistantMessage.tool_calls;
  }

  if (
    assistantMessage.function_call !==
    undefined
  ) {
    cleanAssistantMessage.function_call =
      assistantMessage.function_call;
  }

  if (
    assistantMessage.name !==
    undefined
  ) {
    cleanAssistantMessage.name =
      assistantMessage.name;
  }

  const signature =
    kimiMessageSignature(
      cleanAssistantMessage
    );

  const alreadyExists =
    memory.messages.some(
      (message) =>
        kimiMessageSignature(
          message
        ) === signature
    );

  if (!alreadyExists) {
    memory.messages.push(
      cleanAssistantMessage
    );
  }

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

/*
 * ============================================================
 * EXTRACT COMPLETE NON-STREAM KIMI ASSISTANT MESSAGE
 * ============================================================
 */

function extractKimiAssistantMessage(parsed) {
  const choice = parsed?.choices?.[0];

  if (
    !choice ||
    !choice.message
  ) {
    return null;
  }

  return {
    role:
      choice.message.role ||
      "assistant",

    content:
      choice.message.content,

    reasoning_content:
      choice.message.reasoning_content,

    tool_calls:
      choice.message.tool_calls,

    function_call:
      choice.message.function_call,

    name:
      choice.message.name
  };
}

/*
 * ============================================================
 * EXTRACT STREAMED KIMI ASSISTANT MESSAGE
 * ============================================================
 *
 * Streaming responses arrive as deltas.
 *
 * We reconstruct the assistant message from those deltas so
 * Kimi's reasoning history and tool calls can be preserved for
 * the next turn.
 */

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

  if (toolCall.id !== undefined) {
    target.id = toolCall.id;
  }

  if (toolCall.type !== undefined) {
    target.type = toolCall.type;
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

  /*
   * Preserve any provider-specific fields.
   */
  for (
    const [key, value] of
    Object.entries(toolCall)
  ) {
    if (
      key === "index" ||
      key === "id" ||
      key === "type" ||
      key === "function"
    ) {
      continue;
    }

    target[key] = value;
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

  for (const line of lines) {
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
      parsed = JSON.parse(data);
    } catch (error) {
      continue;
    }

    const choice =
      parsed?.choices?.[0];

    const delta =
      choice?.delta;

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

    /*
     * K3's reasoning history.
     */
    if (
      typeof delta.reasoning_content ===
      "string"
    ) {
      accumulator.reasoningParts.push(
        delta.reasoning_content
      );
    }

    /*
     * Some compatible servers may use "reasoning".
     */
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
            accumulator.functionCall.arguments ||
            ""
          ) +
          delta.function_call.arguments;
      }
    }

    if (
      delta.name !==
      undefined
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
      "assistant"
  };

  const content =
    accumulator.contentParts.join("");

  if (content) {
    message.content = content;
  } else {
    message.content = "";
  }

  const reasoning =
    accumulator.reasoningParts.join("");

  if (reasoning) {
    message.reasoning_content =
      reasoning;
  }

  const toolCalls =
    accumulator.toolCalls.filter(
      (toolCall) =>
        Object.keys(toolCall).length > 0
    );

  if (
    toolCalls.length >
    0
  ) {
    message.tool_calls =
      toolCalls;
  }

  if (
    accumulator.functionCall
  ) {
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

/*
 * ============================================================
 * NVIDIA MODEL CACHE
 * ============================================================
 */

let modelCache = null;
let modelCacheTimestamp = 0;
let modelCachePromise = null;

async function fetchNimModels(
  forceRefresh = false
) {
  const now = Date.now();

  if (
    !forceRefresh &&
    Array.isArray(modelCache) &&
    now - modelCacheTimestamp <
      MODEL_CACHE_TTL
  ) {
    return modelCache;
  }

  if (
    modelCachePromise &&
    !forceRefresh
  ) {
    return modelCachePromise;
  }

  modelCachePromise =
    (async () => {
      try {
        if (!NIM_API_KEY) {
          return Array.isArray(modelCache)
            ? modelCache
            : [];
        }

        const response =
          await axios.get(
            `${NIM_API_BASE}/models`,
            {
              headers: {
                Authorization:
                  `Bearer ${NIM_API_KEY}`,
                Accept:
                  "application/json"
              },

              timeout:
                Math.min(
                  NIM_TIMEOUT,
                  30000
                ),

              validateStatus:
                () => true
            }
          );

        if (
          response.status >= 200 &&
          response.status < 300 &&
          response.data &&
          Array.isArray(
            response.data.data
          )
        ) {
          modelCache =
            response.data.data;

          modelCacheTimestamp =
            Date.now();

          return modelCache;
        }

        console.warn(
          `NVIDIA /v1/models returned HTTP ${response.status}`
        );

        return Array.isArray(modelCache)
          ? modelCache
          : [];
      } catch (error) {
        console.warn(
          "Unable to refresh NVIDIA model list:",
          error?.message || error
        );

        return Array.isArray(modelCache)
          ? modelCache
          : [];
      } finally {
        modelCachePromise = null;
      }
    })();

  return modelCachePromise;
}

async function isKnownToNim(model) {
  const models =
    await fetchNimModels();

  const normalized =
    normalizeModel(model);

  return models.some(
    (item) =>
      normalizeModel(item?.id) ===
      normalized
  );
}

/*
 * ============================================================
 * EXPRESS
 * ============================================================
 */

app.disable("x-powered-by");

app.use(cors());

app.use(
  express.json({
    limit: "100mb"
  })
);

/*
 * ============================================================
 * HELPERS
 * ============================================================
 */

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
  return isFiniteNumber(value)
    ? Number(value)
    : fallback;
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

function getModelConfig(model) {
  return (
    MODELS[
      normalizeModel(model)
    ] || null
  );
}

function isSupportedModel(model) {
  if (getModelConfig(model)) {
    return true;
  }

  return ALLOW_UNKNOWN_MODELS;
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

function clampMaxTokens(
  value,
  modelConfig
) {
  const number =
    numberOrDefault(
      value,
      DEFAULT_MAX_TOKENS
    );

  if (
    modelConfig?.maxOutputTokens
  ) {
    return Math.min(
      modelConfig.maxOutputTokens,
      Math.max(
        1,
        Math.floor(number)
      )
    );
  }

  return Math.max(
    1,
    Math.floor(number)
  );
}

function clampReasoningBudget(value) {
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

function budgetToReasoningEffort(
  budget,
  modelConfig
) {
  const value =
    clampReasoningBudget(budget);

  const levels =
    modelConfig?.reasoningLevels ||
    [];

  if (
    levels.includes("none") &&
    value <= 0
  ) {
    return "none";
  }

  if (
    levels.includes("low") &&
    value <= 8192
  ) {
    return "low";
  }

  if (
    levels.includes("medium") &&
    value <= 16384
  ) {
    return "medium";
  }

  if (
    levels.includes("high") &&
    value <= 24576
  ) {
    return "high";
  }

  if (
    levels.includes("max")
  ) {
    return "max";
  }

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

  if (typeof value === "string") {
    const normalized =
      value.trim().toLowerCase();

    if (
      levels.includes(normalized)
    ) {
      return normalized;
    }

    if (
      normalized === "off" ||
      normalized === "false" ||
      normalized === "0"
    ) {
      if (
        levels.includes("none")
      ) {
        return "none";
      }
    }

    if (
      normalized === "med" ||
      normalized === "medium"
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

function getRequestedReasoningEffort(
  incoming,
  model,
  modelConfig
) {
  if (
    incoming.reasoning_effort !==
    undefined
  ) {
    return modelConfig
      ? normalizeReasoningEffort(
          incoming.reasoning_effort,
          modelConfig
        )
      : String(
          incoming.reasoning_effort
        ).trim();
  }

  if (
    incoming.reasoning_mode !==
    undefined
  ) {
    return modelConfig
      ? normalizeReasoningEffort(
          incoming.reasoning_mode,
          modelConfig
        )
      : String(
          incoming.reasoning_mode
        ).trim();
  }

  if (
    incoming.reasoning_budget !==
    undefined
  ) {
    return modelConfig
      ? budgetToReasoningEffort(
          incoming.reasoning_budget,
          modelConfig
        )
      : null;
  }

  if (
    MODEL_REASONING_EFFORTS[model]
  ) {
    return normalizeReasoningEffort(
      MODEL_REASONING_EFFORTS[model],
      modelConfig
    );
  }

  if (modelConfig) {
    return normalizeReasoningEffort(
      DEFAULT_REASONING_EFFORT,
      modelConfig
    );
  }

  return null;
}

/*
 * ============================================================
 * NORMALIZE MESSAGES
 * ============================================================
 */

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
        if (
          !Array.isArray(
            message.tool_calls
          )
        ) {
          /*
           * Preserve assistant messages that contain
           * reasoning/tool-call information.
           */
          if (
            message.reasoning_content ===
              undefined &&
            message.reasoning ===
              undefined
          ) {
            return false;
          }
        }
      }

      return true;
    })
    .map((message) => ({
      ...message,
      role: message.role.trim()
    }));
}

/*
 * ============================================================
 * BUILD NIM REQUEST
 * ============================================================
 */

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
    request.chat_template_kwargs =
      {
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

  /*
   * ==========================================================
   * DEEPSEEK V4 FLASH
   * ==========================================================
   */

  if (
    model ===
    "deepseek-ai/deepseek-v4-flash-0731"
  ) {
    request.reasoning_effort =
      reasoningEffort;

    request.chat_template_kwargs = {
      ...(request.chat_template_kwargs || {}),
      thinking:
        reasoningEffort !== "none",
      reasoning_effort:
        reasoningEffort
    };

    return request;
  }

  /*
   * ==========================================================
   * DEEPSEEK V4 PRO
   * ==========================================================
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
   * K3 uses top-level reasoning_effort.
   *
   * The memory itself is handled before this function is
   * called, so `messages` here already contains the expanded
   * server-side history.
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
   * NEMOTRON
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
      ...(request.chat_template_kwargs || {}),
      enable_thinking:
        reasoning !== "none"
    };

    if (
      reasoning === "medium"
    ) {
      request.chat_template_kwargs.medium_effort =
        true;
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
      delete request.reasoning_budget;
    }

    if (
      Array.isArray(
        request.tools
      ) &&
      request.tools.length > 0 &&
      reasoning !== "none"
    ) {
      request.chat_template_kwargs.force_nonempty_content =
        true;
    }

    delete request.nvext;
    delete request.reasoning_mode;

    return request;
  }

  return request;
}

/*
 * ============================================================
 * REMOVE REASONING FROM RESPONSE
 * ============================================================
 *
 * This only modifies what JanitorAI receives.
 *
 * Kimi's internal memory has already retained the full assistant
 * message before this function is used.
 */

function stripReasoning(parsed) {
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

  for (const choice of parsed.choices) {
    if (
      !choice ||
      typeof choice !== "object"
    ) {
      continue;
    }

    if (
      choice.delta &&
      typeof choice.delta === "object"
    ) {
      delete choice.delta.reasoning_content;
      delete choice.delta.reasoning;
      delete choice.delta.thinking;
    }

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

/*
 * ============================================================
 * SSE PROCESSOR
 * ============================================================
 */

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

  for (const line of lines) {
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

      stripReasoning(parsed);

      output.push(
        "data: " +
          JSON.stringify(parsed)
      );
    } catch (error) {
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

/*
 * ============================================================
 * READ STREAM
 * ============================================================
 */

function readStream(stream) {
  return new Promise((resolve) => {
    let output = "";
    let finished = false;

    const finish = () => {
      if (finished) {
        return;
      }

      finished = true;
      resolve(output);
    };

    stream.on("data", (chunk) => {
      if (
        output.length >=
        MAX_ERROR_BODY_SIZE
      ) {
        return;
      }

      output += chunk.toString("utf8");

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
    });

    stream.on("end", finish);
    stream.on("close", finish);
    stream.on("error", finish);
  });
}

/*
 * ============================================================
 * EXTRACT UPSTREAM ERROR
 * ============================================================
 */

function extractErrorMessage(data) {
  if (!data) {
    return null;
  }

  if (typeof data === "object") {
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
  } catch (error) {
    return String(data);
  }
}

/*
 * ============================================================
 * ERROR RESPONSE
 * ============================================================
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
    } catch (error) {}

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

/*
 * ============================================================
 * ROOT
 * ============================================================
 */

app.get(
  "/",
  async (req, res) => {
    const upstreamModels =
      await fetchNimModels();

    res.json({
      status: "online",

      service:
        "JanitorAI → NVIDIA NIM Proxy",

      default_model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      upstream_model_count:
        upstreamModels.length,

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

/*
 * ============================================================
 * HEALTH
 * ============================================================
 */

app.get(
  "/health",
  async (req, res) => {
    const upstreamModels =
      await fetchNimModels();

    res.json({
      ok: true,

      model:
        DEFAULT_MODEL,

      explicitly_configured_models:
        Object.keys(MODELS),

      upstream_model_count:
        upstreamModels.length,

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

/*
 * ============================================================
 * NVIDIA MODEL LIST
 * ============================================================
 */

app.get(
  "/v1/models",
  async (req, res) => {
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
        Object.entries(MODELS).map(
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

/*
 * ============================================================
 * FORCE MODEL CACHE REFRESH
 * ============================================================
 */

app.post(
  "/v1/models/refresh",
  async (req, res) => {
    try {
      const models =
        await fetchNimModels(true);

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

/*
 * ============================================================
 * CHAT COMPLETIONS
 * ============================================================
 */

app.post(
  "/v1/chat/completions",
  async (req, res) => {
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
        isPlainObject(req.body)
          ? req.body
          : {};

      /*
       * ======================================================
       * MODEL
       * ======================================================
       */

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

      /*
       * ======================================================
       * MODEL VALIDATION
       * ======================================================
       */

      if (
        !isSupportedModel(model)
      ) {
        return sendError(
          res,
          400,
          `Unsupported model: ${model}`,
          {
            explicitly_configured_models:
              Object.keys(MODELS)
          }
        );
      }

      /*
       * ======================================================
       * MESSAGES
       * ======================================================
       */

      let messages =
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
       * ======================================================
       * KIMI K3 MEMORY
       * ======================================================
       *
       * This is the ONLY point where the memory system changes
       * the request history.
       *
       * DeepSeek and Nemotron continue using exactly the
       * messages supplied by JanitorAI.
       */

      let kimiConversationId = null;
      let kimiStreamAccumulator = null;

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
            "========== KIMI MEMORY =========="
          );

          console.log(
            "Conversation:",
            kimiConversationId
          );

          console.log(
            "Messages:",
            messages.length
          );

          console.log(
            "Estimated input tokens:",
            estimateKimiMessagesTokens(
              messages
            )
          );

          console.log(
            "Context budget:",
            KIMI_CONTEXT_BUDGET
          );

          console.log(
            "=================================="
          );
        }
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
          model,
          messages,
          stream
        );

      /*
       * ======================================================
       * DEBUG
       * ======================================================
       */

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
                responseType: "stream"
              }
            : axiosConfig
        );

      /*
       * ======================================================
       * UPSTREAM ERROR
       * ======================================================
       */

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
          } catch (error) {
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

        if (
          response.status === 400 ||
          response.status === 404
        ) {
          await fetchNimModels(true);
        }

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

      /*
       * ======================================================
       * NON-STREAMING
       * ======================================================
       */

      if (!stream) {
        /*
         * IMPORTANT:
         *
         * Save the COMPLETE Kimi assistant message before
         * stripReasoning() removes internal reasoning from
         * the response.
         */

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

        const cleaned =
          stripReasoning(
            response.data
          );

        return res
          .status(200)
          .json(cleaned);
      }

      /*
       * ======================================================
       * STREAM HEADERS
       * ======================================================
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

      /*
       * Kimi streaming memory accumulator.
       */

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
      let clientDisconnected = false;

      /*
       * ======================================================
       * CLEAN UP UPSTREAM
       * ======================================================
       */

      const destroyUpstream = () => {
        try {
          if (
            upstream &&
            typeof upstream.destroy ===
              "function"
          ) {
            upstream.destroy();
          }
        } catch (error) {}
      };

      /*
       * ======================================================
       * CLIENT DISCONNECT
       * ======================================================
       */

      req.on(
        "aborted",
        () => {
          clientDisconnected = true;
          destroyUpstream();
        }
      );

      res.on(
        "close",
        () => {
          if (!res.writableEnded) {
            clientDisconnected = true;
            destroyUpstream();
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

            /*
             * Reconstruct Kimi's complete assistant message
             * BEFORE stripping reasoning from the outgoing
             * stream.
             */

            if (
              kimiStreamAccumulator
            ) {
              addKimiSSEToAccumulator(
                kimiStreamAccumulator,
                event
              );
            }

            /*
             * JanitorAI receives the normal cleaned stream.
             */

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

              clientDisconnected = true;
              destroyUpstream();
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
          if (ended) {
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
            /*
             * First capture it internally for Kimi.
             */

            if (
              kimiStreamAccumulator
            ) {
              addKimiSSEToAccumulator(
                kimiStreamAccumulator,
                buffer
              );
            }

            /*
             * Then send the cleaned version to JanitorAI.
             */

            const output =
              processSSEEvent(
                buffer
              );

            if (output) {
              try {
                res.write(output);
              } catch (error) {}
            }
          }

          /*
           * Save the COMPLETE Kimi assistant response.
           *
           * This happens even though reasoning_content was
           * removed from the outgoing stream.
           */

          if (
            kimiStreamAccumulator &&
            kimiConversationId &&
            KIMI_MEMORY_ENABLED
          ) {
            const assistantMessage =
              finalizeKimiStreamAccumulator(
                kimiStreamAccumulator
              );

            appendKimiAssistantMessage(
              kimiConversationId,
              assistantMessage
            );
          }

          if (!clientDisconnected) {
            try {
              res.end();
            } catch (error) {}
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
          if (ended) {
            return;
          }

          ended = true;

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
          } catch (endError) {}
        }
      );
    } catch (error) {
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

      if (res.headersSent) {
        try {
          res.end();
        } catch (endError) {}

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

/*
 * ============================================================
 * 404
 * ============================================================
 */

app.use(
  (req, res) => {
    return sendError(
      res,
      404,
      "Endpoint not found."
    );
  }
);

/*
 * ============================================================
 * EXPRESS ERROR HANDLER
 * ============================================================
 */

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

/*
 * ============================================================
 * START SERVER
 * ============================================================
 */

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
        `💭 Default reasoning budget: ${DEFAULT_REASONING_BUDGET}`
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
        `🌐 Unknown NIM models allowed: ${ALLOW_UNKNOWN_MODELS}`
      );

      console.log(
        `🧠 Kimi memory enabled: ${KIMI_MEMORY_ENABLED}`
      );

      console.log(
        `📚 Kimi context budget: ${KIMI_CONTEXT_BUDGET} estimated tokens`
      );

      console.log(
        `💾 Kimi memory conversations: ${kimiConversationStore.size}`
      );

      console.log(
        "📦 Explicitly configured models:"
      );

      Object.entries(MODELS).forEach(
        ([id, config]) => {
          console.log(
            `   - ${id}`
          );

          console.log(
            `     reasoning: ${config.reasoningLevels.join(", ")}`
          );

          console.log(
            `     max output: ${config.maxOutputTokens}`
          );
        }
      );

      console.log(
        "=========================================="
      );

      fetchNimModels()
        .then(
          (models) => {
            console.log(
              `📡 NVIDIA reports ${models.length} available model(s).`
            );
          }
        )
        .catch(
          (error) => {
            console.warn(
              "Initial NVIDIA model discovery failed:",
              error?.message ||
                error
            );
          }
        );
    }
  );

/*
 * ============================================================
 * LONG-RUNNING AI REQUEST SETTINGS
 * ============================================================
 */

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

/*
 * ============================================================
 * SHUTDOWN
 * ============================================================
 */

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
