// server.js - THE ULTIMATE CLEANER (BULLETPROOF STREAMING VERSION)
// ============================================================================
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ============================================================================
// 🔥 MAIN CONTROLS
// ============================================================================
const SHOW_REASONING = false; 
const ENABLE_THINKING_MODE = true; 
const DEEPSEEK_REASONING_MODE = "high";
// ============================================================================

function filterReasoning(text) {
  if (!text) return text;
  
  let cleanText = text;
  cleanText = cleanText.replace(/<think>[\s\S]*?<\/think>/gi, '');

  const garbagePhrases = [
    "\\*Okay, let me analyze", "\\*The scene:", "\\*The user wants me to",
    "\\*Current situation:", "\\*Key elements to include:", "\\*I need to describe:",
    "\\*Physical details to describe:",
    "\\*I need to avoid:", "\\*I should focus on:", "\\*The act of sliding", "\\*Sound integration:"
  ];

  garbagePhrases.forEach(phrase => {
    let regex = new RegExp(phrase + "[\\s\\S]*?\\n\\n", "gi");
    cleanText = cleanText.replace(regex, '');
  });

  cleanText = cleanText.replace(/\n- [\s\S]*?\n\n/gi, '\n\n');
  cleanText = cleanText.replace(/\d\. [\s\S]*?\n\n/gi, '\n\n');

  return cleanText.trim();
}

const MODEL_MAPPING = {
  'gpt-4o': 'google/gemma-4-31b-it',
  'claude-3-sonnet': 'z-ai/glm-5.2',
  'gemini-pro': 'z-ai/glm-5.1',
  'gemma-romance': 'minimaxai/minimax-m3',
  'claude-3-haiku-20240307': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o-latest': 'nvidia/nemotron-3-super-120b-a12b',
  'claude-3-opus-20240229': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4-0613': 'deepseek-ai/deepseek-v4-pro' 
};

app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    let isStream = stream || false; 

    let nimModel = MODEL_MAPPING[model] || model;
    
    const isGLM = nimModel.toLowerCase().includes('glm');
    const isDeepSeek = nimModel.toLowerCase().includes('deepseek');

    let sanitizedMessages = [];
    for (let m of messages) {
      if (!m.content || m.content.trim() === "") continue; 
      let role = m.role === 'system' ? 'user' : m.role; 
      
      if (sanitizedMessages.length > 0 && sanitizedMessages[sanitizedMessages.length - 1].role === role) {
        sanitizedMessages[sanitizedMessages.length - 1].content += "\n\n" + m.content;
      } else {
        sanitizedMessages.push({ role: role, content: m.content });
      }
    }

    if (ENABLE_THINKING_MODE && isGLM && sanitizedMessages.length > 0) {
      const thinkingPrompt = "\n\n[SYSTEM INSTRUCTION: Think deeply before answering. Use <think> tags for reasoning.]";
      sanitizedMessages[sanitizedMessages.length - 1].content += thinkingPrompt;
    }

    const nimRequest = {
      model: nimModel,
      messages: sanitizedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: isStream 
    };

    if (isDeepSeek) {
      nimRequest.reasoning_effort = DEEPSEEK_REASONING_MODE;
    }

    if (!isStream) {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.data.choices && response.data.choices[0].message) {
        let originalContent = response.data.choices[0].message.content;
        if (!SHOW_REASONING) {
          response.data.choices[0].message.content = filterReasoning(originalContent);
        }
      }
      return res.json(response.data);

    } else {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: 'stream'
      });

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      response.data.on('data', (chunk) => {
        res.write(chunk);
      });

      // 🟢 FIX 1: TAMBAH ERROR HANDLER SUPAYA SEBARANG SALURAN PUTUS TAK CRASHKAN PROXY
      response.data.on('error', (err) => {
        console.error('🔥 Stream data error:', err.message);
        res.end(); 
      });

      response.data.on('end', () => {
        res.end();
      });
    }

  } catch (error) {
    console.error('🔥 ERROR:', error.message);
    // 🟢 FIX 2: KALAU HEADERS DAH TERHANTAR TAPI SANGKUT, WAJIB TUTUP SAMBUNGAN SUPAYA TAK HANGING
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({ error: { message: error.message } });
    } else {
      res.end(); 
    }
  }
});

app.listen(PORT, () => console.log(`🚀 Proxy up on ${PORT}`));
