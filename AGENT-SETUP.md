# Agent setup

Any OpenAI-compatible client works:

```
base_url: http://127.0.0.1:8787/v1
api_key:  contents of api-key.txt
model:    big-pickle (or any model from /v1/models)
```

Clients that only speak the Responses protocol can use the same base URL.

**Aider**

```
set OPENAI_API_BASE=http://127.0.0.1:8787/v1
set OPENAI_API_KEY=<key>
aider --model openai/big-pickle
```

**OpenCode** (`opencode.json`)

```json
{
  "provider": {
    "oc-free": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8787/v1", "apiKey": "<key>" },
      "models": { "big-pickle": {} }
    }
  }
}
```

**Python**: see `examples/openai_agent.py` for a small tool-calling agent.

On HTTP 429 you hit OpenCode's free limit; wait and retry.
