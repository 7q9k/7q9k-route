# Agent setup

Any OpenAI-compatible client works:

```
base_url: http://127.0.0.1:8787/v1
api_key:  contents of api-key.txt
model:    big-pickle (or any model from /v1/models)
```

Clients that only speak the Responses protocol can use the same base URL. Anthropic clients use it without `/v1`.

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
    "7q9k-route": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8787/v1", "apiKey": "<key>" },
      "models": { "big-pickle": {} }
    }
  }
}
```

**Claude Code**

One click: dashboard -> Apps -> Claude Code -> Connect. Or add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    "ANTHROPIC_AUTH_TOKEN": "<key>",
    "ANTHROPIC_MODEL": "big-pickle",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "big-pickle",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "big-pickle",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "big-pickle",
    "CLAUDE_CODE_SUBAGENT_MODEL": "big-pickle"
  }
}
```

The base URL has no `/v1`; Claude Code adds `/v1/messages` itself. Then start `claude` and switch models with `/model <name>`.

**Python**: see `examples/openai_agent.py` for a small tool-calling agent.

On HTTP 429 you hit OpenCode's free limit; wait and retry.
