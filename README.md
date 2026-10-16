# oc-free-api

Local server that exposes OpenCode Zen's free models as an OpenAI- and Ollama-compatible API, with a web dashboard. Python standard library only. Personal use.

## Download

From [Releases](https://github.com/7q9k/oc-free-api/releases):

- `oc-free-api.exe`: Windows, no Python needed. Put it in its own folder and run it.
- `oc-free-api-<version>-python.zip`: any OS with Python 3.10+. Unzip and run `python server.py`.

## Run

```
python server.py
```

Dashboard: http://localhost:8787 · Base URL: `http://localhost:8787/v1` · API key: `api-key.txt` (created on first start).

Options: `--port`, `--host 127.0.0.1` (local only), `--key`, `--refresh-models`, `--no-save`.

## Access

- From the same machine: no key needed.
- From the LAN: the API key is always required.
- Key reset, stop and stats reset only work from the server machine.

## API

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions (stream, tools) |
| `POST /v1/responses` | OpenAI Responses API (stateless) |
| `GET /v1/models` | model list |
| `POST /api/chat`, `/api/generate`, `GET /api/tags` | Ollama-compatible |

## Dashboard

Requests, exact input/output tokens, response times, model tests, a chat playground and an activity log. Stats are saved to `stats.json` and survive restarts.

Cost shows what the same tokens would cost on OpenRouter, using live OpenRouter prices. Only models with an exact OpenRouter match get a price.

## Free limit

OpenCode is not unlimited. Free models are rate-limited per IP; there is no official number (users report about 200 requests/day). When the limit is hit, OpenCode returns HTTP 429 `FreeUsageLimitError`. The dashboard shows your requests in the last 5 h and 24 h.

## Tests

```
python -m unittest discover -s tests
```

## Notes

Based on [hermes-oc-free-provider](https://github.com/anpicasso/hermes-oc-free-provider). Follow the [OpenCode terms](https://opencode.ai/legal/terms-of-service): no reselling or sharing access. Free models may use your prompts for training, so do not send secrets.
