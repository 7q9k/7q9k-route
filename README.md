<img src="assets/icon.png" width="64" alt="">

# oc-free-api

Local server that exposes OpenCode Zen's free models as an OpenAI- and Ollama-compatible API, with a web dashboard. Python standard library only. Personal use.

## Download

From [Releases](https://github.com/7q9k/oc-free-api/releases):

- `oc-free-api-<version>.exe`: Windows, no Python needed. Double-click it. Your key, stats and settings are stored in `%LOCALAPPDATA%\oc-free-api`, not next to the exe.
- `oc-free-api-<version>-python.zip`: any OS with Python 3.10+. Unzip and run `python server.py`.

## Run

Double-click `oc-free-api.exe`. A small window shows the dashboard address, the browser opens, and the server keeps running in the background. Opening the exe again just opens the dashboard.

Everything else happens in the dashboard (http://localhost:8787): who can connect, port, start with Windows, API key, restart and stop.

Base URL for apps: `http://localhost:8787/v1` (also works without `/v1`). The API key is shown in the dashboard and stored in `api-key.txt` in the data folder.

Command line options (optional): `--port`, `--host`, `--key`, `--no-browser`, `--background`, `--no-save`.

## Apps

The Apps tab finds supported apps on this computer and adds oc-free-api to them with one click: DeepSeek Harness and Hermes. A backup of each changed file is kept, and connected apps are updated automatically when the key, port or model list changes. New apps are added as a small class in `integrations.py`.

Under Settings you can set your own fixed API key; it stays the same across restarts and updates.

## Access

- Same computer: no key needed.
- Your network: the API key is always required.
- Settings, key reset and stop only work on the computer that runs the server.

## API

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions (stream, tools) |
| `POST /v1/responses` | OpenAI Responses API (stateless) |
| `GET /v1/models` | model list |
| `POST /api/chat`, `/api/generate`, `GET /api/tags` | Ollama-compatible |

## Dashboard

Charts for requests and tokens (30 min / 24 h), tokens and cost per model, response times, model tests, a chat playground and an activity log. Stats are saved to `stats.json` and survive restarts. Cost shows what the same tokens would cost on OpenRouter (live prices, exact model matches only).

## Free limit

OpenCode is not unlimited. Free models are rate-limited per IP; there is no official number (users report about 200 requests/day). When the limit is hit, OpenCode returns HTTP 429 `FreeUsageLimitError`.

## Tests

```
python -m unittest discover -s tests
```

## Notes

Based on [hermes-oc-free-provider](https://github.com/anpicasso/hermes-oc-free-provider). Follow the [OpenCode terms](https://opencode.ai/legal/terms-of-service): no reselling or sharing access. Free models may use your prompts for training, so do not send secrets.
