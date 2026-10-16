# Agent-Setup (nur OpenAI-kompatibel)

Der Proxy spricht exakt den OpenAI-Chat-Completions-Dialekt inklusive
Function Calling. Alle Werte (Key in `api-key.txt`):

```
base_url : http://127.0.0.1:8787/v1
api_key  : sk-oc-free-...        (= Inhalt von api-key.txt, generiert beim ersten Start)
model    : big-pickle            (empfohlen für Coding; alternativ aus GET /v1/models)
```

## Cline / Roo Code (VS Code)

Settings → Language Model → OpenAI Compatible:

```json
{
  "openAiCompatible": {
    "baseUrl": "http://127.0.0.1:8787/v1",
    "apiKey": "<aus api-key.txt>",
    "model": "big-pickle"
  }
}
```

## Aider (CLI)

```bash
set OPENAI_API_BASE=http://127.0.0.1:8787/v1
set OPENAI_API_KEY=<aus api-key.txt>
aider --model openai/big-pickle
```

## Continue (VS Code, config.yaml)

```yaml
models:
  - name: big-pickle (oc-free)
    provider: openai
    apiBase: http://127.0.0.1:8787/v1
    apiKey: <aus api-key.txt>
    model: big-pickle
    roles: [chat, edit, apply]
```

## OpenCode (Kontext diesem Endpoint ändern — ja: OpenCode als Client)

`~/.config/opencode/opencode.json` (bzw. `opencode.json` im Projekt):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "oc-free": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenCode Free (local proxy)",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        "apiKey": "<aus api-key.txt>"
      },
      "models": {
        "big-pickle": { "name": "big-pickle" }
      }
    }
  }
}
```

## Eigener Python-Code (openai-SDK, vgl. examples/openai_agent.py)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key=open("api-key.txt").read().strip(),
)
resp = client.chat.completions.create(
    model="big-pickle",
    messages=[{"role": "user", "content": "hi"}],
    tools=[...deine Tools...],
)
```

Ein vollständiger Mini-Coding-Agent mit write_file/read_file/run_python liegt in
`examples/openai_agent.py` — Start:

```bash
python examples/openai_agent.py "Schreibe Funktion add(a,b) nach agent_demo.py und teste sie"
```

## Hinweise

- Streaming + Tools beide modi getestet.
- Bei `429` hat OpenCode kurz gedrosselt — Agent neu versuchen lassen (die
  meisten Client-Frameworks retrien selbst nicht, ggf. Pause einbauen).
- Server läuft solange das CMD-Fenster von `start.bat` offen ist.
