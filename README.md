# oc-free-api

OpenAI-kompatibler **lokaler API-Server** für OpenCode Zens Gratis-Modelle — damit du sie
überall nutzen kannst, wo ein normaler API-Key eingesetzt wird (Coding-Agenten, IDE-Plugins,
Chat-Tools, SDKs).

Nur Python-Standardbibliothek, keine Abhängigkeiten. Server für **persönliche Nutzung**.

## Start

```powershell
python server.py       # default: http://127.0.0.1:8787
# oder
.\start.bat
```

Beim ersten Start wird ein API-Key generiert, in `api-key.txt` gespeichert und in der
Konsole angezeigt. Optionen:

```
--port 9000            anderer Port
--host 0.0.0.0         auch im LAN erreichbar (Vorsicht!)
--key sk-meinkey       eigenen Key setzen
--refresh-models       Modell-Liste einmalig neu abfragen
```

## Endpoints

| Pfad | Beschreibung |
|---|---|
| `GET /health` | Liveness-Check (ohne Auth) |
| `GET /v1/models` | aktuelle Liste der freien Modelle |
| `POST /v1/chat/completions` | OpenAI Chat Completions (Streaming **und** non-Streaming, Tools) |

## Client-Konfiguration (universell für OpenAI-kompatible Tools)

```
base_url : http://127.0.0.1:8787/v1
api_key  : sk-oc-free-...   (steht in api-key.txt)
model    : big-pickle       (oder jedes Modell aus /v1/models)
```

Python-OpenAI-SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key=open("api-key.txt").read().strip(),
)
resp = client.chat.completions.create(
    model="big-pickle",
    messages=[{"role": "user", "content": "Erzähl einen Witz"}],
)
print(resp.choices[0].message.content)
```

curl:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $(cat api-key.txt)" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","stream":false,"messages":[{"role":"user","content":"hi"}]}'
```

## Agenten-Coding

Function Calling (Tool Calls) läuft transparent durch, wie bei einem ganz normalen
OpenAPI-Endpoint: Der Agent definiert seine Tools → das Modell ruft auf → der Agent
führt aus und sendet das Ergebnis → Finalantwort. Kein extra Setup nötig.

## Wie es intern funktioniert

Die Gratis-Tier von OpenCode Zen akzeptiert nur Anfragen, die exakt dem Fingerabdruck des
offiziellen OpenCode-Clients entsprechen (serverseitige Validierung; alle anderen Clients
erhalten `403 FreeTierError`). Der Server:

- baut die Request-Envelope exakt wie der offizielle OpenCode-CLI
  (User-Agent `opencode/1.19.0 ai-sdk/provider-utils/... runtime/bun/...`,
  `x-opencode-client: cli`, `x-opencode-project: global`,
  Session-/Request-IDs im exakten opencode-ID-Format: Sessions absteigend,
  Requests aufsteigend, 14 Base62-Zufallszeichen)
- nutzt `https://opencode.ai/inference/v1/chat/completions` bzw. `/responses`
  (muse-spark-Modelle nutzen intern die Responses-API; der Proxy übersetzt zusätzlich
  Weiter/Responses-Ereignisse ins OpenAI-Chat-Format)
- leitet Modell-Liste aus `https://opencode.ai/zen/v1/models` ∩ models.dev-Catalog ab
  (nur tool-fähige, komplett kostenlose Modelle) und cached sie in `models.json`
- stellt sicher, dass `bash`/`read` Tools in jeder Anfrage enthalten sind
  (sendet sonst Platzhalter mit echten, nicht leeren Schemas)

## Wichtige Hinweise

- **Persönliche Nutzung:** Basierend auf [`hermes-oc-free-provider`](https://github.com/anpicasso/hermes-oc-free-provider),
  dessen README Compliance mit den [OpenCode ToS](https://opencode.ai/legal/terms-of-service)
  und der [Zen-Dokumentation](https://opencode.ai/docs/zen/) verlangt.
  **Kein Reselling, kein Weiter-Serving an Dritte, keine Quota-Umgehung**, keine
  Mehrfach-Accounts für mehr Gratis-Kontingent.
- **Veränderliche Pipeline:** OpenCode kann die Transportprüfung jederzeit ändern oder
  abschalten — dann gibt dieser Server den Upstream-Fehler transparent zurück.
- **Key geheim halten:** `api-key.txt` enthält deinen Key. Wer ihn hat, kann über
  deinen Zugang arbeiten; teilen gilt gegenüber den OpenCode-Bedingungen als
  Bereitstellung für Dritte.
- **Daten-Verantwortung:** Gratis-/Trial-/Contributor-Modelle dürfen Prompts und
  Completions zur Modellverbesserung nutzen — keine Secrets oder regulierten Daten
  senden, außer die aktuellen Bedingungen erlauben es ausdrücklich.

