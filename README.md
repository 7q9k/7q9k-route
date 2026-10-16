# oc-free-api

OpenAI-kompatibler **lokaler API-Server** für OpenCode Zens Gratis-Modelle — damit du sie
überall nutzen kannst, wo ein normaler API-Key eingesetzt wird (Coding-Agenten, IDE-Plugins,
Chat-Tools, SDKs).

Nur Python-Standardbibliothek, keine Abhängigkeiten. Server für **persönliche Nutzung**.

## Start

```powershell
python server.py       # default: localhost + LAN (bindet auf 0.0.0.0)
# oder
.\start.bat
```

Nur lokal binden (LAN aus): `python server.py --host 127.0.0.1`. Im LAN-Modus brauchen
andere Geräte immer den API-Key (siehe unten).

Beim ersten Start wird ein API-Key generiert, in `api-key.txt` gespeichert und in der
Konsole angezeigt. Optionen:

```
--port 9000            anderer Port
--host 127.0.0.1       nur lokal (kein LAN)
--key sk-meinkey       eigenen Key setzen
--refresh-models       Modell-Liste einmalig neu abfragen
```

## Zugriff und API-Key

| Woher kommt die Anfrage? | Was gilt |
|---|---|
| Von diesem Rechner (`127.0.0.1`) | Kein Key nötig. Clients mit Platzhalter-Keys (`ollama`, `lm-studio` …) funktionieren. |
| Aus dem LAN | **Key immer nötig** (`Authorization: Bearer <key>` oder `x-api-key`). |
| Von einer fremden Webseite im Browser | Gesperrt (Origin-/Host-Prüfung gegen CSRF und DNS-Rebinding). |

Key neu erzeugen und Server stoppen geht nur vom Server-Rechner selbst. Ohne Key offen
sind nur Dashboard-Seite, `/health`, Modell-Listen und die Probe-Endpunkte.

## Dashboard

`http://localhost:8787/` zeigt Anfragen, Fehlerquote, Tokens, Antwortzeiten, einen
Verlauf der letzten 30 Minuten, alle Modelle mit Test-Knopf, einen Bereich zum
Ausprobieren (gestreamt) und das Aktivitätsprotokoll. Wer das Dashboard aus dem LAN
öffnet, wird nach dem Key gefragt; der Key selbst wird dort nie angezeigt.

## Endpoints

| Pfad | Zugriff | Beschreibung |
|---|---|---|
| `GET /` · `GET /dashboard` | offen | Web-Dashboard |
| `GET /api/dashboard` · `GET /api/dashboard/stats` | Key / lokal | Status und Statistiken fürs Dashboard |
| `POST /api/dashboard/test` · `/refresh` | Key / lokal | Modell testen, Modell-Liste neu abfragen |
| `POST /api/dashboard/key` · `/stop` | nur lokal | Key neu erzeugen, Server stoppen |
| `GET /health` | offen | Liveness-Check |
| `GET /v1/models` · `GET /api/v1/models` | offen | Modell-Liste (OpenAI-Format) |
| `GET /api/tags` · `GET /api/ps` · `POST /api/show` | offen | **Ollama**-Modell-Endpoints |
| `POST /api/chat` · `POST /api/generate` | Key / lokal | **Ollama** Chat/Generate (streamt standardmäßig, Tools) |
| `GET /version` · `GET /v1/props` · `GET /props` | offen | **LM Studio / llama.cpp** Probe-Endpoints |
| `POST /v1/chat/completions` | Key / lokal | OpenAI Chat Completions (Stream und non-Stream, Tools) |

Fehler mitten im Stream kommen als `data: {"error": …}` (OpenAI) bzw.
`{"error": "…"}`-Zeile (Ollama) beim Client an. Request-Bodys sind auf 32 MB begrenzt.

### Ollama-kompatible Tools / LM Studio

Der Server emuliert die Endpunkte, mit denen Clients Modelle entdecken — Listener
einfach auf `http://127.0.0.1:8787` zeigen lassen (kein Suffix; vom selben Rechner ohne Key, aus dem LAN mit Key).

- **Ollama-Clients** (Open WebUI, Continue mit Ollama-Provider, …): Base URL
  `http://127.0.0.1:8787` — Modelle erscheinen wie lokal installierte.
- **LM Studio / llama.cpp-Clients**: OpenAI-Konfiguration `http://127.0.0.1:8787/v1`
  plus eigenem Key; die Probe-Endpunkte (`/version`, `/props`) antworten ebenfalls.
- **Agent-Fähigkeit**: Ollama-`/api/chat` unterstützt Tool Calls inklusive
  Tool-Rückgabe (Feld `tool_name`) und konvertiert `options` (`.temperature`,
  `.num_predict`, `top_p`, `stop`, `seed`) sowie `format: "json"`.

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

## Tests

```powershell
python -m unittest discover -s tests -v
```

Die Tests laufen ohne Netzwerk (Upstream wird ersetzt) und prüfen Umsetzung der Formate,
Streaming-Fehler, Zugriffsregeln und Statistik.

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
- **LAN-Modus:** Andere Geräte kommen nur mit Key rein. Trotzdem gilt: Wer den Key hat,
  arbeitet über deinen Zugang.
- **Key geheim halten:** `api-key.txt` enthält deinen Key. Wer ihn hat, kann über
  deinen Zugang arbeiten; teilen gilt gegenüber den OpenCode-Bedingungen als
  Bereitstellung für Dritte.
- **Daten-Verantwortung:** Gratis-/Trial-/Contributor-Modelle dürfen Prompts und
  Completions zur Modellverbesserung nutzen — keine Secrets oder regulierten Daten
  senden, außer die aktuellen Bedingungen erlauben es ausdrücklich.

