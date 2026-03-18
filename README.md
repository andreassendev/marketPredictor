<div align="center">

<img src="./static/image/MiroFish_logo_compressed.jpeg" alt="MarketPredictor Logo" width="75%"/>

**MarketPredictor** — Multi-agent AI-prediksjonsmotor for markeder, sport og fremtidsscenarier

*Basert på [MiroFish](https://github.com/666ghj/MiroFish) — A Simple and Universal Swarm Intelligence Engine*

[![GitHub Stars](https://img.shields.io/github/stars/andreassendev/marketPredictor?style=flat-square&color=DAA520)](https://github.com/andreassendev/marketPredictor/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/andreassendev/marketPredictor?style=flat-square)](https://github.com/andreassendev/marketPredictor/network)

</div>

## ⚡ Oversikt

**MarketPredictor** er en AI-prediksjonsmotor drevet av multi-agent-teknologi. Ved å mate inn seed-informasjon fra den virkelige verden (sportsstatistikk, nyheter, markedsdata, finansielle signaler) bygger den automatisk en høy-fidelitets parallell digital verden. Inne i denne verdenen interagerer tusenvis av intelligente agenter med egne personligheter, langtidsminne og adferdslogikk fritt — og gjennomgår sosial evolusjon.

Du kan injisere variabler dynamisk fra et «gudeperspektiv» for å presist utlede fremtidige utfall — **la fremtiden øve seg i en digital sandkasse, og ta bedre beslutninger etter utallige simuleringer**.

> Du trenger bare å: Laste opp seed-materiale (datarapporter, nyheter, statistikk) og beskrive hva du vil forutsi med naturlig språk
>
> MarketPredictor returnerer: En detaljert prediksjonsrapport og en interaktiv digital verden du kan utforske

### Bruksområder

- **Sport & Polymarket** — Simuler kampresultater basert på lagstatistikk, skader, form og historikk
- **Finans** — Markedssentiment, trender og prisutvikling
- **Politikk** — Valg, folkeavstemninger, policy-endringer
- **Kultur** — Oscar-vinnere, virale trender, underholdning
- **Geopolitikk** — Konflikter, handelsavtaler, diplomatiske utfall

## 🔄 Arbeidsflyt

1. **Grafkonstruksjon** — Seed-ekstraksjon, individ- og gruppeminne-injeksjon, GraphRAG-bygging
2. **Miljøoppsett** — Entitetsrelasjoner, personagenerering, agent-konfigurasjon
3. **Simulering** — Dobbel-plattform parallellsimulering, automatisk tolkning av prediksjonskrav, dynamisk tidsminne
4. **Rapportgenerering** — ReportAgent med rikt verktøysett for dyp interaksjon med simuleringsmiljøet
5. **Dyp interaksjon** — Snakk med hvilken som helst agent i den simulerte verdenen

## 🚀 Hurtigstart

### Forutsetninger

| Verktøy | Versjon | Beskrivelse | Sjekk installasjon |
|---------|---------|-------------|-------------------|
| **Node.js** | 18+ | Frontend-kjøremiljø, inkluderer npm | `node -v` |
| **Python** | ≥3.11, ≤3.12 | Backend-kjøremiljø | `python --version` |
| **uv** | Nyeste | Python-pakkebehandler | `uv --version` |

### 1. Konfigurer miljøvariabler

```bash
# Kopier eksempel-konfigurasjonsfilen
cp .env.example .env

# Rediger .env-filen og fyll inn nødvendige API-nøkler
```

**Nødvendige miljøvariabler:**

```env
# LLM API-konfigurasjon (støtter alle LLM-APIer med OpenAI SDK-format)
# Alternativer: OpenRouter, Groq (gratis tier), OpenAI, Qwen
LLM_API_KEY=din_api_nøkkel
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=google/gemini-flash-2.0

# Zep Cloud-konfigurasjon
# Gratis månedlig kvote er nok for enkel bruk: https://app.getzep.com/
ZEP_API_KEY=din_zep_api_nøkkel
```

### 2. Installer avhengigheter

```bash
# Installer alle avhengigheter på én gang (root + frontend + backend)
npm run setup:all
```

Eller steg for steg:

```bash
# Installer Node-avhengigheter (root + frontend)
npm run setup

# Installer Python-avhengigheter (backend, oppretter virtuelt miljø automatisk)
npm run setup:backend
```

### 3. Start tjenestene

```bash
# Start både frontend og backend (kjør fra prosjektets rotmappe)
npm run dev
```

**Tjenesteadresser:**
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5001`

**Start enkeltvis:**

```bash
npm run backend   # Kun backend
npm run frontend  # Kun frontend
```

### Docker-alternativ

```bash
# 1. Konfigurer miljøvariabler (samme som over)
cp .env.example .env

# 2. Hent image og start
docker compose up -d
```

Leser `.env` fra rotmappen som standard, mapper porter `3000 (frontend) / 5001 (backend)`

## 📄 Kreditering

- Basert på **[MiroFish](https://github.com/666ghj/MiroFish)** av MiroFish-teamet med strategisk støtte fra Shanda Group
- Simuleringsmotoren drives av **[OASIS](https://github.com/camel-ai/oasis)** fra CAMEL-AI-teamet

## 📜 Lisens

AGPL-3.0 — Se [LICENSE](./LICENSE) for detaljer.
