# LoL Match Outcome Predictor

## Project Overview
This repository contains the backend data collection scripts for a personal academic project. The main goal is to build a dataset of League of Legends match timelines to train Transformer-based models for predicting match outcomes based on early-game metrics.

> Note: This is purely for personal study and academic research. There is no commercial intent, and no public application or service will be deployed from this code.

---

## Data Collection Pipeline

The collection scripts are designed to run asynchronously on an AWS EC2 (Ubuntu) instance. To respect Riot's rate limits and prevent duplicate data, the pipeline follows a sequential approach:

1. Fetch high-tier player ladders **(Challenger & Grandmaster)** using the `LEAGUE-V4` API.
2. Convert `summonerId` to `PUUID` via the `SUMMONER-V4` API.
3. Retrieve recent match histories and detailed timeline data via the `MATCH-V5` API.
   - Python `Set`s are used to filter out duplicate matches across different players before making heavy data requests.

---

## Environment

### ☁️ Data Collection (AWS EC2)

| Item | Detail |
|------|--------|
| Language | Python 3.10+ |
| Key Libraries | `aiohttp`, `asyncio`, `pandas` |
| Storage | CSV / Parquet (chunked for memory efficiency) |

### 💻 Model Training (Local Machine)

| Item | Detail |
|------|--------|
| OS | Windows / WSL2 (Ubuntu) |
| CPU | AMD Ryzen 9 7900X3D |
| GPU | NVIDIA GeForce RTX 4070 Super |
| Frameworks | PyTorch, scikit-learn |

---

## API Usage & Compliance

This project operates under the **Personal Project API** tier provided by Riot Games.

- ✅ Collected raw datasets (PUUIDs, match timelines) are stored **locally** for offline training only.
- ✅ Data will **not** be shared, distributed, or monetized in any form.
- ✅ All scripts implement **exponential backoff** and sleep cycles based on `Retry-After` headers to strictly adhere to Riot's rate limits.

---

## Disclaimer

> This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc.
