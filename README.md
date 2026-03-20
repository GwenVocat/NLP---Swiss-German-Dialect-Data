# 🇨🇭 Schweizerdeutsche Dialekt-Analyse

**Forschungsfrage:** *Wie unterscheidet sich der Wortschatz der sieben Schweizerdeutschen Dialektregionen, und können diese Unterschiede zur automatischen Dialekterkennung genutzt werden?*

---

## 📋 Projektübersicht

Dieses Projekt untersucht Schweizerdeutsche Dialekte anhand von Audioaufnahmen aus dem [Swiss German Speech Corpus](https://www.sds-www.ch/). Sprachaufnahmen werden mit **OpenAI Whisper** transkribiert und anschliessend mittels **TF-IDF** und **Machine Learning** analysiert, um Dialektregionen automatisch zu erkennen.

### Dialektregionen (7)

| Region | Kantone (Beispiele) |
|--------|-------------------|
| Basel | BS, BL |
| Bern | BE |
| Graubünden | GR |
| Innerschweiz | LU, UR, SZ, OW, NW, ZG |
| Ostschweiz | SG, TG, AR, AI, GL |
| Wallis | VS |
| Zürich | ZH |

---

## 🗂 Projektstruktur

```
NLP---Swiss-German-Dialect-Data/
├── README.md                      ← Du bist hier
├── requirements.txt               ← Python-Abhängigkeiten
├── analysis.ipynb                 ← Jupyter Notebook: Datenexploration & Visualisierung
├── transcribe.py                  ← Whisper-Transkription aller 1400 Samples
├── test_whisper_settings.py       ← A/B-Test verschiedener Whisper-Modelle
├── classify.py                    ← TF-IDF, Classifier, Plots (TODO)
├── Data/
│   ├── test.tsv                   ← Originaldaten (24'605 Aufnahmen)
│   ├── train_all.tsv              ← Trainingsdaten (komplett)
│   ├── train_balanced.tsv         ← Trainingsdaten (balanciert)
│   ├── valid.tsv                  ← Validierungsdaten
│   ├── sample.tsv                 ← Gefiltertes Sample (1'400 Aufnahmen, 200/Region)
│   ├── transcriptions.csv         ← Whisper-Transkriptionen (Output von transcribe.py)
│   └── clips__test/               ← Audiodateien (.mp3)
│       └── [speaker_id]/[clip_hash].mp3
└── .venv/                         ← Virtuelle Umgebung (nicht im Repo)
```

---

## 🔧 Setup

### Voraussetzungen
- Python 3.12+
- macOS (Apple Silicon empfohlen für MPS-Beschleunigung)
- `ffmpeg` (`brew install ffmpeg`)
- ~10 GB Speicher für Whisper-Modell

### Installation

```bash
# Repo klonen
git clone https://github.com/GwenVocat/NLP---Swiss-German-Dialect-Data.git
cd NLP---Swiss-German-Dialect-Data

# ffmpeg installieren (falls nicht vorhanden)
brew install ffmpeg

# Virtuelle Umgebung erstellen & aktivieren
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### ⚠️ Wichtig: Immer `.venv` verwenden!

```bash
# ✅ Richtig (mit aktiviertem .venv)
source .venv/bin/activate
python transcribe.py

# ❌ Falsch (Anaconda → NumPy-Konflikte!)
/opt/anaconda3/bin/python transcribe.py
```

---

## 📊 Datensatz

- **Quelle:** Swiss German Speech Corpus
- **Gesamtdaten:** 24'605 Aufnahmen (test.tsv)
- **Sample:** 1'400 Aufnahmen (200 pro Region, duration 2–15s)
- **Sampling Rate:** 16'000 Hz
- **Format:** MP3

### TSV-Spalten

| Spalte | Beschreibung |
|--------|-------------|
| `path` | Relativer Pfad zur MP3-Datei (`speaker_id/clip_hash.mp3`) |
| `duration` | Aufnahmedauer in Sekunden |
| `sentence` | ⚠️ **Hochdeutscher Quellsatz** – NICHT der gesprochene Dialekt! |
| `sentence_source` | Herkunft des Satzes (parliament, news_switz, news_cultu, ...) |
| `client_id` | Anonymisierte Speaker-ID |
| `dialect_region` | Dialektregion (Basel, Bern, Graubünden, Innerschweiz, Ostschweiz, Wallis, Zürich) |
| `canton` | Kanton (BS, BE, GR, LU, ...) |
| `zipcode` | PLZ des Sprechers |
| `age` | Altersgruppe (teens, twenties, thirties, ...) |
| `gender` | Geschlecht (male, female) |

---

## 🚀 Pipeline

### Schritt 1 – Exploration & Sampling (`analysis.ipynb`)

Jupyter Notebook mit interaktiven Visualisierungen:

1. **Daten laden** – TSV einlesen, Überblick
2. **Visualisierungen:**
   - Aufnahmen pro Dialektregion
   - Verteilung der Aufnahmedauer (Histogramm + Boxplot pro Region)
   - Geschlechterverteilung (gesamt + pro Region)
   - Altersverteilung (gesamt + pro Region)
   - Top-15 Kantone, Satzquellen
3. **Filtern** – Nur Aufnahmen mit 2s ≤ duration ≤ 15s
4. **Stratifiziertes Sampling** – 200 Aufnahmen pro Region → 1'400 total
5. **Audio-Check** – Waveform + Mel-Spektrogramm einer Beispieldatei

```bash
jupyter notebook analysis.ipynb
```

### Schritt 2 – Transkription (`transcribe.py`)

Transkribiert alle 1'400 Sample-Dateien mit **Whisper large-v2**:

- Erkennt automatisch Apple Silicon (MPS), CUDA oder CPU
- Fortschrittsanzeige alle 50 Dateien
- Speichert Ergebnisse als `Data/transcriptions.csv`

```bash
python transcribe.py
# ⏱ Dauer: ~27 Min (Apple Silicon), ~2-3h (CPU)
```

**Output-Spalten in `transcriptions.csv`:**

| Spalte | Beschreibung |
|--------|-------------|
| `path` | Dateipfad |
| `dialect_region` | Dialektregion |
| `sentence` | Original-Hochdeutsch (Referenz) |
| `transcription` | Whisper-Transkription (Hochdeutsch mit Dialekt-Leakage) |

### Schritt 3 & 4 – Analyse & Klassifikation (`classify.py`) 🔜

*Wird von Teamkollege übernommen:*

- **TF-IDF** pro Dialektregion → Top-Wörter extrahieren
- **Cosine-Similarity Heatmap** zwischen Regionen
- **Wordclouds** pro Region
- **Classifier:** Logistic Regression vs. Naive Bayes (80/20 Split)
- **Confusion Matrix** + **Schweizerkarte** mit Accuracy pro Region

---

## ⚠️ Bekannte Limitierung: Whisper normalisiert Dialekt zu Hochdeutsch

### Das Problem

Whisper (alle Versionen) ist darauf trainiert, **Schweizerdeutsch → Hochdeutsch** zu übersetzen. Schweizerdeutsch hat keine standardisierte Schriftform, weshalb alle verfügbaren ASR-Modelle auf Hochdeutsch-Output trainiert sind.

Wir haben mehrere Ansätze getestet:

| Getestet | Ergebnis |
|----------|----------|
| `whisper-large-v2` mit `language="german"` | Stark normalisiertes Hochdeutsch |
| `whisper-large-v2` ohne Language-Zwang | Identisches Ergebnis |
| `Flurin17/whisper-large-v3-turbo-swiss-german` (feingetunt auf 350h CH-Deutsch) | Kaum Unterschied – auch Hochdeutsch-Output |

**Quelle:** [Does Whisper Understand Swiss German? (ACL VarDial 2024)](https://aclanthology.org/2024.vardial-1.3/)

### Was trotzdem durchkommt ("Leakage")

Trotz der Normalisierung bleiben **subtile Dialekt-Unterschiede** in den Transkriptionen erhalten:

| Phänomen | Original (HD) | Whisper-Output | Erklärung |
|----------|--------------|----------------|-----------|
| Perfekt statt Präteritum | "hatte trainiert" | "**hat** trainiert" | CH-Deutsch kennt kein Präteritum |
| Wortersetzungen | "getilgt" | "**herausgestrichen**" | Dialektaler Wortschatz leakt |
| Umformulierungen | "Deshalb sei..." | "**Wegen dem** sei..." | Andere Satzstruktur |
| Totalausfälle | "An Ostern..." | "**Der Ast...**" | Starker Dialekt → Whisper versteht Müll |

### Strategie

Wir nutzen genau dieses Leakage: **TF-IDF über 200 Samples pro Region** sollte systematische Unterschiede zwischen den Dialektregionen aufdecken. Falls die Classifier-Accuracy tief ist, ist das selbst ein interessantes Ergebnis.

### Mögliche Alternative (nicht umgesetzt)

Statt textbasiert könnte man **Audio-Features direkt** vergleichen (Mel-Spektrogramme, MFCCs als Input für den Classifier). Das umgeht das Transkriptionsproblem komplett.

---

## 📁 Wichtige Dateien für den nächsten Schritt

Wenn `transcribe.py` durchgelaufen ist, brauchst du:

| Datei | Wozu |
|-------|------|
| `Data/transcriptions.csv` | Input für TF-IDF & Classifier |
| `Data/sample.tsv` | Metadaten zum Sample (Region, Alter, Geschlecht, etc.) |
| `analysis.ipynb` | Explorative Analyse nachvollziehen |

---

## 🛠 Tech Stack

| Tool | Version | Zweck |
|------|---------|-------|
| Python | 3.12+ | Programmiersprache |
| pandas | – | Datenverarbeitung |
| librosa | – | Audio laden & analysieren |
| transformers | – | Whisper-Modell (HuggingFace) |
| torch | – | Deep Learning Backend |
| scikit-learn | – | TF-IDF, Classifier, Metriken |
| matplotlib | – | Visualisierungen |
| wordcloud | – | Wordcloud-Grafiken |
| ffmpeg | – | Audio-Dekodierung (System-Dependency) |

---

## 👥 Team

| Wer | Aufgabe |
|-----|---------|
| Gwen | Schritte 1–2: Exploration, Sampling, Transkription |
| Teamkollege | Schritte 3–4: TF-IDF, Classifier, Visualisierungen |

---

## 📝 Weitere Hinweise

- **NumPy-Konflikt:** Anaconda-Python (`/opt/anaconda3/bin/python`) hat einen Versionskonflikt. Immer das `.venv` verwenden!
- **`sentence`-Spalte:** Enthält den hochdeutschen Quellsatz, NICHT die gesprochene Dialektversion. Für die Analyse die `transcription`-Spalte aus `transcriptions.csv` verwenden.
- **Whisper-Modell:** Wird beim ersten Start heruntergeladen (~6 GB). Danach ist es lokal gecacht unter `~/.cache/huggingface/`.
