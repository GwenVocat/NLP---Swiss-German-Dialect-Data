# 🇨🇭 Schweizerdeutsche Dialekt-Analyse

**Forschungsfrage:** *Wie lassen sich die häufigsten Wörter des Ostschweiz-Dialekts systematisch hochdeutschen Äquivalenten zuordnen?*

Abrgrenzen, und vielleicht nur einzelne wörter eine Region

---

## 📋 Projektübersicht

Dieses Projekt untersucht Schweizerdeutsche Dialekte anhand von Audioaufnahmen aus dem [Swiss German Speech Corpus](https://www.sds-www.ch/). Sprachaufnahmen werden mit zwei Modellen transkribiert: Whisper liefert Hochdeutsch-Text, wav2vec2-lv-60-espeak-cv-ft liefert IPA-Phoneme. Anschliessend mittels **TF-IDF** und **Machine Learning** analysiert, um Dialektregionen automatisch zu erkennen.

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
├── README.md                  ← Du bist hier
├── requirements.txt           ← Python-Abhängigkeiten
├── analysis.ipynb             ← Jupyter Notebook: Datenexploration & Visualisierung
├── transcribe.py              ← Whisper-Transkription aller Samples von der Ostschweizer Dialektes
├── classify.py                ← TF-IDF, Classifier, Plots (TODO)
├── Data/
│   ├── test.tsv               ← Originaldaten (24'605 Aufnahmen)
│   ├── train_all.tsv          ← Trainingsdaten (komplett)
│   ├── train_balanced.tsv     ← Trainingsdaten (balanciert)
│   ├── valid.tsv              ← Validierungsdaten
│   ├── sample.tsv             ← Gefiltertes Sample (1'400 Aufnahmen, 200/Region)
│   ├── transcriptions.csv     ← Whisper-Transkriptionen (Output von transcribe.py)
│   └── clips__test/           ← Audiodateien (.mp3)
│       └── [speaker_id]/[clip_hash].mp3
└── .venv/                     ← Virtuelle Umgebung (nicht im Repo)
```

---

## 🔧 Setup

### Voraussetzungen
- Python 3.12+
- macOS (Apple Silicon empfohlen für MPS-Beschleunigung)
- ~10 GB Speicher für Whisper-Modell

### Installation

```bash
# Repo klonen
git clone https://github.com/<user>/NLP---Swiss-German-Dialect-Data.git
cd NLP---Swiss-German-Dialect-Data

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

Transkribiert alle 1'400 Sample-Dateien mit **2 Modellen parallel**:

| Modell | Output | Zweck |
|--------|--------|-------|
| `openai/whisper-large-v2` | Hochdeutsch | Baseline – subtiles Dialekt-"Leakage" |
| `neurlang/ipa-whisper-base` | IPA-Phoneme | Direkte Lautschrift – starkes Dialekt-Signal |

```bash
python transcribe.py
# ⏱ Dauer: ~45-60 Min (Apple Silicon)
```

**Output-Spalten in `transcriptions.csv`:**

| Spalte | Beschreibung |
|--------|-------------|
| `path` | Dateipfad |
| `dialect_region` | Dialektregion |
| `sentence` | Original-Hochdeutsch (Referenz) |
| `transcription_whisper` | Hochdeutsch (Whisper large-v2) |
| `transcription_ipa` | IPA-Phoneme (neurlang/ipa-whisper-base) |

> **Warum IPA?** Whisper normalisiert Dialekt zu Hochdeutsch – alle Regionen sehen fast gleich aus.
> IPA transkribiert die **tatsächliche Aussprache**: Walliser `frˈyːɛɪk` vs. Zürcher `ruhig` wird direkt sichtbar.
> Für den Classifier kann man beide Spalten vergleichen und schauen, welche besser performt.

### Schritt 3 & 4 – Analyse & Klassifikation (`classify.py`) 🔜

- **TF-IDF** pro Dialektregion → Top-Wörter extrahieren
- **Cosine-Similarity Heatmap** zwischen Regionen
- **Wordclouds** pro Region
- **Classifier:** Logistic Regression vs. Naive Bayes (80/20 Split)
- **Confusion Matrix** + **Schweizerkarte** mit Accuracy pro Region

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

---

## 👥 Team

| Wer | Aufgabe |
|-----|---------|
| Gwen | Schritte 1–2: Exploration, Sampling, Transkription |
| Chris | Schritte 3–4: TF-IDF, Classifier, Visualisierungen |

---

## 📝 Bekannte Hinweise

- **NumPy-Konflikt:** Anaconda-Python (`/opt/anaconda3/bin/python`) hat einen Versionskonflikt. Immer das `.venv` verwenden!
- **`sentence`-Spalte:** Enthält den hochdeutschen Quellsatz, NICHT die gesprochene Dialektversion. Für die Analyse die `transcription`-Spalte aus `transcriptions.csv` verwenden.
- **Whisper-Modell:** Wird beim ersten Start heruntergeladen (~6 GB). Danach ist es lokal gecacht.
