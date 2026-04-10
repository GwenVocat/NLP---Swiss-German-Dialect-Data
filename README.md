# Schweizerdeutsche Dialekt-Analyse

**Forschungsfrage:** *Wie lassen sich die häufigsten Wörter des Ostschweiz-Dialekts systematisch hochdeutschen Äquivalenten zuordnen?*

---

## Projektübersicht

Dieses Projekt untersucht den Ostschweizer Dialekt anhand von Audioaufnahmen aus dem [Swiss German Speech Corpus](https://www.sds-www.ch/). Die Aufnahmen werden mit drei Modellen verarbeitet:

- **`neurlang/ipa-whisper-base`** – transkribiert Audio direkt zu IPA-Phonemen (tatsächliche Aussprache)
- **`Flurin17/whisper-large-v3-turbo-swiss-german`** – transkribiert Audio zu Hochdeutsch
- **Phonemizer (espeak-ng)** – wandelt die Hochdeutsch-Referenzsätze in IPA um

Durch **Co-Occurrence-Analyse** werden IPA-Dialektwörter systematisch ihren hochdeutschen Entsprechungen zugeordnet.

### Dialektregionen im Datensatz (7)

| Region | Kantone (Beispiele) |
|--------|-------------------|
| Basel | BS, BL |
| Bern | BE |
| Graubünden | GR |
| Innerschweiz | LU, UR, SZ, OW, NW, ZG |
| **Ostschweiz** | **SG, TG, AR, AI, GL** ← Fokus dieser Analyse |
| Wallis | VS |
| Zürich | ZH |

---

## Projektstruktur

```
NLP---Swiss-German-Dialect-Data/
├── README.md                          ← Du bist hier
├── requirements.txt                   ← Python-Abhängigkeiten
├── transcribe.py                      ← Schritt 1: Transkription (3 Modelle)
├── clean.py                           ← Schritt 2: Datenbereinigung
├── classify.py                        ← Schritt 3: Ostschweiz-Mapping (Co-Occurrence)
├── analysis.ipynb                     ← Weitere Analyse
├── check_ipa.ipynb                    ← IPA-Output prüfen & debuggen
├── test_whisper_settings_ostschweiz.ipynb  ← Whisper-Parameter testen
├── Data/
│   ├── test.tsv                       ← Originaldaten (24'605 Aufnahmen)
│   ├── train_all.tsv                  ← Trainingsdaten (komplett)
│   ├── train_balanced.tsv             ← Trainingsdaten (balanciert)
│   ├── valid.tsv                      ← Validierungsdaten
│   ├── sample.tsv                     ← Gefiltertes Sample (1'400 Aufnahmen, 200/Region)
│   ├── transcriptions.csv             ← Rohe Transkriptionen (Output von transcribe.py)
│   ├── transcriptions_clean.csv       ← Bereinigte Transkriptionen (Output von clean.py)
│   ├── ostschweiz_mapping_results.csv ← Mapping-Ergebnisse (Output von classify.py)
│   ├── test_whisper_ostschweiz.csv    ← Whisper-Testläufe
│   └── clips__test/                   ← Audiodateien (.mp3)
│       └── [speaker_id]/[clip_hash].mp3
└── .venv/                             ← Virtuelle Umgebung (nicht im Repo)
```

---

## Setup

### Voraussetzungen
- Python 3.1+
- macOS (Apple Silicon empfohlen für MPS-Beschleunigung)
- ~10 GB Speicher für die Whisper-Modelle

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

### Wichtig: Immer `.venv` verwenden!

```bash
# Richtig (mit aktiviertem .venv)
source .venv/bin/activate
python transcribe.py

# Falsch (Anaconda → NumPy-Konflikte!)
/opt/anaconda3/bin/python transcribe.py
```

---

## Datensatz

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
| `sentence` | **Hochdeutscher Quellsatz** – NICHT der gesprochene Dialekt! |
| `sentence_source` | Herkunft des Satzes (parliament, news_switz, news_cultu, ...) |
| `client_id` | Anonymisierte Speaker-ID |
| `dialect_region` | Dialektregion (Basel, Bern, Graubünden, Innerschweiz, Ostschweiz, Wallis, Zürich) |
| `canton` | Kanton (BS, BE, GR, LU, ...) |
| `zipcode` | PLZ des Sprechers |
| `age` | Altersgruppe (teens, twenties, thirties, ...) |
| `gender` | Geschlecht (male, female) |

---

## Pipeline

### Schritt 1 – Transkription (`transcribe.py`)

Transkribiert alle Ostschweizer Sample-Dateien mit **3 Modellen sequenziell** (um RAM zu sparen):

| Modell | Output | Zweck |
|--------|--------|-------|
| `neurlang/ipa-whisper-base` | IPA-Phoneme | Tatsächliche Aussprache aus Audio |
| `Flurin17/whisper-large-v3-turbo-swiss-german` | Hochdeutsch | Swiss-Whisper-Transkription |
| Phonemizer (espeak-ng) | IPA-Phoneme | IPA der Hochdeutsch-Referenzsätze |

```bash
python transcribe.py
# Dauer: ~45-60 Min (Apple Silicon MPS)
```

**Output-Spalten in `transcriptions.csv`:**

| Spalte | Beschreibung |
|--------|-------------|
| `path` | Dateipfad |
| `dialect_region` | Dialektregion |
| `sentence` | Original-Hochdeutsch (Referenz) |
| `ipa_reference` | IPA der Referenz (via Phonemizer) |
| `ipa_audio` | IPA aus Audio (via IPA-Whisper) |
| `ipa_swiss_whisper` | IPA der Swiss-Whisper-Transkription (via Phonemizer) |

> **Warum IPA aus Audio?** Whisper normalisiert Dialekt zu Hochdeutsch – `ipa_audio` transkribiert die **tatsächliche Aussprache**. Der Vergleich von `ipa_audio` mit `ipa_reference` macht Dialektunterschiede direkt messbar.

### Schritt 2 – Datenbereinigung (`clean.py`)

Filtert `transcriptions.csv` → `transcriptions_clean.csv`:

1. Nur `dialect_region == "Ostschweiz"`
2. Zeilen aus `errors.csv` ausschliessen (falls vorhanden)
3. Leere / zu kurze IPA-Felder entfernen (< 3 Zeichen)
4. Garbled-Output-Erkennung: < 20% echte IPA-Zeichen → raus
5. Repetitions-Erkennung: Muster von 2–6 Zeichen, ≥ 4× hintereinander → raus
6. IPA-Normalisierung: Betonungszeichen (ˈ ˌ) entfernen, Whitespace normalisieren

```bash
python clean.py
```

### Schritt 3 – Ostschweiz-Mapping (`classify.py`)

Liest `transcriptions_clean.csv` und analysiert, welche IPA-Dialektwörter am häufigsten mit welchen Hochdeutschen Wörtern im selben Satz auftauchen (**Co-Occurrence**):

1. `ipa_audio` in Einzelwörter zerlegen
2. Hochdeutsche Referenz bereinigen (Kleinschreibung, Satzzeichen entfernen)
3. Für jedes IPA-Wort: häufigstes co-occurrendes Hochdeutsch-Wort finden
4. Top-50 häufigste Ostschweizer IPA-Wörter mit ihrer Hochdeutsch-Zuordnung ausgeben

```bash
python classify.py
# Output: Data/ostschweiz_mapping_results.csv
```

**Output-Spalten in `ostschweiz_mapping_results.csv`:**

| Spalte | Beschreibung |
|--------|-------------|
| `IPA_Dialekt` | Häufigstes IPA-Wort aus der Ostschweizer Aussprache |
| `Hochdeutsch_Zuordnung` | Wahrscheinlichstes hochdeutsches Äquivalent |
| `Gemeinsame_Treffer` | Anzahl Sätze, in denen beide Wörter vorkommen |

---

## Tech Stack

| Tool | Zweck |
|------|-------|
| Python 3.11+ | Programmiersprache |
| pandas | Datenverarbeitung |
| librosa | Audio laden & analysieren |
| transformers | Whisper-Modelle (HuggingFace) |
| torch | Deep Learning Backend (MPS/CUDA/CPU) |
| phonemizer | Text → IPA (espeak-ng) |
| scikit-learn | Metriken |
| matplotlib | Visualisierungen |

---

## Team

| Wer | Aufgabe |
|-----|---------|
| Gwen | Schritte 1–2: Transkription, Datenbereinigung |
| Chris | Schritt 3: Mapping-Analyse, Visualisierungen |

---

## Bekannte Hinweise

- **NumPy-Konflikt:** Anaconda-Python (`/opt/anaconda3/bin/python`) hat einen Versionskonflikt mit den verwendeten Bibliotheken. Immer das `.venv` verwenden!
- **`sentence`-Spalte:** Enthält den hochdeutschen Quellsatz, NICHT die gesprochene Dialektversion. Für die Analyse `ipa_audio` aus `transcriptions_clean.csv` verwenden.
- **Modell-Download:** Die Whisper-Modelle werden beim ersten Start heruntergeladen (~6–10 GB). Danach lokal gecacht.
