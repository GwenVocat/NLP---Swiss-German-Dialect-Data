"""
Whisper-Transkription aller 1400 Sample-Dateien

Modell: openai/whisper-large-v2
Aufgabe: Schweizerdeutsche Audiodateien → Hochdeutsche Transkription
Hinweis: Whisper normalisiert Dialekt zu Hochdeutsch. Subtile Dialekt-Unterschiede
         ("Leakage") bleiben erhalten und werden für die Analyse genutzt.
         Siehe README.md → "Bekannte Limitierungen" für Details.

Verwendung: python transcribe.py
Dauer:      ~27 Min auf Apple Silicon (MPS), ~2-3h auf CPU
"""

import os
import time
import warnings
import pandas as pd
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Warnungen unterdrücken (deprecated forced_decoder_ids, attention mask)
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")

# ============================================================
# Konfiguration
# ============================================================
MODEL_NAME = "openai/whisper-large-v2"
SAMPLE_TSV = "Data/sample.tsv"
CLIPS_DIR = "Data/clips__test"
OUTPUT_CSV = "Data/transcriptions.csv"

# ============================================================
# Schritt 2.1 – Whisper laden
# ============================================================
print("=" * 60)
print("SCHRITT 2.1 – Whisper-Modell laden")
print("=" * 60)

# Device bestimmen (MPS für Apple Silicon, sonst CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🖥  Device: Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🖥  Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("🖥  Device: CPU (wird langsam!)")

print(f"📦 Lade Modell: {MODEL_NAME} ...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("✅ Whisper-Modell erfolgreich geladen!\n")

# ============================================================
# Schritt 2.2 – Test mit einer einzelnen Datei
# ============================================================
print("=" * 60)
print("SCHRITT 2.2 – Test mit einer Datei")
print("=" * 60)

sample = pd.read_csv(SAMPLE_TSV, sep="\t")
print(f"📄 Sample geladen: {len(sample):,} Zeilen\n")

# Erste Datei testen
test_row = sample.iloc[0]
test_path = os.path.join(CLIPS_DIR, test_row["path"])

y, sr = librosa.load(test_path, sr=16000)
input_features = processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)

forced_decoder_ids = processor.get_decoder_prompt_ids(language="german", task="transcribe")
with torch.no_grad():
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"📁 Datei:          {test_row['path']}")
print(f"🗺  Region:         {test_row['dialect_region']}")
print(f"📝 Original (HD):  {test_row['sentence']}")
print(f"🎤 Transkription:  {transcription}")
print(f"\n✅ Einzeltest erfolgreich!\n")

# ============================================================
# Schritt 2.3 – Alle 1400 Dateien transkribieren
# ============================================================
print("=" * 60)
print("SCHRITT 2.3 – Transkription aller Dateien")
print("=" * 60)

results = []
errors = []
total = len(sample)
start_time = time.time()

for i, (_, row) in enumerate(sample.iterrows()):
    audio_path = os.path.join(CLIPS_DIR, row["path"])

    try:
        y, sr = librosa.load(audio_path, sr=16000)
        input_features = processor(
            y, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        results.append({
            "path": row["path"],
            "dialect_region": row["dialect_region"],
            "sentence": row["sentence"],
            "transcription": transcription,
        })

    except Exception as e:
        errors.append({"path": row["path"], "error": str(e)})
        results.append({
            "path": row["path"],
            "dialect_region": row["dialect_region"],
            "sentence": row["sentence"],
            "transcription": "",
        })

    # Fortschritt alle 50 Dateien
    if (i + 1) % 50 == 0 or (i + 1) == total:
        elapsed = time.time() - start_time
        per_file = elapsed / (i + 1)
        remaining = per_file * (total - i - 1)
        print(
            f"  [{i+1:4d}/{total}]  "
            f"{elapsed:.0f}s vergangen | ~{remaining:.0f}s verbleibend"
        )

# Speichern
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

elapsed_total = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"✅ Fertig!")
print(f"   Erfolgreich: {len(results) - len(errors):,} / {total:,}")
print(f"   Fehler:      {len(errors):,}")
print(f"   Dauer:       {elapsed_total / 60:.1f} Minuten")
print(f"   Gespeichert: {OUTPUT_CSV}")
print(f"{'=' * 60}")
