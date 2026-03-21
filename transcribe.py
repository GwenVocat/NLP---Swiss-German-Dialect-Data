"""
Whisper-Transkription aller 1400 Sample-Dateien mit 2 Modellen

Modell A: openai/whisper-large-v2 → Hochdeutsch (Baseline)
Modell B: neurlang/ipa-whisper-base → IPA-Phoneme (Dialekt-Signal)

Die IPA-Transkription zeigt die tatsächliche Aussprache – Dialektunterschiede
werden direkt sichtbar (z.B. Walliser /frˈyːɛɪk/ vs. Zürcher "ruhig").

Verwendung: python transcribe.py
Dauer:      ~45-60 Min auf Apple Silicon (MPS)
"""

import os
import time
import warnings
import pandas as pd
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Warnungen unterdrücken
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# ============================================================
# Konfiguration
# ============================================================
MODEL_A = "openai/whisper-large-v2"
MODEL_B = "neurlang/ipa-whisper-base"
SAMPLE_TSV = "Data/sample.tsv"
CLIPS_DIR = "Data/clips__test"
OUTPUT_CSV = "Data/transcriptions.csv"

# ============================================================
# Device
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🖥  Device: Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🖥  Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("🖥  Device: CPU (wird langsam!)")

# ============================================================
# Schritt 2.1 – Beide Modelle laden
# ============================================================
print("=" * 60)
print("SCHRITT 2.1 – Modelle laden")
print("=" * 60)

# Modell A: Whisper large-v2 → Hochdeutsch
print(f"\n📦 Lade Modell A: {MODEL_A} ...")
processor_a = WhisperProcessor.from_pretrained(MODEL_A)
model_a = WhisperForConditionalGeneration.from_pretrained(MODEL_A).to(device)
model_a.eval()
forced_decoder_ids = processor_a.get_decoder_prompt_ids(
    language="german", task="transcribe"
)
print("✅ Modell A geladen (Whisper → Hochdeutsch)")

# Modell B: IPA Whisper → Phoneme
print(f"\n📦 Lade Modell B: {MODEL_B} ...")
processor_b = WhisperProcessor.from_pretrained(MODEL_B)
model_b = WhisperForConditionalGeneration.from_pretrained(MODEL_B).to(device)
model_b.eval()
print("✅ Modell B geladen (Whisper → IPA)\n")

# ============================================================
# Schritt 2.2 – Test mit einer einzelnen Datei
# ============================================================
print("=" * 60)
print("SCHRITT 2.2 – Test mit einer Datei (beide Modelle)")
print("=" * 60)

sample = pd.read_csv(SAMPLE_TSV, sep="\t")
print(f"📄 Sample geladen: {len(sample):,} Zeilen\n")

test_row = sample.iloc[0]
test_path = os.path.join(CLIPS_DIR, test_row["path"])
y, sr = librosa.load(test_path, sr=16000)

# Modell A
input_features = processor_a(
    y, sampling_rate=16000, return_tensors="pt"
).input_features.to(device)
with torch.no_grad():
    predicted_ids = model_a.generate(
        input_features, forced_decoder_ids=forced_decoder_ids
    )
text_a = processor_a.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

# Modell B
input_features_b = processor_b(
    y, sampling_rate=16000, return_tensors="pt"
).input_features.to(device)
with torch.no_grad():
    predicted_ids_b = model_b.generate(input_features_b)
text_b = processor_b.batch_decode(predicted_ids_b, skip_special_tokens=True)[0].strip()

print(f"📁 Datei:              {test_row['path']}")
print(f"🗺  Region:             {test_row['dialect_region']}")
print(f"📝 Original (HD):      {test_row['sentence']}")
print(f"🔵 A (Hochdeutsch):    {text_a}")
print(f"🟡 B (IPA):            {text_b}")
print(f"\n✅ Einzeltest erfolgreich!\n")

# ============================================================
# Schritt 2.3 – Alle 1400 Dateien transkribieren (beide Modelle)
# ============================================================
print("=" * 60)
print("SCHRITT 2.3 – Transkription aller Dateien (2 Modelle)")
print("=" * 60)

results = []
errors = []
total = len(sample)
start_time = time.time()

for i, (_, row) in enumerate(sample.iterrows()):
    audio_path = os.path.join(CLIPS_DIR, row["path"])

    text_a = ""
    text_b = ""

    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # Modell A: Whisper → Hochdeutsch
        input_features = processor_a(
            y, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model_a.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )
        text_a = processor_a.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        # Modell B: Whisper → IPA
        input_features_b = processor_b(
            y, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids_b = model_b.generate(input_features_b)
        text_b = processor_b.batch_decode(
            predicted_ids_b, skip_special_tokens=True
        )[0].strip()

    except Exception as e:
        errors.append({"path": row["path"], "error": str(e)})

    results.append({
        "path": row["path"],
        "dialect_region": row["dialect_region"],
        "sentence": row["sentence"],
        "transcription_whisper": text_a,
        "transcription_ipa": text_b,
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
print(f"\n📊 Output-Spalten:")
print(f"   transcription_whisper → Hochdeutsch (Baseline)")
print(f"   transcription_ipa     → IPA-Phoneme (Dialekt-Signal)")
