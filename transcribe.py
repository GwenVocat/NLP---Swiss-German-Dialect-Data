"""
Whisper-Transkription aller 1400 Sample-Dateien mit 2 Modellen

Modell A: openai/whisper-large-v2 (Standard)
Modell B: Flurin17/whisper-large-v3-turbo-swiss-german (Swiss German fine-tuned)

Beide Transkriptionen werden nebeneinander gespeichert, damit man in der
Analyse vergleichen kann, welches Modell besser für die Dialekterkennung ist.

Verwendung: python transcribe.py
Dauer:      ~45-60 Min auf Apple Silicon (MPS)
"""

import os
import time
import warnings
import pandas as pd
import librosa
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

# Warnungen unterdrücken
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# ============================================================
# Konfiguration
# ============================================================
MODEL_A = "openai/whisper-large-v2"
MODEL_B = "Flurin17/whisper-large-v3-turbo-swiss-german"
SAMPLE_TSV = "Data/sample.tsv"
CLIPS_DIR = "Data/clips__test"
OUTPUT_CSV = "Data/transcriptions.csv"

# ============================================================
# Device
# ============================================================
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32  # MPS braucht float32
    print("🖥  Device: Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
    print("🖥  Device: CUDA GPU")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("🖥  Device: CPU (wird langsam!)")

# ============================================================
# Schritt 2.1 – Beide Modelle laden
# ============================================================
print("=" * 60)
print("SCHRITT 2.1 – Modelle laden")
print("=" * 60)

# Modell A: whisper-large-v2
print(f"\n📦 Lade Modell A: {MODEL_A} ...")
processor_a = WhisperProcessor.from_pretrained(MODEL_A)
model_a = WhisperForConditionalGeneration.from_pretrained(MODEL_A).to(device)
model_a.eval()
forced_decoder_ids = processor_a.get_decoder_prompt_ids(
    language="german", task="transcribe"
)
print("✅ Modell A geladen")

# Modell B: Swiss German fine-tuned
print(f"\n📦 Lade Modell B: {MODEL_B} ...")
model_b = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_B, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor_b = AutoProcessor.from_pretrained(MODEL_B)
pipe_b = pipeline(
    "automatic-speech-recognition",
    model=model_b,
    tokenizer=processor_b.tokenizer,
    feature_extractor=processor_b.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
print("✅ Modell B geladen\n")

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

# Modell A
y, sr = librosa.load(test_path, sr=16000)
input_features = processor_a(
    y, sampling_rate=16000, return_tensors="pt"
).input_features.to(device)
with torch.no_grad():
    predicted_ids = model_a.generate(
        input_features, forced_decoder_ids=forced_decoder_ids
    )
text_a = processor_a.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

# Modell B
result_b = pipe_b(test_path)
text_b = result_b["text"].strip()

print(f"📁 Datei:              {test_row['path']}")
print(f"🗺  Region:             {test_row['dialect_region']}")
print(f"📝 Original (HD):      {test_row['sentence']}")
print(f"🔵 A (large-v2):       {text_a}")
print(f"🟢 B (Swiss-German):   {text_b}")
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
        # Modell A: whisper-large-v2
        y, sr = librosa.load(audio_path, sr=16000)
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

        # Modell B: Swiss German fine-tuned
        result = pipe_b(audio_path)
        text_b = result["text"].strip()

    except Exception as e:
        errors.append({"path": row["path"], "error": str(e)})

    results.append({
        "path": row["path"],
        "dialect_region": row["dialect_region"],
        "sentence": row["sentence"],
        "transcription_v2": text_a,
        "transcription_swiss": text_b,
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
print(f"   transcription_v2    → Whisper large-v2 (Standard)")
print(f"   transcription_swiss → Swiss German fine-tuned")
print(f"   → In classify.py beide Spalten vergleichen!")
