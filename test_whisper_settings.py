"""
A/B-Test: Whisper large-v2 vs. Swiss German fine-tuned Modell
Testet 1 Datei pro Region (7 total) mit 2 Modellen.
"""

import os
import pandas as pd
import librosa
import torch
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
)

# Setup
CLIPS_DIR = "Data/clips__test"
sample = pd.read_csv("Data/sample.tsv", sep="\t")

# 1 Datei pro Region
test_files = sample.groupby("dialect_region").first().reset_index()
print(f"Teste {len(test_files)} Dateien (1 pro Region)\n")

# Device
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32  # MPS braucht float32
elif torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32
print(f"Device: {device}\n")

# ============================================================
# Modell A: Whisper large-v2 (aktuell)
# ============================================================
print("=" * 80)
print("Lade Modell A: openai/whisper-large-v2 ...")
processor_a = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model_a = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
model_a.eval()
forced_decoder_ids = processor_a.get_decoder_prompt_ids(language="german", task="transcribe")
print("✅ Modell A geladen\n")

# ============================================================
# Modell B: Swiss German fine-tuned (Flurin17)
# ============================================================
print("Lade Modell B: Flurin17/whisper-large-v3-turbo-swiss-german ...")
model_id_b = "Flurin17/whisper-large-v3-turbo-swiss-german"
model_b = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id_b, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor_b = AutoProcessor.from_pretrained(model_id_b)

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
# Vergleich
# ============================================================
print("=" * 80)
print("VERGLEICH: 1 Datei pro Region")
print("=" * 80)

for _, row in test_files.iterrows():
    audio_path = os.path.join(CLIPS_DIR, row["path"])
    y, sr = librosa.load(audio_path, sr=16000)

    # Modell A: whisper-large-v2
    input_features = processor_a(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = model_a.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    text_a = processor_a.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    # Modell B: Swiss German fine-tuned
    result_b = pipe_b(audio_path)
    text_b = result_b["text"].strip()

    print(f"\n🗺  Region:            {row['dialect_region']}")
    print(f"📝 Original (HD):     {row['sentence']}")
    print(f"🔵 A (large-v2):      {text_a}")
    print(f"🟢 B (Swiss-German):  {text_b}")
    print("-" * 80)

print("\n✅ Fertig! Vergleiche A vs. B – welches zeigt mehr Dialekt?")
