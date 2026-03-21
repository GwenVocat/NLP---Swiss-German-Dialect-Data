"""
A/B/C-Test: 3 Modelle im Vergleich
  A: whisper-large-v2 → Hochdeutsch (Baseline)
  B: Flurin17/whisper-large-v3-turbo-swiss-german → Hochdeutsch (fine-tuned)
  C: facebook/wav2vec2-lv-60-espeak-cv-ft → IPA-Phoneme (sprachunabhängig!)

Testet 1 Datei pro Region (7 total).
Idee: IPA-Output zeigt die tatsächliche Aussprache – Dialektunterschiede
werden direkt sichtbar (z.B. Walliser /ʃ/ vs. Zürcher /x/).
"""

import os
import pandas as pd
import librosa
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)
import warnings

warnings.filterwarnings("ignore")

# Setup
CLIPS_DIR = "Data/clips__test"
sample = pd.read_csv("Data/sample.tsv", sep="\t")

# 1 Datei pro Region
test_files = sample.groupby("dialect_region").first().reset_index()
print(f"Teste {len(test_files)} Dateien (1 pro Region)\n")

# Device
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
elif torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32
print(f"Device: {device}\n")

# ============================================================
# Modell A: Whisper large-v2 (Standard → Hochdeutsch)
# ============================================================
print("=" * 80)
print("Lade Modell A: openai/whisper-large-v2 ...")
processor_a = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model_a = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2"
).to(device)
model_a.eval()
forced_decoder_ids = processor_a.get_decoder_prompt_ids(
    language="german", task="transcribe"
)
print("✅ Modell A geladen\n")

# ============================================================
# Modell B: Swiss German fine-tuned Whisper (→ Hochdeutsch)
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
# Modell C: Wav2Vec2 Phoneme → IPA (sprachunabhängig)
# facebook/wav2vec2-lv-60-espeak-cv-ft
# ============================================================
print("Lade Modell C: facebook/wav2vec2-lv-60-espeak-cv-ft (IPA) ...")
model_id_c = "facebook/wav2vec2-lv-60-espeak-cv-ft"
processor_c = Wav2Vec2Processor.from_pretrained(model_id_c)
model_c = Wav2Vec2ForCTC.from_pretrained(model_id_c).to(device)
model_c.eval()
print("✅ Modell C geladen\n")

# ============================================================
# Modell D: neurlang/ipa-whisper-base (Whisper → IPA)
# ============================================================
model_d = None
processor_d = None
model_id_d = "neurlang/ipa-whisper-base"
try:
    print(f"Lade Modell D: {model_id_d} ...")
    processor_d = WhisperProcessor.from_pretrained(model_id_d)
    model_d = WhisperForConditionalGeneration.from_pretrained(model_id_d).to(device)
    model_d.eval()
    print("✅ Modell D geladen\n")
except Exception as e:
    print(f"⚠️  Modell D nicht verfügbar: {type(e).__name__} – überspringe\n")

# ============================================================
# Vergleich
# ============================================================
print("=" * 80)
print("VERGLEICH: 1 Datei pro Region")
print("=" * 80)

for _, row in test_files.iterrows():
    audio_path = os.path.join(CLIPS_DIR, row["path"])
    y, sr = librosa.load(audio_path, sr=16000)

    # Modell A: Whisper large-v2 → Hochdeutsch
    input_features = processor_a(
        y, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        predicted_ids = model_a.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )
    text_a = processor_a.batch_decode(predicted_ids, skip_special_tokens=True)[
        0
    ].strip()

    # Modell B: Swiss German Whisper → Hochdeutsch
    result_b = pipe_b(audio_path)
    text_b = result_b["text"].strip()

    # Modell C: Wav2Vec2 Phoneme → IPA
    input_values = processor_c(
        y, sampling_rate=16000, return_tensors="pt"
    ).input_values.to(device)
    with torch.no_grad():
        logits = model_c(input_values).logits
    predicted_ids_c = torch.argmax(logits, dim=-1)
    text_c = processor_c.batch_decode(predicted_ids_c)[0].strip()

    # Modell D: Whisper IPA (falls verfügbar)
    text_d = "(nicht verfügbar)"
    if model_d is not None:
        input_features_d = processor_d(
            y, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids_d = model_d.generate(input_features_d)
        text_d = processor_d.batch_decode(predicted_ids_d, skip_special_tokens=True)[
            0
        ].strip()

    print(f"\n🗺  Region:                {row['dialect_region']}")
    print(f"📝 Original (HD):         {row['sentence']}")
    print(f"🔵 A (Whisper v2):        {text_a}")
    print(f"🟢 B (Whisper Swiss):     {text_b}")
    print(f"🟡 C (Wav2Vec2 IPA):      {text_c}")
    print(f"🟠 D (Whisper IPA):       {text_d}")
    print("-" * 80)

print("\n✅ Fertig!")
print("\nLegende:")
print("  A = Whisper large-v2 → Hochdeutsch (Baseline)")
print("  B = Whisper Swiss German fine-tuned → Hochdeutsch")
print("  C = facebook/wav2vec2-lv-60-espeak-cv-ft → IPA-Phoneme")
print(f"  D = {model_id_d} → IPA-Phoneme (Whisper-basiert)")
print()
print("→ Vergleiche die IPA-Outputs (C/D) zwischen den Regionen!")
print("  Wenn z.B. Wallis andere Phoneme zeigt als Zürich → das ist unser Signal.")
print("  IPA-Unterschiede lassen sich mit TF-IDF/n-gram-Analyse nutzen.")
