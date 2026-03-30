"""
Whisper-Transkription aller 1400 Sample-Dateien mit 2 Modellen
+ Wort-Alignment via phonetische Distanz

Modell A: openai/whisper-large-v2 → Hochdeutsch (Baseline)
Modell B: neurlang/ipa-whisper-base → IPA-Phoneme (Dialekt-Signal)

Schritt 2.4: Referenzsatz (HD) → IPA, dann Wort-Alignment mit gesprochener
IPA via Levenshtein-Distanz. So wird sichtbar, welches Dialektwort welchem
Hochdeutschen Wort entspricht und wie gross die Abweichung ist.

Verwendung: python transcribe.py
Dauer:      ~45-60 Min auf Apple Silicon (MPS)
"""

import os
import json
import time
import warnings
import pandas as pd
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from phonemizer import phonemize
from phonemizer.separator import Separator
import Levenshtein

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

# Zwischenspeichern (vor Alignment)
df_results = pd.DataFrame(results)

elapsed_total = time.time() - start_time
print(f"\n{'=' * 60}")
print(f"Transkription abgeschlossen!")
print(f"   Erfolgreich: {len(results) - len(errors):,} / {total:,}")
print(f"   Fehler:      {len(errors):,}")
print(f"   Dauer:       {elapsed_total / 60:.1f} Minuten")
print(f"{'=' * 60}")

# ============================================================
# Schritt 2.4 – Wort-Alignment: Referenz-IPA ↔ gesprochene IPA
# ============================================================
print("\n" + "=" * 60)
print("SCHRITT 2.4 – Wort-Alignment (Referenz-IPA ↔ gesprochene IPA)")
print("=" * 60)

IPA_SEP = Separator(phone=" ", word="  ", syllable="")


def sentence_to_ipa(text):
    """Deutschen Satz Wort-für-Wort in IPA umwandeln via espeak-ng."""
    ipa = phonemize(
        text,
        language="de",
        backend="espeak",
        separator=IPA_SEP,
        strip=True,
        preserve_punctuation=False,
    )
    return ipa


def align_words(ref_ipa, spoken_ipa):
    """
    Wort-Alignment via minimale Levenshtein-Distanz.

    Für jedes gesprochene IPA-Wort wird das ähnlichste Referenz-IPA-Wort
    gesucht (greedy, ohne Wiederverwendungs-Einschränkung).

    Returns: Liste von Dicts mit ref_word, spoken_word, distance
    """
    ref_words = ref_ipa.split()
    spoken_words = spoken_ipa.split()

    if not ref_words or not spoken_words:
        return []

    alignment = []
    for sp_word in spoken_words:
        best_ref = None
        best_dist = float("inf")
        for r_word in ref_words:
            dist = Levenshtein.distance(sp_word, r_word)
            if dist < best_dist:
                best_dist = dist
                best_ref = r_word
        alignment.append({
            "ref_ipa": best_ref,
            "spoken_ipa": sp_word,
            "distance": best_dist,
        })

    return alignment


# Referenzsätze → IPA (Batch für Performance)
print("\nKonvertiere Referenzsätze → IPA (espeak-ng) ...")
sentences = df_results["sentence"].tolist()
sentences_ipa_raw = phonemize(
    sentences,
    language="de",
    backend="espeak",
    separator=IPA_SEP,
    strip=True,
    preserve_punctuation=False,
)
# phonemize gibt bei Listen eine Liste zurück
if isinstance(sentences_ipa_raw, str):
    sentences_ipa_raw = [sentences_ipa_raw]

df_results["sentence_ipa"] = sentences_ipa_raw
print(f"   {len(sentences_ipa_raw):,} Sätze konvertiert.")

# Wort-Alignment berechnen
print("Berechne Wort-Alignments ...")
alignments = []
for _, row in df_results.iterrows():
    ref_ipa = row.get("sentence_ipa", "")
    spoken_ipa = row.get("transcription_ipa", "")
    if ref_ipa and spoken_ipa:
        al = align_words(str(ref_ipa), str(spoken_ipa))
        alignments.append(json.dumps(al, ensure_ascii=False))
    else:
        alignments.append("[]")

df_results["word_alignment"] = alignments

# Mittlere Distanz pro Zeile als schnelle Metrik
def mean_distance(alignment_json):
    al = json.loads(alignment_json)
    if not al:
        return None
    return round(sum(a["distance"] for a in al) / len(al), 2)

df_results["mean_ipa_distance"] = df_results["word_alignment"].apply(mean_distance)

# Speichern
df_results.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'=' * 60}")
print(f"✅ Fertig!")
print(f"   Gespeichert: {OUTPUT_CSV}")
print(f"{'=' * 60}")
print(f"\n📊 Output-Spalten:")
print(f"   transcription_whisper → Hochdeutsch (Baseline)")
print(f"   transcription_ipa     → IPA-Phoneme (Dialekt-Signal)")
print(f"   sentence_ipa          → Referenzsatz in IPA (espeak-ng)")
print(f"   word_alignment        → JSON: Wortpaare + Levenshtein-Distanz")
print(f"   mean_ipa_distance     → Mittlere Distanz pro Satz")
