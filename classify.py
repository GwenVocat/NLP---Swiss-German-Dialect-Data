"""
Schritt 3.1 – Pipeline: IPA-Bereinigung + TF-IDF Berechnung

Laedt transcriptions.csv, bereinigt IPA-Strings und berechnet
TF-IDF fuer beide Transkriptions-Varianten (Whisper + IPA).

Input:  Data/transcriptions.csv
Output: Data/tfidf_results.pkl  (TF-IDF Matrizen, Vektoren, Top-Words)
        Data/transcriptions_clean.csv (mit ipa_clean Spalte)

Verwendung: python classify.py
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. Daten laden
# ============================================================
print("1. Daten laden...")
df = pd.read_csv("Data/transcriptions.csv")
df["transcription_whisper"] = df["transcription_whisper"].fillna("")
df["transcription_ipa"] = df["transcription_ipa"].fillna("")

regions = sorted(df["dialect_region"].unique())
print(f"   {len(df):,} Zeilen, {len(regions)} Regionen: {regions}")





# ============================================================
# 2. TF-IDF berechnen
# ============================================================
print("\n3. TF-IDF berechnen...")

# --- 3a: Whisper (Wort-basiert) ---
print("   Whisper (Wort-TF-IDF)...")
region_docs_whisper = df.groupby("dialect_region")["transcription_whisper"].apply(" ".join)

vec_whisper = TfidfVectorizer(max_features=5000)
tfidf_whisper = vec_whisper.fit_transform(region_docs_whisper)
features_whisper = vec_whisper.get_feature_names_out()

top_whisper = {}
for i, region in enumerate(region_docs_whisper.index):
    scores = tfidf_whisper[i].toarray().flatten()
    top_idx = scores.argsort()[-20:][::-1]
    top_whisper[region] = [(features_whisper[j], float(scores[j])) for j in top_idx]

# --- 3b: IPA (Character N-Grams, bereinigt) ---
print("   IPA (Character N-Gram TF-IDF, bereinigt)...")
region_docs_ipa = df.groupby("dialect_region")["ipa_clean"].apply(" ".join)

vec_ipa = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=5000,
)
tfidf_ipa = vec_ipa.fit_transform(region_docs_ipa)
features_ipa = vec_ipa.get_feature_names_out()

top_ipa = {}
for i, region in enumerate(region_docs_ipa.index):
    scores = tfidf_ipa[i].toarray().flatten()
    top_idx = scores.argsort()[-20:][::-1]
    top_ipa[region] = [(features_ipa[j], float(scores[j])) for j in top_idx]


# ============================================================
# 3. Ergebnisse speichern
# ============================================================
print("\n4. Ergebnisse speichern...")

results = {
    # Top-Words pro Region
    "top_whisper": top_whisper,
    "top_ipa": top_ipa,
    # TF-IDF Matrizen (fuer Heatmap etc.)
    "tfidf_whisper": tfidf_whisper,
    "tfidf_ipa": tfidf_ipa,
    # Vectorizer (fuer spaetere Klassifikation)
    "vec_whisper": vec_whisper,
    "vec_ipa": vec_ipa,
    # Region-Labels
    "regions_whisper": region_docs_whisper.index.tolist(),
    "regions_ipa": region_docs_ipa.index.tolist(),
}

with open("Data/tfidf_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("   Gespeichert: Data/tfidf_results.pkl")

# Kurze Zusammenfassung
print("\n" + "=" * 50)
print("Fertig! Zusammenfassung:")
print("=" * 50)
for region in sorted(top_whisper):
    w_top3 = ", ".join(w[0] for w in top_whisper[region][:3])
    i_top3 = ", ".join(f'"{w[0]}"' for w in top_ipa[region][:3])
    print(f"  {region}:")
    print(f"    Whisper: {w_top3}")
    print(f"    IPA:     {i_top3}")
