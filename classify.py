"""
Neuer Schritt 3 – Mapping: Ostschweizer Dialekt zu Hochdeutsch

Liest transcriptions_clean.csv ein, filtert nach der Ostschweiz und
zählt, welche IPA-Wörter (Dialekt) am häufigsten mit welchen
Hochdeutschen Wörtern im selben Satz auftauchen (Co-Occurrence).

Forschungsfrage: Wie lassen sich die häufigsten Wörter des Ostschweiz-Dialekts
systematisch hochdeutschen Äquivalenten zuordnen?
"""

import pandas as pd
from collections import defaultdict, Counter
import re

print("1. Lade Daten und filtere nach Ostschweiz...")
df = pd.read_csv("Data/transcriptions_clean.csv")

# Wir betrachten nur die Ostschweiz
df_ost = df[df["dialect_region"] == "Ostschweiz"].copy()

# Leeres Dictionary für unser Mapping: IPA-Wort -> Zähler für Hochdeutsche Wörter
# Beispiel: mapping["vitə"]["weiter"] = 15
word_mapping = defaultdict(Counter)

print(f"2. Analysiere {len(df_ost)} Sätze auf Kookkurrenzen...")

for index, row in df_ost.iterrows():
    # 1. Hochdeutschen Satz bereinigen (Satzzeichen weg, alles klein)
    hg_sentence = str(row["sentence"]).lower()
    hg_sentence = re.sub(r'[^\w\s]', '', hg_sentence) # Entfernt Punkte, Kommas etc.
    hg_words = hg_sentence.split()

    # 2. Schweizerdeutsches IPA nehmen (ist durch Leerzeichen getrennt)
    ipa_sentence = str(row["ipa_audio"])
    ipa_words = ipa_sentence.split()

    # 3. Kookkurrenz zählen: Jedes IPA-Wort mit jedem Hochdeutschen Wort verknüpfen
    for ipa_w in ipa_words:
        # Kurze Laute (wie "a" oder "i") ignorieren wir oft, da sie Rauschen erzeugen
        if len(ipa_w) > 1:
            for hg_w in hg_words:
                word_mapping[ipa_w][hg_w] += 1

print("\n3. Extrahiere die systematischen Zuordnungen...")
# Wir suchen uns die häufigsten IPA-Wörter und ihr wahrscheinlichstes Hochdeutsches Pendant
systematic_mapping = []

# Sortiere IPA-Wörter danach, wie oft sie generell vorkommen
sorted_ipa = sorted(word_mapping.keys(), key=lambda k: sum(word_mapping[k].values()), reverse=True)

# Nimm die Top 50 häufigsten Ostschweizer IPA-Wörter
for ipa_word in sorted_ipa[:50]:
    # Das häufigste Hochdeutsche Wort, das zusammen mit diesem IPA-Wort auftaucht
    most_common_hg = word_mapping[ipa_word].most_common(1)[0]

    systematic_mapping.append({
        "IPA_Dialekt": ipa_word,
        "Hochdeutsch_Zuordnung": most_common_hg[0],
        "Gemeinsame_Treffer": most_common_hg[1]
    })

# Ergebnisse als DataFrame anzeigen und speichern
results_df = pd.DataFrame(systematic_mapping)
print("\nTop 10 Ergebnisse:")
print(results_df.head(10))

# Speichern für den Bericht
results_df.to_csv("Data/ostschweiz_mapping_results.csv", index=False)
print("\nErfolgreich gespeichert unter Data/ostschweiz_mapping_results.csv")