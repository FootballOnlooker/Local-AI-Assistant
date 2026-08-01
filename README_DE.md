![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![License](https://img.shields.io/badge/Usage-Educational-green)

# Lokaler KI-Assistent

🇬🇧 **English version:** [README.md](README.md)

Ein lokaler Desktop-KI-Assistent zur Verarbeitung deutscher Kundenanfragen.

Die Anwendung kombiniert drei KI-Komponenten:

1. ein lokal ausgeführtes Sprachmodell über **Ollama**,
2. einen fein abgestimmten **German DistilBERT-Klassifikator**,
3. ein hybrides lokales **Retrieval-System**: exakter Abgleich von Referenz-Codes plus mehrsprachige
   Satz-Embeddings über einzelne Dokument-Abschnitte.

Die Tkinter-Oberfläche zeigt die Benutzereingabe, die vorhergesagte Kategorie mit Confidence Score sowie eine
KI-generierte Antwort auf Grundlage der lokalen Wissensdatenbank an.

---

# Funktionen

- Vollständig lokale Ausführung ohne Cloud-LLM
- Desktop-Anwendung mit Tkinter
- Klassifikation deutscher Kundenanfragen
- Vier Kategorien:
  - Anfrage
  - Reklamation
  - Rechnung
  - Sonstiges
- Confidence Score auf Basis der höchsten Softmax-Wahrscheinlichkeit
- Lokale Wissensdatenbank aus Textdateien
- Hybrides Retrieval: exakter Referenz-Code-Abgleich + mehrsprachige Satz-Embeddings
  (`sentence-transformers`) auf Absatzebene mit Cosine Similarity
- Gedächtnis über mehrere Nachrichten hinweg sowie dauerhafte Benutzeranweisungen
  (z. B. „Antworte ab jetzt immer in maximal drei Sätzen")
- Sprachanpassung: antwortet in der Sprache der aktuellen Nachricht (primär Deutsch,
  mit zuverlässiger Unterstützung für Englisch)
- Kontextbasierte Antworten mit Llama 3.2
- Separate Skripte für Training, Vorhersage, Datensatzprüfung, Datensatz-Generierung und Retrieval-Tests

---

# Architektur

```text
Benutzernachricht
        │
        ├──► DistilBERT-Klassifikator
        │         └──► Kategorie + Confidence
        │
        └──► Hybrides Retrieval
                    ├──► exakter Referenz-Code-Abgleich (Regex)       ─┐
                    └──► Embedding-Suche auf Absatzebene              ─┤
                                                                       ▼
                                                       relevantestes Dokument
                                                                       │
                                                                       ▼
                                                          Ollama / Llama 3.2
                                                                       │
                                                                       ▼
                                                                  KI-Antwort

Alle Ergebnisse werden in der Tkinter-Oberfläche angezeigt.
```

---

# Projektstruktur

```text
Local-AI-Assistant/
│
├── teil1/
│   ├── knowledge/
│   │   ├── garantie.txt
│   │   ├── kontakt.txt
│   │   ├── lieferung.txt
│   │   ├── rechnung.txt
│   │   └── reklamation.txt
│   ├── chat.py
│   ├── gui.py
│   ├── main.py
│   └── retrieval.py
│
├── teil2/
│   ├── data/
│   │   └── dataset.csv
│   ├── model/
│   │   └── final_model/          # entsteht nach dem Training
│   ├── check_dataset.py
│   ├── classifier.py
│   ├── predict.py
│   └── train.py
│
├── images/
│
├── README.md
├── README_DE.md
└── requirements.txt
```

---

# Teil 1 – Lokaler KI-Chat

## Ziel

Entwicklung eines Desktop-Assistenten, der Kundenfragen mithilfe einer lokalen Wissensdatenbank beantwortet, sich
über mehrere Nachrichten hinweg an den bisherigen Gesprächsverlauf erinnert und dauerhafte Anweisungen befolgt.

## Verwendete Technologien

- Python
- Tkinter
- Ollama
- Llama 3.2:3b
- sentence-transformers
- Scikit-learn

---

# Retrieval

Vor jeder Anfrage wird zunächst das passendste Dokument aus der Wissensdatenbank ermittelt. Das Retrieval ist ein
**Hybrid** aus zwei Strategien, in dieser Reihenfolge:

**1. Exakter Referenz-Code-Abgleich (Regex).** Referenz-Codes wie Sendungs-, Rechnungs- oder Policennummern
(`EXP-70531`, `RE-2024-1187`, `VP-30987`) sind kurze, seltene alphanumerische Tokens. In Tests haben semantische
Embeddings diese nicht zuverlässig gefunden — die Ähnlichkeitswerte lagen knapp unter der Relevanzschwelle, weil
Embedding-Modelle auf natürlichsprachliche Bedeutung trainiert sind, nicht auf exakte kurze Codes. Eine Regex
(`[A-Z]{2,5}-[0-9][0-9A-Z-]+`) extrahiert einen solchen Code aus der Frage und prüft zuerst auf einen wörtlichen
(Groß-/Kleinschreibung wird ignoriert) Treffer über alle Wissensdokumente. Kommt derselbe Code in mehreren Dokumenten
vor, werden alle passenden Dokumente klar getrennt in den Kontext übernommen, statt willkürlich nur das erste zu
wählen.

**2. Semantische Suche auf Absatzebene (Satz-Embeddings).** Wird kein Referenz-Code in der Frage gefunden, greift die
Anwendung auf `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) zurück. Modell und Embeddings werden
einmalig beim Start berechnet, nicht bei jeder Nachricht neu.

Dokumente werden dabei **nicht** als Ganzes eingebettet, sondern zunächst in Absätze aufgeteilt
(`chunk_documents()`), jeder Absatz einzeln kodiert, und die Frage wird gegen diese Absätze statt gegen ganze
Dokumente verglichen. Das ist eine gezielte Korrektur eines real gemessenen Problems: Das Embedding-Modell hat ein
`max_seq_length` von 128 Tokens. Mehrere Wissensdokumente (100–190 Wörter, nach deutscher Subword-Tokenisierung
grob 130–300 Tokens) überschritten dieses Limit — Inhalte gegen Ende eines Dokuments wurden dadurch beim Kodieren
stillschweigend abgeschnitten und waren für die Suche unsichtbar, unabhängig davon, wie wichtig sie waren. Die
Aufteilung in Absätze hält jede Sucheinheit sicher unter dem Limit (geprüft: der längste Absatz in der aktuellen
Wissensdatenbank liegt bei geschätzt ~91 Tokens), ohne das zu verkleinern, was das Sprachmodell am Ende sieht — bei
einem Treffer wird weiterhin der **vollständige** Dokumenttext als Kontext zurückgegeben.

Die Suchanfrage kombiniert die aktuelle Frage mit den letzten zwei Nutzer-Nachrichten, damit Folgefragen wie
„Wie lange würde der Transport *dorthin* dauern?" trotzdem dem richtigen Thema zugeordnet werden können, ohne den
Ort erneut zu nennen.

```text
Wissensdokumente
      │
      └──► Aufteilung in Absätze (chunk_documents)
                  │
                  └──► Absatz-Embeddings (einmalig beim Start berechnet)

Benutzerfrage
      │
      └──► Referenz-Code in der Frage?
                  ├─ ja  ──► exakter Teilstring-Abgleich über alle Dokumente
                  └─ nein ──► Frage-Embedding
                                    │
                                    └──► Cosine Similarity zu den Absatz-Embeddings
                                                │
                                                └──► bester Treffer
                                                            │
                                                            └──► vollständiges Ursprungsdokument als Kontext
```

Für den semantischen Pfad wird ein Mindest-Ähnlichkeitswert (`MIN_SIMILARITY = 0.35`) verwendet — kalibriert anhand
dreier echter Testfragen, nicht geschätzt: eine Frage nah am Dokumentwortlaut erzielte ~0,69; eine inhaltlich
relevante, aber anders formulierte Frage („Ich habe meine Rechnung nicht erhalten" gegenüber der formaleren
Dokumentformulierung) erzielte ~0,41; eine thematisch irrelevante Frage erzielte ~0,26. Der Schwellenwert liegt mit
Sicherheitsabstand über dem irrelevanten und unter beiden relevanten Fällen.

Wird kein ausreichend ähnlicher Absatz gefunden und auch kein Referenz-Code erkannt, weist die Anwendung das
Sprachmodell ausdrücklich darauf hin, dass keine dokumentierte Information vorliegt, statt es eine Antwort erfinden
zu lassen.

Als zusätzliche Absicherung führt die Anwendung außerdem eine kleine deterministische Prüfung durch: Sie liest die
in `lieferung.txt` tatsächlich dokumentierten Städte aus und ergänzt, falls nach einer Lieferung in eine nicht
gelistete Stadt gefragt wird, eine kurze, eigenständige Anweisung an das Sprachmodell, für diese Stadt keine
Lieferzeit zu erfinden. Diese Prüfung ist bewusst eine Ergänzung zur Prompt-Anweisung, kein vollständiger Ersatz des
Sprachmodells — rein prompt-basierte Anweisungen wurden vom 3B-Modell in Tests nicht durchgängig zuverlässig
befolgt.

---

# Teil 2 – Deutsche Textklassifikation

## Ziel

Feinabstimmung eines DistilBERT-Modells zur Klassifikation deutscher Kundenanfragen.

## Kategorien

- Anfrage
- Reklamation
- Rechnung
- Sonstiges

---

# Datensatz

Der Datensatz besteht aus handgeschriebenen Satzvorlagen, wobei jede Vorlage **genau einmal** verwendet wird
(zum Zeitpunkt der Erstellung programmatisch geprüft, nicht nur behauptet):

- 180 deutsche Textbeispiele
- 4 ausgewogene Kategorien
- jeweils 45 Beispiele pro Kategorie
- keine fehlenden Werte
- keine Duplikate
- **100 % eindeutige Vorlagen** — bestätigt durch Normalisierung von Städten/Referenz-Codes zu Platzhaltern und
  Prüfung auf Kollisionen; es gibt keine

Eine frühere Version dieses Datensatzes (800 Beispiele) verwendete denselben Satzbau mit lediglich ausgetauschter
Stadt oder Referenznummer, teils bis zu 10-mal. Das führte zu einer irreführend perfekten Testgenauigkeit von 100 %,
weil ein zufälliger Split nahezu identische Sätze auf beide Seiten von Training und Test verteilen konnte.
`train.py` verwendet `StratifiedGroupKFold`, um dies strukturell zu verhindern — unabhängig davon, wie der
Datensatz aufgebaut ist. Zusätzlich wurde der Datensatz selbst so neu geschrieben, dass er an der Quelle bereits
eindeutig ist — beide Absicherungen zusammen statt nur einer.

---

# Trainingsablauf

1. CSV-Datensatz laden und überprüfen
2. Kategorien in numerische IDs umwandeln
3. Vorlagen-bewussten, stratifizierten Train/Test-Split erstellen (`StratifiedGroupKFold`), damit
   Vorlagen-Duplikate nicht zwischen Training und Test „durchsickern"
4. Hugging Face Dataset erzeugen
5. Texte tokenisieren
6. DistilBERT feinabstimmen, bis zu 6 Epochen mit Early Stopping (Geduld 2) und automatischer Auswahl des besten
   Checkpoints nach Genauigkeit
7. Modell evaluieren, inklusive Klassifikationsbericht (Precision/Recall/F1 je Kategorie) und einer gespeicherten
   Konfusionsmatrix (`teil2/confusion_matrix.png`)
8. Modell und Tokenizer lokal speichern

---

# Trainingsergebnisse

Diese Ergebnisse stammen vom finalen Datensatz (180 vollständig eindeutige Beispiele, `StratifiedGroupKFold`-Split)
— keine Vorlagen-Duplikate zwischen Training und Test. Die Genauigkeit liegt nahe am allerersten, kleineren
Testlauf mit 200 Beispielen (87,5 %), was ein gutes unabhängiges Indiz dafür ist, dass das Modell tatsächlich
verallgemeinert und nicht Satzvorlagen auswendig lernt.

| Parameter | Wert |
|-----------|------|
| Basismodell | `distilbert-base-german-cased` |
| Datensatzgröße | 180 |
| Kategorien | 4 |
| Trainingsdaten | 144 |
| Testdaten | 36 |
| Maximal konfigurierte Epochen | 6 |
| Tatsächlich trainierte Epochen | 6,0 |
| Batch Size | 8 |
| Learning Rate | `2e-5` |
| Trainingszeit | 2,10 min |
| Evaluation Loss | 0,6385 |
| Test Accuracy | **86,11 %** |

Die Ergebnisse können je nach System geringfügig variieren. Siehe `teil2/confusion_matrix.png` für eine visuelle
Aufschlüsselung, welche Kategorien miteinander verwechselt werden.

**Klassifikationsbericht:**

| Kategorie   | Precision | Recall | F1-Score | Support |
|-------------|----------:|-------:|---------:|--------:|
| Anfrage     |      0,88 |   0,78 |     0,82 |       9 |
| Reklamation |      0,89 |   0,89 |     0,89 |       9 |
| Rechnung    |      0,88 |   0,78 |     0,82 |       9 |
| Sonstiges   |      0,82 |   1,00 |     0,90 |       9 |

`Sonstiges` hat einen perfekten Recall (jede tatsächliche `Sonstiges`-Nachricht wurde gefunden), aber eine
niedrigere Precision — laut Konfusionsmatrix stammt der größte Anteil davon aus 2 `Anfrage`-Beispielen, die
fälschlich als `Sonstiges` eingeordnet wurden. Das erklärt gleichzeitig den niedrigeren Recall von `Anfrage`
(7 von 9). Die übrigen Fehler sind eher verstreut: `Rechnung` hatte je ein Beispiel, das als `Anfrage` bzw.
`Reklamation` fehlklassifiziert wurde, `Reklamation` ein Beispiel, das als `Rechnung` eingeordnet wurde. Keine
Kategorie wird durchgängig mit genau einer anderen verwechselt — `Sonstiges` ist als semantisch diffuseste Kategorie
der Hauptanziehungspunkt für Grenzfälle.

---

# Beispiel

## Eingabe

```text
Welche Rechnungsnummer gehört zur Sendung EXP-70531?
```

## Ausgabe

```text
Kategorie:
Rechnung (98,76 %)

KI:
Laut der Dokumentation gehört zur Sendung EXP-70531 die Rechnungsnummer RE-2024-1187.
```

Der Klassifikator und das Retrieval arbeiten unabhängig voneinander. Der Klassifikator bestimmt lediglich die
Nachrichtenkategorie, während das Retrieval das passende Dokument für den Kontext des Sprachmodells auswählt.

---

# Voraussetzungen

- Python 3.10 oder neuer
- Lokal installiertes Ollama
- Git
- Internetverbindung für den ersten Download der Abhängigkeiten sowie der Modelle (Llama 3.2, DistilBERT,
  sentence-transformers)

Tkinter ist bei den meisten Python-Installationen bereits enthalten.

---

# Installation

## Repository klonen

```bash
git clone https://github.com/FootballOnlooker/Local-AI-Assistant.git
cd Local-AI-Assistant
```

## Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

## Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

## Llama herunterladen

```bash
ollama pull llama3.2:3b
```

Anschließend Ollama starten:

```bash
ollama serve
```

## Modell trainieren

```bash
python teil2/train.py
```

Dadurch wird das Verzeichnis

```text
teil2/model/final_model/
```

erstellt.

## Anwendung starten

```bash
python -m teil1.main
```

---

# Weitere Befehle

Datensatz überprüfen:

```bash
python teil2/check_dataset.py
```

Klassifikation testen:

```bash
python teil2/predict.py
```

Retrieval testen:

```bash
python -m teil1.retrieval
```

---

# Beispielanfragen

```text
Wie lange gilt die Garantie?

Wie bekomme ich eine Rechnung?

Wie lange dauert die Lieferung?

Wie kann ich den Kundenservice kontaktieren?

Wie melde ich eine Reklamation?
```

Außerhalb der Wissensdatenbank:

```text
Wie wird das Wetter morgen?
```

In diesem Fall informiert die Anwendung den Benutzer darüber, dass sich diese Information nicht in den
bereitgestellten Dokumenten befindet, statt eine Antwort zu erfinden.

---

# Screenshots

## Lokaler KI-Assistent

![Local AI Assistant](images/teil1_chat.png)

## DistilBERT-Training

![Training](images/training_model_teil2.png)

---

# Einschränkungen

- Der Klassifikator wurde mit einem relativ kleinen, aber vollständig eindeutigen (keine Vorlagen-Duplikate)
  Datensatz trainiert.
- Der Confidence Score entspricht der höchsten Softmax-Wahrscheinlichkeit und garantiert keine korrekte
  Vorhersage; bei manchen Formulierungen liegt die Konfidenz nahe an der Entscheidungsgrenze zwischen zwei
  Kategorien.
- Die semantische Suche (sentence-transformers) ist auf Bedeutung statt auf exakten Wortlaut ausgelegt; sehr
  seltene Fachbegriffe oder ungewöhnlich formulierte Fragen können ein niedrigeres Ähnlichkeitsmaß erhalten als
  erwartet. Exakte Referenz-Codes umgehen dies über einen separaten Regex-Abgleich (siehe Abschnitt „Retrieval").
- Das lokale 3B-LLM (`llama3.2:3b`) befolgt rein prompt-basierte Anweisungen nicht mit 100%iger Zuverlässigkeit,
  besonders wenn mehrere Regeln gleichzeitig gelten (Gesprächsgedächtnis, Satzlimit, Sprachanpassung und „keine
  Fakten erfinden" zugleich). Dort, wo dies in Tests am relevantesten war (nicht dokumentierte Lieferstädte), wurde
  eine zusätzliche, code-basierte deterministische Prüfung als Verstärkung ergänzt — das verringert, beseitigt aber
  nicht vollständig das Risiko gelegentlich erfundener Antworten.
- Ollama sowie das trainierte Klassifikationsmodell müssen lokal verfügbar sein, bevor die Anwendung gestartet
  werden kann.
- Die GUI führt Modellaufrufe derzeit synchron aus und kann während der Antwortgenerierung durch das lokale
  Sprachmodell kurzzeitig nicht reagieren.

---

# Mögliche Erweiterungen

- Den Klassifikationsdatensatz durch weitere, ebenso eindeutige Vorlagen erweitern.
- Modellaufrufe in einem Hintergrundthread ausführen, damit die GUI während der Verarbeitung reaktionsfähig bleibt.
- Für maximale Zuverlässigkeit bei nicht dokumentierten Fällen könnte die deterministische Städte-Prüfung zu einem
  vollständigen Bypass ausgebaut werden (Sprachmodell komplett überspringen, feste Antwort zurückgeben) — bewusst
  nicht umgesetzt, um die Antworten natürlich statt schablonenhaft zu halten.
- Automatisierte Tests für Retrieval und Klassifikation hinzufügen.

---

# Projektziel

Dieses Projekt wurde als technische Demonstration entwickelt und zeigt:

- die Integration eines lokal ausgeführten Large Language Models (LLM),
- das Fine-Tuning eines Transformer-Modells für die Textklassifikation,
- Retrieval-Augmented Generation (RAG) mit einer lokalen, in Absätze gegliederten Wissensdatenbank,
- sorgfältige Datensatz-Validierung (Vorlagen-Duplikate, Data Leakage) statt unkritischer Verwendung von
  Kennzahlen,
- die Entwicklung einer Desktop-Anwendung mit Tkinter,
- sowie die Kombination mehrerer KI-Komponenten in einer Python-Anwendung.
