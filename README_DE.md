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
3. ein hybrides lokales **Retrieval-System**: exakter Abgleich von Referenz-Codes plus mehrsprachige Satz-Embeddings
   über einzelne Dokument-Abschnitte.

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
- Hybrides Retrieval: exakter Referenz-Code-Abgleich + mehrsprachige Satz-Embeddings (`sentence-transformers`) auf
  Absatzebene mit Cosine Similarity
- Gedächtnis über mehrere Nachrichten hinweg sowie dauerhafte Benutzeranweisungen (z. B. „Antworte ab jetzt immer in
  maximal drei Sätzen")
- Sprachanpassung: antwortet in der Sprache der aktuellen Nachricht (primär Deutsch, mit zuverlässiger Unterstützung für
  Englisch)
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

Entwicklung eines Desktop-Assistenten, der Kundenfragen mithilfe einer lokalen Wissensdatenbank beantwortet, sich über
mehrere Nachrichten hinweg an den bisherigen Gesprächsverlauf erinnert und dauerhafte Anweisungen befolgt.

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
vor, werden alle passenden Dokumente klar getrennt in den Kontext übernommen, statt willkürlich nur das erste zu wählen.

**2. Semantische Suche auf Absatzebene (Satz-Embeddings).** Wird kein Referenz-Code in der Frage gefunden, greift die
Anwendung auf `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) zurück. Modell und Embeddings werden
einmalig beim Start berechnet, nicht bei jeder Nachricht neu.

Dokumente werden dabei **nicht** als Ganzes eingebettet, sondern zunächst in Absätze aufgeteilt (`chunk_documents()`),
jeder Absatz einzeln kodiert, und die Frage wird gegen diese Absätze statt gegen ganze Dokumente verglichen. Das ist
eine gezielte Korrektur eines real gemessenen Problems: Das Embedding-Modell hat ein
`max_seq_length` von 128 Tokens. Mehrere Wissensdokumente (100–190 Wörter, nach deutscher Subword-Tokenisierung grob
130–300 Tokens) überschritten dieses Limit — Inhalte gegen Ende eines Dokuments wurden dadurch beim Kodieren
stillschweigend abgeschnitten und waren für die Suche unsichtbar, unabhängig davon, wie wichtig sie waren. Die
Aufteilung in Absätze hält jede Sucheinheit sicher unter dem Limit (geprüft: der längste Absatz in der aktuellen
Wissensdatenbank liegt bei geschätzt ~91 Tokens), ohne das zu verkleinern, was das Sprachmodell am Ende sieht — bei
einem Treffer wird weiterhin der **vollständige** Dokumenttext als Kontext zurückgegeben.

Die Suchanfrage kombiniert die aktuelle Frage mit den letzten zwei Nutzer-Nachrichten, damit Folgefragen wie „Wie lange
würde der Transport *dorthin* dauern?" trotzdem dem richtigen Thema zugeordnet werden können, ohne den Ort erneut zu
nennen.

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
dreier echter Testfragen, nicht geschätzt: eine Frage nah am Dokumentwortlaut erzielte ~0,69; eine inhaltlich relevante,
aber anders formulierte Frage („Ich habe meine Rechnung nicht erhalten" gegenüber der formaleren Dokumentformulierung)
erzielte ~0,41; eine thematisch irrelevante Frage erzielte ~0,26. Der Schwellenwert liegt mit Sicherheitsabstand über
dem irrelevanten und unter beiden relevanten Fällen.

Wird kein ausreichend ähnlicher Absatz gefunden und auch kein Referenz-Code erkannt, weist die Anwendung das
Sprachmodell ausdrücklich darauf hin, dass keine dokumentierte Information vorliegt, statt es eine Antwort erfinden zu
lassen.

Als zusätzliche Absicherung führt die Anwendung außerdem eine kleine deterministische Prüfung durch: Sie liest die in
`lieferung.txt` tatsächlich dokumentierten Städte aus und ergänzt, falls nach einer Lieferung in eine nicht gelistete
Stadt gefragt wird, eine kurze, eigenständige Anweisung an das Sprachmodell, für diese Stadt keine Lieferzeit zu
erfinden. Diese Prüfung ist bewusst eine Ergänzung zur Prompt-Anweisung, kein vollständiger Ersatz des Sprachmodells —
rein prompt-basierte Anweisungen wurden vom 3B-Modell in Tests nicht durchgängig zuverlässig befolgt.

## Test: Mehrschrittiges Gespräch

**Ziel:** prüfen, ob der Assistent (a) sich Angaben aus früheren Nachrichten merkt und eine spätere Folgefrage, die
darauf verweist, korrekt auflöst, ohne dass die Angabe wiederholt werden muss, und (b) eine dauerhafte
Verhaltensanweisung („Antworte ab jetzt in maximal drei Sätzen") über mehrere folgende Nachrichten zuverlässig einhält.

**Test-Gespräch** (5 aufeinanderfolgende Nachrichten, `llama3.2:3b`, keine Nachricht wiederholt):

1. `Hallo, meine Sendungsnummer lautet EXP-88213, und der Zielort ist Budapest.`
2. `Antworten Sie ab jetzt bitte immer in maximal drei Sätzen.`
3. `Wie hoch ist die Standardhaftung bei Transportschäden?`
4. `Und wie sieht es mit der erweiterten Transportversicherung aus?`
5. `Wie lange würde der Transport dorthin ungefähr dauern?` — nennt absichtlich nicht erneut „Budapest", um zu prüfen,
   ob Nachricht 1 vier Nachrichten später noch erinnert wird.

**Endergebnis:**

| # | Antwort des Assistenten                                                                                                                                                                    | Prüfung                                                                             |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 1 | „Sie hat Ihre Sendungsnummer (EXP-88213) und den Zielort (Budapest) genannt."                                                                                                              | Nur Bestätigung, keine Rückfrage, keine Behauptung fehlender Daten                  |
| 2 | „Entschuldigung, ich werde mich daran erinnern. Die Lieferung von Sendungsnummer EXP-88213 nach Budapest wird bearbeitet. Wir werden sich um die korrekte Abholung und Lieferung kümmern." | 3 Sätze, bleibt beim Thema                                                          |
| 3 | „Die Standardhaftung für Transportschäden beträgt den gesetzlichen Höchstbetrag pro Kilogramm Rohgewicht der beschädigten oder verlorenen Ware."                                           | Wortgetreu aus `garantie.txt`, keine erfundenen Zahlen                              |
| 4 | „Die erweiterte Transportversicherung deckt den deklarierten Warenwert bis zur vereinbarten Versicherungssumme ab, Zuschlag abhängig vom Warenwert."                                       | Wortgetreu aus `garantie.txt`                                                       |
| 5 | „Die Lieferung von Sendungsnummer EXP-88213 nach Budapest wird in der Regel 1-2 Werktage dauern."                                                                                          | Löst „dorthin" korrekt zu Budapest auf — vier Nachrichten später, ohne Wiederholung |

**Beim Testen gefundene und behobene Fehler** (alle in `teil1/chat.py` / `teil1/retrieval.py`):

1. **Zielort ging über mehrere Nachrichten hinweg verloren.** Die Retrieval-Anfrage kombinierte nur die letzten 2
   Benutzer-Nachrichten; bei Nachricht 5 war Nachricht 1 (wo „Budapest" genannt wurde) bereits aus diesem Fenster
   herausgefallen, wodurch die semantische Suche `lieferung.txt` nicht mehr fand und das Modell fälschlich behauptete,
   für Budapest liege keine Lieferzeit vor — obwohl sie dokumentiert ist. Behoben durch eine separate, deterministische
   Stadt-Prüfung (`find_mentioned_documented_city()`), die den **gesamten** bisherigen Gesprächsverlauf durchsucht
   (nicht nur ein kleines Fenster) und bei einer dokumentierten Stadt die exakte Lieferzeit direkt ins Prompt gibt,
   sobald die aktuelle Frage tatsächlich nach der Dauer fragt.
2. **Falsches Dokument bei thematisch anderen Folgefragen.** Das Kombinieren mehrerer Nachrichten in einer
   Embedding-Anfrage ließ ein früheres Thema eine spätere, inhaltlich andere Frage überlagern: „Sendungsnummer
   EXP-88213" (Nachricht 1) zog bei Nachricht 3–4 fälschlich `rechnung.txt` heran (dort steht ein ähnlich formatierter
   Beispielcode), obwohl diese Fragen eindeutig Haftung/Versicherung betrafen (`garantie.txt`). Behoben, indem zuerst
   nur mit der aktuellen Frage gesucht wird und erst bei keinem Treffer auf die kombinierte Mehr-Nachrichten-Anfrage
   zurückgefallen wird.
3. **Erfundene Zahlen.** Vor den obigen Fixes erfand das Modell Werte, die in keinem Dokument stehen (z. B. „8 %",
   „100 % des Transportpreises") — ein direkter Verstoß gegen die Regel, keine Fakten zu erfinden. Das verschwand
   größtenteils, sobald das Retrieval das richtige Dokument fand; zusätzlich wurde die Prompt-Formulierung verschärft,
   um auch unbelegte erklärende Zusätze (nicht nur Zahlen) ausdrücklich zu verbieten.
4. **Regelverstoß bei reinen Mitteilungen.** Wenn der Nutzer nur Informationen mitteilte (keine Frage), sprach der
   Assistent trotzdem über „fehlende Dokumentation", statt kurz zu bestätigen. Behoben, indem der Dokument-Kontext nur
   noch ins Prompt aufgenommen wird, wenn die aktuelle Nachricht tatsächlich ein Fragezeichen enthält — andernfalls wird
   stattdessen eine explizite Anweisung ergänzt, die reine Bestätigung ohne Rückfrage verlangt.
5. **Verfälschter Fragen-Log.** `retrieve_document()` kann ein Dokument „finden", ohne dass es in der Antwort verwendet
   wird (siehe Punkt 4); der Fragen-Log in `database.py` trägt `document_id` / `was_found_in_kb` jetzt nur noch ein,
   wenn ein Dokument tatsächlich Teil der Antwort war.

**Bekannte verbleibende Einschränkung:** gelegentliche kleine Grammatikfehler (z. B. „Sie hat" statt „Sie haben")
und selten eine unaufgeforderte, leichte Ausschmückung über den Quelltext hinaus. Beides liegt am 3B-Parameter-Modell
selbst, nicht an der Programmlogik — bestätigt durch mehrfaches Wiederholen desselben Tests: alle **deterministischen**
Prüfungen (Stadt, Dokument, Zahlen) blieben dabei durchgehend korrekt, nur einzelne Formulierungen variierten leicht.

## Test: Referenz-Code-Retrieval

**Ziel:** prüfen, ob der Assistent Informationen zu tatsächlich dokumentierten Referenz-Codes korrekt findet und
zitiert — und ebenso wichtig: für einen plausibel aussehenden, aber komplett erfundenen Code **keinen** Status erfindet.

**Testfragen** (jeweils eine frische, unabhängige Frage, kein gemeinsamer Gesprächsverlauf):

| # | Frage                                                  | Code           | Dokumentiert?                       |
|---|--------------------------------------------------------|----------------|-------------------------------------|
| 1 | `Zu welcher Sendung gehört die Rechnung RE-2024-1187?` | `RE-2024-1187` | Ja (`rechnung.txt`)                 |
| 2 | `Was deckt die Versicherungspolice VP-30987 ab?`       | `VP-30987`     | Ja (`garantie.txt`)                 |
| 3 | `Wie ist der Status der Sendung GRP-44190?`            | `GRP-44190`    | Ja (`lieferung.txt`)                |
| 4 | `Was ist mit der Sendung EXP-99999 passiert?`          | `EXP-99999`    | **Nein** — für diesen Test erfunden |

**Ergebnis:**

| # | Retrieval                                                                                                   | Antwort des Assistenten                                                                                                                                                                         |
|---|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `rechnung.txt`, Ähnlichkeit 1,0 (exakter Code-Treffer)                                                      | „Die Rechnung RE-2024-1187 wurde zur Sendung EXP-70531 gestellt." — wortgetreu aus dem Dokument                                                                                                 |
| 2 | `garantie.txt`, Ähnlichkeit 1,0 (exakter Code-Treffer)                                                      | „Die Versicherungspolice VP-30987 deckt den deklarierten Warenwert bis zur vereinbarten Versicherungssumme ab, Zuschlag abhängig vom Warenwert." — korrekt, entspricht der Dokumentbeschreibung |
| 3 | `lieferung.txt`, Ähnlichkeit 1,0 (exakter Code-Treffer)                                                     | „Die Sendung GRP-44190 wurde am Montag abgeholt und ist laut Sendungsverfolgung am Dienstagvormittag beim Empfänger eingetroffen." — wortgetreu aus dem Dokument                                |
| 4 | `lieferung.txt`, Ähnlichkeit 0,54 (nur semantischer Fallback — **kein** exakter Code-Treffer, wie erwartet) | „Es gibt keine dokumentierte Information über die Sendung EXP-99999." — verweigert korrekt das Erfinden eines Status                                                                            |

Alle vier Antworten bestanden im ersten Durchlauf, ohne weitere Codeänderungen: Der exakte Regex-Abgleich (siehe
„Retrieval" oben) findet echte Codes zuverlässig mit Ähnlichkeit `1,0`, und beim erfundenen Code fällt die Suche korrekt
auf den rein semantischen Pfad zurück, der unterhalb der Schwelle bleibt, ab der eine konkrete Antwort gegeben würde —
das Modell wird also ausdrücklich informiert, dass keine dokumentierte Information vorliegt, statt selbst zu raten.
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

Der Datensatz besteht aus handgeschriebenen Satzvorlagen, wobei jede Vorlage **genau einmal** verwendet wird (zum
Zeitpunkt der Erstellung programmatisch geprüft, nicht nur behauptet):

- 180 deutsche Textbeispiele
- 4 ausgewogene Kategorien
- jeweils 45 Beispiele pro Kategorie
- keine fehlenden Werte
- keine Duplikate
- **100 % eindeutige Vorlagen** — bestätigt durch Normalisierung von Städten/Referenz-Codes zu Platzhaltern und Prüfung
  auf Kollisionen; es gibt keine

Eine frühere Version dieses Datensatzes (800 Beispiele) verwendete denselben Satzbau mit lediglich ausgetauschter Stadt
oder Referenznummer, teils bis zu 10-mal. Das führte zu einer irreführend perfekten Testgenauigkeit von 100 %, weil ein
zufälliger Split nahezu identische Sätze auf beide Seiten von Training und Test verteilen konnte.
`train.py` verwendet `StratifiedGroupKFold`, um dies strukturell zu verhindern — unabhängig davon, wie der Datensatz
aufgebaut ist. Zusätzlich wurde der Datensatz selbst so neu geschrieben, dass er an der Quelle bereits eindeutig ist —
beide Absicherungen zusammen statt nur einer.

---

# Trainingsablauf

1. CSV-Datensatz laden und überprüfen
2. Kategorien in numerische IDs umwandeln
3. Vorlagen-bewussten, stratifizierten Train/Test-Split erstellen (`StratifiedGroupKFold`), damit Vorlagen-Duplikate
   nicht zwischen Training und Test „durchsickern"
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
— keine Vorlagen-Duplikate zwischen Training und Test. Die Genauigkeit liegt nahe am allerersten, kleineren Testlauf mit
200 Beispielen (87,5 %), was ein gutes unabhängiges Indiz dafür ist, dass das Modell tatsächlich verallgemeinert und
nicht Satzvorlagen auswendig lernt.

| Parameter                      | Wert                           |
|--------------------------------|--------------------------------|
| Basismodell                    | `distilbert-base-german-cased` |
| Datensatzgröße                 | 180                            |
| Kategorien                     | 4                              |
| Trainingsdaten                 | 144                            |
| Testdaten                      | 36                             |
| Maximal konfigurierte Epochen  | 6                              |
| Tatsächlich trainierte Epochen | 6,0                            |
| Batch Size                     | 8                              |
| Learning Rate                  | `2e-5`                         |
| Trainingszeit                  | 2,10 min                       |
| Evaluation Loss                | 0,6385                         |
| Test Accuracy                  | **86,11 %**                    |

Die Ergebnisse können je nach System geringfügig variieren. Siehe `teil2/confusion_matrix.png` für eine visuelle
Aufschlüsselung, welche Kategorien miteinander verwechselt werden.

**Klassifikationsbericht:**

| Kategorie   | Precision | Recall | F1-Score | Support |
|-------------|----------:|-------:|---------:|--------:|
| Anfrage     |      0,88 |   0,78 |     0,82 |       9 |
| Reklamation |      0,89 |   0,89 |     0,89 |       9 |
| Rechnung    |      0,88 |   0,78 |     0,82 |       9 |
| Sonstiges   |      0,82 |   1,00 |     0,90 |       9 |

`Sonstiges` hat einen perfekten Recall (jede tatsächliche `Sonstiges`-Nachricht wurde gefunden), aber eine niedrigere
Precision — laut Konfusionsmatrix stammt der größte Anteil davon aus 2 `Anfrage`-Beispielen, die fälschlich als
`Sonstiges` eingeordnet wurden. Das erklärt gleichzeitig den niedrigeren Recall von `Anfrage`
(7 von 9). Die übrigen Fehler sind eher verstreut: `Rechnung` hatte je ein Beispiel, das als `Anfrage` bzw.
`Reklamation` fehlklassifiziert wurde, `Reklamation` ein Beispiel, das als `Rechnung` eingeordnet wurde. Keine Kategorie
wird durchgängig mit genau einer anderen verwechselt — `Sonstiges` ist als semantisch diffuseste Kategorie der
Hauptanziehungspunkt für Grenzfälle.

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

In diesem Fall informiert die Anwendung den Benutzer darüber, dass sich diese Information nicht in den bereitgestellten
Dokumenten befindet, statt eine Antwort zu erfinden.

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
- Der Confidence Score entspricht der höchsten Softmax-Wahrscheinlichkeit und garantiert keine korrekte Vorhersage; bei
  manchen Formulierungen liegt die Konfidenz nahe an der Entscheidungsgrenze zwischen zwei Kategorien.
- Die semantische Suche (sentence-transformers) ist auf Bedeutung statt auf exakten Wortlaut ausgelegt; sehr seltene
  Fachbegriffe oder ungewöhnlich formulierte Fragen können ein niedrigeres Ähnlichkeitsmaß erhalten als erwartet. Exakte
  Referenz-Codes umgehen dies über einen separaten Regex-Abgleich (siehe Abschnitt „Retrieval").
- Das lokale 3B-LLM (`llama3.2:3b`) befolgt rein prompt-basierte Anweisungen nicht mit 100%iger Zuverlässigkeit,
  besonders wenn mehrere Regeln gleichzeitig gelten (Gesprächsgedächtnis, Satzlimit, Sprachanpassung und „keine Fakten
  erfinden" zugleich). Dort, wo dies in Tests am relevantesten war (nicht dokumentierte Lieferstädte), wurde eine
  zusätzliche, code-basierte deterministische Prüfung als Verstärkung ergänzt — das verringert, beseitigt aber nicht
  vollständig das Risiko gelegentlich erfundener Antworten.
- Ollama sowie das trainierte Klassifikationsmodell müssen lokal verfügbar sein, bevor die Anwendung gestartet werden
  kann.
- Die GUI führt Modellaufrufe derzeit synchron aus und kann während der Antwortgenerierung durch das lokale Sprachmodell
  kurzzeitig nicht reagieren.

---

# Mögliche Erweiterungen

- Den Klassifikationsdatensatz durch weitere, ebenso eindeutige Vorlagen erweitern.
- Modellaufrufe in einem Hintergrundthread ausführen, damit die GUI während der Verarbeitung reaktionsfähig bleibt.
- Für maximale Zuverlässigkeit bei nicht dokumentierten Fällen könnte die deterministische Städte-Prüfung zu einem
  vollständigen Bypass ausgebaut werden (Sprachmodell komplett überspringen, feste Antwort zurückgeben) — bewusst nicht
  umgesetzt, um die Antworten natürlich statt schablonenhaft zu halten.
- Automatisierte Tests für Retrieval und Klassifikation hinzufügen.

---

# Projektziel

Dieses Projekt wurde als technische Demonstration entwickelt und zeigt:

- die Integration eines lokal ausgeführten Large Language Models (LLM),
- das Fine-Tuning eines Transformer-Modells für die Textklassifikation,
- Retrieval-Augmented Generation (RAG) mit einer lokalen, in Absätze gegliederten Wissensdatenbank,
- sorgfältige Datensatz-Validierung (Vorlagen-Duplikate, Data Leakage) statt unkritischer Verwendung von Kennzahlen,
- die Entwicklung einer Desktop-Anwendung mit Tkinter,
- sowie die Kombination mehrerer KI-Komponenten in einer Python-Anwendung.
