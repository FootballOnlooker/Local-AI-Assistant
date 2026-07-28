# Lokaler KI-Assistent

🇬🇧 **English version:** [README.md](README.md)

Ein lokaler Desktop-KI-Assistent zur Verarbeitung deutscher Kundenanfragen.

Die Anwendung kombiniert drei KI-Komponenten:

1. ein lokal ausgeführtes Sprachmodell über **Ollama**,
2. einen fein abgestimmten **German DistilBERT-Klassifikator**,
3. ein lokales **Retrieval-System** auf Basis von TF-IDF und Cosine Similarity.

Die Tkinter-Oberfläche zeigt die Benutzereingabe, die vorhergesagte Kategorie mit Confidence Score sowie eine KI-generierte Antwort auf Grundlage der lokalen Wissensdatenbank an.

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
- Dokumentensuche mittels TF-IDF und Cosine Similarity
- Kontextbasierte Antworten mit Llama 3.2
- Separate Skripte für Training, Vorhersage, Datensatzprüfung und Retrieval-Tests

---

# Architektur

```text
Benutzernachricht
        │
        ├──► DistilBERT-Klassifikator
        │         └──► Kategorie + Confidence
        │
        └──► TF-IDF-Retrieval
                    │
                    └──► relevantestes Dokument
                                │
                                └──► Ollama / Llama 3.2
                                            │
                                            └──► KI-Antwort

Alle Ergebnisse werden in der Tkinter-Oberfläche angezeigt.
```

---

# Projektstruktur

```text
Local-AI-Assistant/
│
├── teil1/
│   ├── knowledge/
│   ├── chat.py
│   ├── gui.py
│   ├── main.py
│   └── retrieval.py
│
├── teil2/
│   ├── data/
│   ├── model/
│   ├── classifier.py
│   ├── train.py
│   ├── predict.py
│   └── check_dataset.py
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

Entwicklung eines Desktop-Assistenten, der Kundenfragen mithilfe einer lokalen Wissensdatenbank beantwortet.

## Verwendete Technologien

- Python
- Tkinter
- Ollama
- Llama 3.2:3b
- Scikit-learn

---

# Retrieval

Vor jeder Anfrage wird zunächst das passendste Dokument aus der Wissensdatenbank ermittelt.

```text
Dokumente
      │
      └──► TF-IDF-Dokumentvektoren

Benutzerfrage
      │
      └──► TF-IDF-Fragevektor
                │
                └──► Cosine Similarity
                            │
                            └──► relevantestes Dokument
                                        │
                                        └──► Kontext für das Sprachmodell
```

Wird kein ausreichend ähnliches Dokument gefunden, gibt die Anwendung eine feste Meldung aus, anstatt das Sprachmodell eine Antwort erfinden zu lassen.

Die Implementierung ist bewusst einfach gehalten und dient Demonstrationszwecken. TF-IDF vergleicht hauptsächlich gemeinsame Wörter und besitzt kein tiefes semantisches Verständnis wie embeddingbasierte Verfahren.

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

Der Datensatz wurde manuell erstellt und enthält:

- 200 deutsche Textbeispiele
- 4 ausgewogene Kategorien
- jeweils 50 Beispiele pro Kategorie
- keine Duplikate
- keine fehlenden Werte

---

# Trainingsablauf

1. CSV-Datensatz laden und überprüfen
2. Kategorien in numerische IDs umwandeln
3. Stratified Train/Test Split erstellen
4. Hugging Face Dataset erzeugen
5. Texte tokenisieren
6. DistilBERT feinabstimmen
7. Modell evaluieren
8. Modell und Tokenizer lokal speichern

---

# Trainingsergebnisse

| Parameter | Wert |
|-----------|------|
| Basismodell | distilbert-base-german-cased |
| Datensatz | 200 |
| Trainingsdaten | 160 |
| Testdaten | 40 |
| Kategorien | 4 |
| Epochen | 3 |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Trainingszeit | ca. 49 Sekunden |
| Evaluation Loss | 0.8794 |
| Test Accuracy | **87,5 %** |

Die Ergebnisse können je nach System geringfügig variieren.

---

# Beispiel

## Eingabe

```text
Wie bekomme ich eine Rechnung?
```

## Ausgabe

```text
Kategorie:
Anfrage (67,46 %)

KI:

Die Rechnung wird nach Abschluss der Bestellung automatisch per E-Mail verschickt. Falls sie nicht im Posteingang zu finden ist, sollte auch der Spam-Ordner geprüft werden.
```

Der Klassifikator und das Retrieval arbeiten unabhängig voneinander.

Der Klassifikator bestimmt lediglich die Nachrichtenkategorie, während das Retrieval das passende Dokument für den Kontext des Sprachmodells auswählt.

---

# Voraussetzungen

- Python 3.10 oder neuer
- Lokal installiertes Ollama
- Git
- Internetverbindung für den ersten Download der Abhängigkeiten sowie des Modells

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

Datensatz überprüfen

```bash
python teil2/check_dataset.py
```

Klassifikation testen

```bash
python teil2/predict.py
```

Retrieval testen

```bash
python teil1/retrieval.py
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

In diesem Fall informiert die Anwendung den Benutzer darüber, dass sich diese Information nicht in den bereitgestellten Dokumenten befindet.

---

# Screenshots

## Lokaler KI-Assistent

![Local AI Assistant](images/teil1_chat.png)

## DistilBERT-Training

![Training](images/training_model_teil2.png)

---

# Einschränkungen

- Der Klassifikator wurde mit einem relativ kleinen, manuell erstellten Datensatz trainiert.
- Der Confidence Score entspricht der höchsten Softmax-Wahrscheinlichkeit und garantiert keine korrekte Vorhersage.
- Das TF-IDF-Retrieval basiert hauptsächlich auf gemeinsamen Wörtern und kann bei ungewöhnlichen Formulierungen ein weniger passendes Dokument auswählen.
- Ollama sowie das trainierte Klassifikationsmodell müssen lokal verfügbar sein, bevor die Anwendung gestartet werden kann.
- Die GUI führt Modellaufrufe derzeit synchron aus und kann während der Antwortgenerierung durch das lokale Sprachmodell kurzzeitig nicht reagieren.

---

# Mögliche Erweiterungen

- Den Klassifikationsdatensatz erweitern und durch vielfältigere Beispiele verbessern.
- Modellaufrufe in einem Hintergrundthread ausführen, damit die GUI während der Verarbeitung reaktionsfähig bleibt.
- TF-IDF-Dokumentvektoren zwischenspeichern, anstatt sie bei jeder Anfrage neu zu berechnen.
- Automatisierte Tests für Retrieval und Klassifikation hinzufügen.

---

# Projektziel

Dieses Projekt wurde als technische Demonstration entwickelt und zeigt:

- die Integration eines lokal ausgeführten Large Language Models (LLM),
- das Fine-Tuning eines Transformer-Modells für die Textklassifikation,
- Retrieval-Augmented Generation (RAG) mit einer lokalen Wissensdatenbank,
- die Entwicklung einer Desktop-Anwendung mit Tkinter,
- sowie die Kombination mehrerer KI-Komponenten in einer Python-Anwendung.
