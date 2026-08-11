# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A local desktop AI assistant for German customer-service messages, combining three independent components:

1. **`teil1/`** — a Tkinter chat GUI backed by a local LLM (Ollama, `llama3.2:3b`) with a hybrid retrieval system over a local knowledge base.
2. **`teil2/`** — a fine-tuned German DistilBERT (`distilbert-base-german-cased`) text classifier for four categories: `Anfrage`, `Reklamation`, `Rechnung`, `Sonstiges`.
3. The two are wired together only at the GUI layer (`teil1/gui.py` calls both `teil1.chat.ask_ai` and `teil2.classifier.classify_text`) — the classifier's predicted category and the retrieval-selected document are independent and not used to influence each other.

`README.md` (English) / `README_DE.md` (German) contain the full write-up, including the calibration story behind the retrieval threshold and the dataset-leakage fix — read it before changing retrieval or training logic, since the current constants encode specific empirical findings, not defaults.

## Commands

```bash
# Run the desktop app (module form is required — teil1/gui.py uses package-relative imports)
python -m teil1.main

# Train the classifier (writes teil2/model/final_model/, not tracked in git)
python teil2/train.py

# Standalone classification from the terminal
python teil2/predict.py

# Dataset quality check (missing values, duplicates, label distribution)
python teil2/check_dataset.py

# Test retrieval alone from the terminal (prompts for a question)
python teil1/retrieval.py
```

There is no lint or test suite configured in this repo. Before running the GUI, Ollama must be running locally with `ollama pull llama3.2:3b` already done, and `teil2/train.py` must have been run at least once to produce `teil2/model/final_model/`.

## Architecture

### teil1 — chat assistant

- **`database.py`** is the source of truth for the knowledge base: it owns a SQLite file (`teil1/knowledge.db`, gitignored-equivalent but currently present untracked) with a `documents` table and a `questions` log table. `migrate_txt_to_db()` does a one-time, idempotent (`INSERT OR IGNORE` + `UNIQUE(name)`) import of `teil1/knowledge/*.txt` into `documents` on every startup; editing a `.txt` file after the first run does **not** update the DB row — the DB is authoritative once migrated. Every user question is logged via `log_question()`, deduplicated by a normalized (lowercased, punctuation-stripped) text match against `normalized_text`.
- **`retrieval.py`** implements hybrid retrieval, tried in this order:
  1. **Exact reference-code match** — a regex (`[A-ZÄÖÜ]{2,5}-[0-9][0-9A-Z-]+`) pulls codes like `EXP-70531` out of the question and does a literal substring match across all documents. If a code legitimately appears in more than one document, all matches are combined into the context rather than picking one.
  2. **Chunk-level semantic search** — `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, loaded once at module import time, not per-call). Documents are split into paragraph-sized chunks (`chunk_documents()`) before embedding because the model's `max_seq_length` (128 tokens) silently truncates several full documents otherwise; the chunk match still returns the **full parent document** as context, not just the matching chunk. `MIN_SIMILARITY = 0.35` is an empirically calibrated threshold (see comments in `retrieval.py`/README) — don't tune it without similarly-documented justification.
  3. A separate deterministic check, `find_undocumented_delivery_city()`, scans `lieferung.txt` for `"Wien nach <Stadt>"` routes and — independent of the LLM prompt — injects an extra system message when the user asks about an undocumented city, because the 3B model doesn't reliably follow the "don't invent facts" instruction on its own for this case.
- **`chat.py`** builds the `ChatSession`: it computes the module-level `DOCUMENTS`/`CHUNKS`/`CHUNK_VECTORS`/`DOCUMENTED_CITIES` once at import time (re-embedding per message would reload the model on every chat turn), holds conversation history capped at `max_history_messages`, and builds the retrieval query from the current question plus the last `retrieval_context_turns` user messages (so follow-ups like "and to Graz?" still resolve). `SYSTEM_PROMPT` encodes the behavioral contract (German/English language-matching, formal "Sie" address, no markdown output, no fact invention, persistent user instructions like "answer in 3 sentences") — most GUI-observable behavior changes trace back to this prompt or to `build_context_message()`, which crafts a different system message depending on whether retrieval found an exact-code match, a semantic match, or nothing.
- **`gui.py`** is a single-file Tkinter UI; it calls the LLM synchronously (no threading), so the window is unresponsive while `ollama.chat()` is running.

### teil2 — classifier

- **`train.py`** loads `teil2/data/dataset.csv` (`;`-separated), and — critically — computes a `template_group` per example (`get_template_group()`: lowercase, strip punctuation, replace reference codes and known city names with placeholders) used as the `groups` argument to `StratifiedGroupKFold`. This exists because the dataset is built from sentence templates with swapped cities/codes; a naive random split leaks near-duplicate templates across train/test and inflates accuracy. If you add new cities or code formats to the dataset, update `CITY_PATTERN`/`CODE_PATTERN` in `train.py` accordingly, or grouping silently stops working for those examples.
- Training fine-tunes `distilbert-base-german-cased` for up to 6 epochs with early stopping (patience 2, `metric_for_best_model="accuracy"`), then saves a per-class report and confusion matrix (`teil2/confusion_matrix.png`) plus the model to `teil2/model/final_model/` (gitignored — must be regenerated locally via `train.py`).
- **`classifier.py`** loads the tokenizer/model from `teil2/model/final_model/` once at import time and exposes `classify_text(text) -> (label, confidence)`, confidence being the max softmax probability (not a calibrated confidence measure).

## Working conventions in this repo

- Non-trivial constants (`MIN_SIMILARITY`, chunking thresholds, regex patterns for codes/cities) were arrived at empirically and are explained in inline comments — preserve or update those comments if you change the values, don't just silently retune them.
- Comments and docstrings in `teil1/` and `teil2/` are written in German, matching the target audience (German customer-service domain); keep new comments in the same file consistent with that.
- Knowledge documents live in `teil1/knowledge/*.txt` but are only picked up by the running app through the one-time SQLite migration in `database.py` — when editing knowledge content, be aware whether `knowledge.db` already contains a stale copy that needs to be regenerated (delete the row or the whole DB file to force re-migration).
