import re

import ollama

from teil1.database import (
    init_db,
    load_documents_from_db,
    log_question,
    migrate_txt_to_db,
)
from teil1.retrieval import (
    chunk_documents,
    encode_chunks,
    extract_documented_cities,
    find_undocumented_delivery_city,
    retrieve_document,
    extract_delivery_times,
    find_mentioned_documented_city
)

# Datenbank vorbereiten: Tabellen anlegen und .txt-Dateien einmalig
# hineinmigrieren.
init_db()
migrate_txt_to_db()

# Wissensdatenbank wird jetzt aus SQLite geladen statt aus Textdateien.
DOCUMENTS = load_documents_from_db()
# Für den Fragen-Log brauchen wir später die Datenbank-ID zu einem
# gefundenen Dokumentnamen. Einmal als Nachschlage-Tabelle aufgebaut,
# statt bei jeder Nachricht neu zu suchen.
DOCUMENT_ID_BY_NAME = {document["name"]: document["id"] for document in DOCUMENTS}
# Dokumente in Chunks aufteilen (siehe chunk_documents() in retrieval.py),
# einmal beim Start berechnet statt bei jeder Nachricht neu.
CHUNKS = chunk_documents(DOCUMENTS)
CHUNK_VECTORS = encode_chunks(CHUNKS)
# Für die deterministischen Stadt-Prüfungen in ask() unten.
DOCUMENTED_CITIES = extract_documented_cities(DOCUMENTS)
# Stadt -> dokumentierte Lieferzeit (z. B. "Budapest" -> "in der Regel
# 1-2 Werktage"). Gegenstück zu DOCUMENTED_CITIES: wird genutzt, um bei
# einer Folgefrage zu einer WOHL dokumentierten Stadt die exakte Angabe
# deterministisch ins Prompt zu geben, statt sich darauf zu verlassen,
# dass das LLM sie selbst im vollständigen Dokumenttext wiederfindet
# (siehe find_mentioned_documented_city() in retrieval.py für den
# Hintergrund – in Tests nicht zuverlässig genug).
DELIVERY_TIMES = extract_delivery_times(DOCUMENTS)

# Einfache Heuristik für "ist die Nachricht Englisch?": englische
# Funktionswörter vorhanden, deutsche nicht. Nur für die Extra-Anweisung
# unten genutzt, ändert nichts am Verhalten für Deutsch.
ENGLISH_MARKERS = re.compile(
    r"\b(the|is|are|you|could|would|does|what|how|please|cover|covers|"
    r"delivery|shipment|invoice)\b",
    re.IGNORECASE,
)
GERMAN_MARKERS = re.compile(
    r"\b(der|die|das|und|ist|sie|wie|für|nach|mit|ein|eine|sind|hoch|lange)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """
Du bist ein zuverlässiger Kundenservice-Assistent.

Regeln für das Gespräch:

1. Merke dir alle konkreten Informationen, die der Benutzer im Verlauf nennt,
   insbesondere Sendungsnummern, Zielorte, Mengen, Namen und Termine. Ein vom
   Benutzer genannter "Zielort" ist das Lieferziel, nicht der Abholort.
   Sofern kein anderer Abholort genannt wurde, ist Wien der Standardabholort.

2. Informationen, die der Benutzer selbst genannt hat, gelten als bekannte
   Gesprächsinformationen. Behaupte später nicht, dass diese Informationen
   fehlen.

3. Bei Folgefragen wie „dorthin", „diese Sendung", „sie" oder „davon"
   musst du den bisherigen Gesprächsverlauf berücksichtigen.

4. Befolge dauerhafte Anweisungen des Benutzers auch in den folgenden
   Nachrichten, zum Beispiel:
   „Antworte ab jetzt immer in maximal drei Sätzen."
   Diese Anweisung gilt so lange, bis der Benutzer sie ändert oder aufhebt,
   auch wenn sie in einer früheren Nachricht gegeben wurde.

5. Erfinde keine zusätzlichen Fakten. Das gilt insbesondere für Referenz-,
   Sendungs- und Rechnungsnummern sowie technische Angaben, die nicht im
   Gespräch oder im bereitgestellten Dokument genannt wurden.

6. Stelle keine unnötigen Rückfragen, wenn die benötigte Information bereits
   im Gespräch genannt wurde. Wenn der Benutzer lediglich Informationen
   mitteilt (z. B. eine Sendungsnummer oder einen Zielort nennt), ohne eine
   konkrete Frage zu stellen, bestätige dies kurz und freundlich. Sprich in
   diesem Fall NICHT über fehlende Dokumentation in der Wissensdatenbank –
   das ist nur relevant, wenn tatsächlich nach einer konkreten Angabe
   (Preis, Dauer, Nummer, Zeit) gefragt wird.

7. Antworte in der Sprache der aktuellen Nutzernachricht. Der Regelbetrieb
   ist Deutsch; wird eine Frage auf Englisch gestellt, antworte auf
   Englisch. Wechsle nicht ohne Grund die Sprache, wenn der Benutzer sie
   nicht gewechselt hat.

Beispiel für Regel 7:
Nutzer (Englisch): "Can you tell me how long delivery to Budapest takes?"
Assistent (Englisch): "According to our documentation, delivery from
Vienna to Budapest usually takes 1-2 business days."

Nutzer (Deutsch, direkt danach): "Und wie sieht es mit Graz aus?"
Assistent (Deutsch): "Von Wien nach Graz dauert die Lieferung in der Regel
1 Werktag."

8. Sprich den Benutzer immer mit der höflichen Anrede "Sie" an, nie mit
   "du". Dies ist ein professioneller Kundenservice-Kontext.

9. Formatiere Antworten als einfachen Fließtext ohne Markdown (keine
   Sternchen für Fettschrift, keine nummerierten oder mit Bindestrich
   eingeleiteten Listen, keine Überschriften). Die Oberfläche zeigt reinen
   Text an; Markdown-Zeichen erscheinen sonst wörtlich als Sternchen.
""".strip()


def build_context_message(retrieval_result):
    """Turn a retrieval result into a system message for the LLM."""

    if retrieval_result["name"]:
        if retrieval_result.get("exact_match"):
            return (
                "Relevantes Dokument aus der Wissensdatenbank "
                f"(Quelle: {retrieval_result['name']}):\n\n"
                f"{retrieval_result['text']}\n\n"
                "Der vom Benutzer genannte Code/die Referenznummer wurde "
                "nachweislich WÖRTLICH in diesem Dokument gefunden (exakter "
                "Treffer, keine Vermutung). Suche im obigen Text gezielt den "
                "Satz, der diesen Code enthält, und beantworte die Frage "
                "direkt und ohne Einschränkung mit der dort genannten "
                "Information. Sei dir sicher: der Code steht wirklich im "
                "Text, auch wenn im Dokument noch weitere Zahlen oder "
                "Formatbeispiele vorkommen – verwechsle die Frage nicht mit "
                "einem allgemeinen Formatbeispiel (z. B. 'RE-JJJJ-XXXX')."
            )

        return (
            "Relevantes Dokument aus der Wissensdatenbank "
            f"(Quelle: {retrieval_result['name']}):\n\n"
            f"{retrieval_result['text']}\n\n"
            "Beantworte die aktuelle Frage des Benutzers auf Grundlage dieses "
            "Dokuments und der bisher im Gespräch genannten Informationen. "
            "Übernimm keine Zahlen oder Fakten, die weder im Dokument noch im "
            "bisherigen Gespräch stehen. Ergänze die Antwort auch NICHT um "
            "eigene erklärende Zusätze, Begründungen oder Einschränkungen "
            "(z. B. ob etwas Standard/optional ist, wovon etwas abhängt), "
            "wenn das nicht wortwörtlich im Dokument steht – im Zweifel "
            "lieber kürzer antworten als etwas Plausibles hinzuzudichten.\n\n"
            "Wichtig: Dieses Dokument ist nur thematisch relevant, es muss die "
            "konkret gefragte Angabe (z. B. eine bestimmte Stadt, Sendungs- oder "
            "Rechnungsnummer) nicht enthalten. Prüfe explizit, ob der genaue "
            "gefragte Wert im Dokumenttext vorkommt. Ist er nicht enthalten, "
            "schätze ihn NICHT anhand ähnlicher Beispiele und nenne KEINE eigene "
            "Zahl. Sag stattdessen klar, dass für diesen konkreten Fall keine "
            "dokumentierte Angabe vorliegt, und biete optional die nächstgelegene "
            "dokumentierte Strecke oder Referenz zum Vergleich an (ausdrücklich "
            "als Vergleich gekennzeichnet, nicht als Antwort auf die gestellte "
            "Frage)."
        )

    return (
        "Es wurde kein passendes Dokument in der Wissensdatenbank gefunden. "
        "Weise den Benutzer freundlich darauf hin, dass diese Information "
        "nicht in den bereitgestellten Unterlagen enthalten ist, anstatt eine "
        "Antwort zu erfinden. Nutze weiterhin die bisher im Gespräch genannten "
        "Informationen (z. B. Sendungsnummern oder Orte), falls sie für die "
        "Antwort relevant sind."
    )


class ChatSession:
    def __init__(self, max_history_messages=20, retrieval_context_turns=6):
        self.history = []
        self.max_history_messages = max_history_messages
        # Fenster für die EMBEDDING-Dokumentensuche, bewusst klein: ein
        # größeres Fenster verwässerte in Tests die Trefferqualität für
        # andere Themen im Fenster. Die Stadt-Prüfungen weiter unten in
        # ask() nutzen bewusst NICHT dieses Fenster, sondern die volle
        # Konversation (siehe all_user_text dort).
        self.retrieval_context_turns = retrieval_context_turns

    def _build_retrieval_query(self, question):
        previous_user_messages = [
            message["content"]
            for message in self.history
            if message["role"] == "user"
        ][-self.retrieval_context_turns:]

        combined = previous_user_messages + [question]
        # Exakte Wiederholungen entfernen, Reihenfolge bleibt erhalten.
        deduplicated = list(dict.fromkeys(combined))

        return " ".join(deduplicated)

    def ask(self, question):
        question = question.strip()

        if not question:
            return "Bitte geben Sie eine Frage ein."

        # Erst mit der aktuellen Frage allein suchen; die History wird nur
        # als Fallback verwendet, falls das nichts findet (z. B. "dorthin"
        # ohne Kontext). Verhindert, dass eine ältere Nachricht eine klar
        # anders gelagerte aktuelle Frage verfälscht.
        retrieval_result = retrieve_document(question, DOCUMENTS, CHUNKS, CHUNK_VECTORS)
        if not retrieval_result["name"]:
            retrieval_query = self._build_retrieval_query(question)
            retrieval_result = retrieve_document(retrieval_query, DOCUMENTS, CHUNKS, CHUNK_VECTORS)
        # Diagnose-Ausgabe: zeigt, welches Dokument (falls überhaupt eines)
        # gefunden wurde. Rein informativ, beeinflusst die Antwort nicht.
        print(
            f"[retrieval] Frage: {question!r} -> Dokument: "
            f"{retrieval_result['name']!r}, "
            f"Ähnlichkeit: {retrieval_result.get('similarity')}"
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            ]
        # Regel 7 (Sprache) zusätzlich verstärken, wenn die Frage Englisch
        # aussieht — reine Prompt-Anweisung reichte dafür nicht immer aus,
        # besonders wenn der Dokument-Kontext unten auf Deutsch ist.
        if ENGLISH_MARKERS.search(question) and not GERMAN_MARKERS.search(question):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "WICHTIG: Die aktuelle Nachricht des Benutzers ist auf "
                        "Englisch verfasst. Antworte AUSSCHLIESSLICH auf "
                        "Englisch, auch wenn die Dokumente/Informationen unten "
                        "auf Deutsch sind. Übersetze die relevanten Fakten "
                        "korrekt ins Englische, ohne den Inhalt zu verändern "
                        "oder etwas hinzuzufügen."
                    ),
                }
            )
        # Dokument-Kontext nur bei einer echten Frage ("?") anhängen —
        # sonst kollidiert build_context_message() mit Regel 6, wenn der
        # Nutzer nur Informationen mitteilt statt zu fragen.
        is_question = "?" in question
        if is_question:
            messages.append(
                {
                    "role": "system",
                    "content": build_context_message(retrieval_result),
                }
            )
        else:
            # Regel 6 zusätzlich verstärken: kurze Bestätigung ohne
            # Rückfrage, statt über fehlende Dokumentation zu sprechen.
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Der Nutzer hat gerade nur Informationen mitgeteilt "
                        "(keine Frage gestellt, kein '?'). Bestätige dies in "
                        "GENAU EINEM kurzen Aussagesatz, freundlich und ohne "
                        "Rückfrage. Formuliere KEINE Frage zurück an den "
                        "Nutzer und sprich nicht über fehlende Dokumentation "
                        "oder Lieferzeiten, solange nicht danach gefragt wurde."
                    ),
                }
            )
        # Für die beiden Stadt-Prüfungen unten bewusst die volle
        # Konversation nutzen, nicht das begrenzte
        # retrieval_context_turns-Fenster — reiner Textabgleich verwässert
        # nicht wie eine Embedding-Suche.
        previous_user_messages_full = [
            message["content"]
            for message in self.history
            if message["role"] == "user"
        ]
        all_user_text = " ".join(previous_user_messages_full + [question])
        # Deterministische Prüfung: undokumentierte Stadt -> explizite
        # Anweisung, keine Lieferzeit dafür zu erfinden.
        undocumented_city = find_undocumented_delivery_city(all_user_text, DOCUMENTED_CITIES)
        if undocumented_city:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"WICHTIG: Für '{undocumented_city}' liegt in der "
                        "Wissensdatenbank KEINE dokumentierte Lieferzeit vor. "
                        "Nenne für diese Stadt keine Anzahl an Werktagen oder "
                        "Tagen, auch nicht als Schätzung. Sag ausdrücklich, "
                        "dass dazu keine dokumentierte Angabe existiert."
                    ),
                }
            )

        # Gegenstück: dokumentierte Stadt -> exakte Lieferzeit explizit
        # mitgeben, statt sich auf das LLM zu verlassen, das die Angabe im
        # vollständigen Dokumenttext nicht immer zuverlässig fand. Nur bei
        # einer Frage nach Dauer ausgelöst, sonst vermischt es sich mit
        # Regel 6 bei reinen Mitteilungen.
        asks_about_duration = bool(
            re.search(r"lange|dauer|wann|werktag|tage\b", question, re.IGNORECASE)
        )
        documented_city, documented_delivery_time = (
            find_mentioned_documented_city(all_user_text, DELIVERY_TIMES)
            if asks_about_duration
            else (None, None)
        )
        if documented_city:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"WICHTIG, hat Vorrang vor allen anderen Angaben oben: "
                        f"Für '{documented_city}' liegt in der Wissensdatenbank "
                        f"eine dokumentierte Lieferzeit vor: Wien nach "
                        f"{documented_city}: {documented_delivery_time} Antworte "
                        f"auf die aktuelle Frage nach der Lieferzeit/Dauer nach "
                        f"'{documented_city}' NUR mit genau dieser Angabe. "
                        f"Ignoriere jeden gegenteiligen Hinweis oben, dass dazu "
                        f"keine oder keine spezifische Information vorliege – "
                        f"das gilt für '{documented_city}' NICHT."
                    ),
                }
            )

        messages.extend(self.history)

        messages.append(
            {
                "role": "user",
                'content': question,
            })

        try:
            response = ollama.chat(
                model="llama3.2:3b",
                messages=messages,
                options={
                    "temperature": 0,
                },
            )

            answer = response["message"]["content"].strip()
            # llama3.2:3b erzeugt gelegentlich ein isoliertes "Sie." am
            # Anfang der Antwort; wird hier entfernt.
            answer = re.sub(r"^Sie\.\s+", "", answer)

            # Frage protokollieren (siehe log_question() in database.py).
            # document_id nur setzen, wenn das Dokument auch wirklich in
            # der Antwort verwendet wurde (is_question) — sonst würde ein
            # rein lexikalischer Zufallstreffer von retrieve_document()
            # fälschlich im Log erscheinen.
            matched_name = retrieval_result["name"] if is_question else None
            first_matched_name = matched_name.split(", ")[0] if matched_name else None
            document_id = DOCUMENT_ID_BY_NAME.get(first_matched_name)
            log_question(
                question_text=question,
                document_id=document_id,
                answer_text=answer,
                was_found=bool(matched_name),
            )

            self.history.append(
                {
                    "role": "user",
                    'content': question,
                }
            )
            self.history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            self.limit_history()
            return answer

        except Exception as error:
            print(f'Ollama error:{error}')
            return (
                "Die KI-Antwort konnte nicht erstellt werden. "
                "Bitte prüfen Sie, ob Ollama gestartet ist."
            )

    def limit_history(self):
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages:]

    def reset_history(self):
        self.history.clear()


chat_session = ChatSession()


def ask_ai(question):
    return chat_session.ask(question)


def reset():
    chat_session.reset_history()
