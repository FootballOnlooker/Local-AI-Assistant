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

# Datenbank beim Start vorbereiten: Tabellen anlegen (falls neu) und die
# .txt-Dateien einmalig hineinmigrieren (bereits vorhandene Namen werden
# übersprungen, siehe migrate_txt_to_db()).
init_db()
migrate_txt_to_db()

# Wissensdatenbank wird jetzt aus SQLite geladen statt aus Textdateien.
DOCUMENTS = load_documents_from_db()
# Für den Fragen-Log brauchen wir später die Datenbank-ID zu einem
# gefundenen Dokumentnamen. Einmal als Nachschlage-Tabelle aufgebaut,
# statt bei jeder Nachricht neu zu suchen.
DOCUMENT_ID_BY_NAME = {document["name"]: document["id"] for document in DOCUMENTS}
# Für die Suche werden Dokumente in Absätze (Chunks) aufgeteilt, damit die
# 128-Token-Grenze des Embedding-Modells keine Inhalte "unsichtbar" macht
# (siehe ausführlicher Kommentar bei chunk_documents() in retrieval.py).
# Wird nur einmal beim Start berechnet, nicht bei jeder Chat-Nachricht neu
# (siehe Diskussion: ansonsten "Loading weights" bei jedem Aufruf, weil
# sonst auch das Modell neu geladen würde).
CHUNKS = chunk_documents(DOCUMENTS)
CHUNK_VECTORS = encode_chunks(CHUNKS)
# Städte, für die lieferung.txt tatsächlich eine Lieferzeit dokumentiert.
# Wird für eine deterministische, code-basierte Prüfung genutzt (siehe
# ask() unten) – zusätzlich zur Anweisung im Prompt, nicht nur anstelle
# davon, weil reine Prompt-Anweisungen bei diesem konkreten Fall in
# Tests nicht zuverlässig genug eingehalten wurden.
DOCUMENTED_CITIES = extract_documented_cities(DOCUMENTS)
# Stadt -> dokumentierte Lieferzeit (z. B. "Budapest" -> "in der Regel
# 1-2 Werktage"). Gegenstück zu DOCUMENTED_CITIES: wird genutzt, um bei
# einer Folgefrage zu einer WOHL dokumentierten Stadt die exakte Angabe
# deterministisch ins Prompt zu geben, statt sich darauf zu verlassen,
# dass das LLM sie selbst im vollständigen Dokumenttext wiederfindet
# (siehe find_mentioned_documented_city() in retrieval.py für den
# Hintergrund – in Tests nicht zuverlässig genug).
DELIVERY_TIMES = extract_delivery_times(DOCUMENTS)

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
        # Wie viele vorherige Benutzer-Nachrichten zusätzlich zur aktuellen
        # Frage für die Dokumentensuche verwendet werden. Das hilft bei
        # Folgefragen wie "Wie lange dauert der Transport dorthin?", bei
        # denen das eigentliche Thema (z. B. "Lieferung") erst durch den
        # Gesprächsverlauf klar wird.
        self.retrieval_context_turns = retrieval_context_turns

    def _build_retrieval_query(self, question):
        previous_user_messages = [
            message["content"]
            for message in self.history
            if message["role"] == "user"
        ][-self.retrieval_context_turns:]

        combined = previous_user_messages + [question]
        # Exakte Wiederholungen entfernen (z. B. wenn derselbe Wortlaut
        # zweimal hintereinander gefragt wird) – Reihenfolge bleibt
        # erhalten. dict.fromkeys() nutzt aus, dass Dictionary-Keys seit
        # Python 3.7 ihre Einfügereihenfolge behalten und Duplikate
        # automatisch zusammenfallen.
        deduplicated = list(dict.fromkeys(combined))

        return " ".join(deduplicated)

    def ask(self, question):
        question = question.strip()

        if not question:
            return "Bitte geben Sie eine Frage ein."

        # Erst mit der AKTUELLEN Frage allein suchen (ohne Gesprächsverlauf).
        # Hintergrund: Ein Test zeigte, dass das Kombinieren mit älteren
        # Nachrichten (siehe _build_retrieval_query()) eine klar erkennbare
        # aktuelle Frage verfälschen kann – z. B. zog "Sendungsnummer
        # EXP-88213" aus einer früheren Nachricht die Suche für eine
        # spätere, eindeutig anders gelagerte Frage zur Standardhaftung
        # fälschlich zu rechnung.txt (das selbst ein ähnliches Beispiel
        # "EXP-70531" enthält), obwohl garantie.txt eindeutig gepasst hätte.
        # Der Gesprächsverlauf wird nur noch als Fallback verwendet, wenn
        # die aktuelle Frage allein NICHTS Passendes findet (z. B. bei
        # "Wie lange dauert der Transport dorthin?", wo "dorthin" ohne
        # Kontext keinen Anhaltspunkt bietet).
        retrieval_result = retrieve_document(question, DOCUMENTS, CHUNKS, CHUNK_VECTORS)
        if not retrieval_result["name"]:
            retrieval_query = self._build_retrieval_query(question)
            retrieval_result = retrieve_document(retrieval_query, DOCUMENTS, CHUNKS, CHUNK_VECTORS)
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
        # Den Dokument-Kontext nur anhängen, wenn tatsächlich eine Frage
        # gestellt wurde (einfache Heuristik: enthält "?"). Hintergrund:
        # build_context_message() weist das Modell explizit an, zu prüfen,
        # ob ein konkreter Wert im Dokument steht und andernfalls klar zu
        # sagen, dass keine dokumentierte Angabe vorliegt – das kollidiert
        # mit Regel 6 im SYSTEM_PROMPT, wenn der Nutzer gar nichts gefragt,
        # sondern nur Informationen mitgeteilt hat (z. B. "Mein Zielort ist
        # Budapest."). In einem Test führte genau das dazu, dass auf eine
        # reine Mitteilung mit "keine Informationen zur Lieferzeit" reagiert
        # wurde, obwohl gar keine Frage dazu gestellt war.
        is_question = "?" in question
        if is_question:
            messages.append(
                {
                    "role": "system",
                    "content": build_context_message(retrieval_result),
                }
            )
        else:
            # Explizite Verstärkung von Regel 6 für genau diesen Fall,
            # analog zu den Stadt-Prüfungen weiter unten: reines Vertrauen
            # auf die allgemeine Regel im SYSTEM_PROMPT reichte in einem
            # Test nicht – statt einer kurzen Bestätigung stellte das
            # Modell eine rhetorische Rückfrage ("...möchten Sie wissen,
            # wie lange...?"), was Regel 6 ("ohne Rückfrage bestätigen")
            # ebenfalls nicht ganz trifft.
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
        # Für die beiden Stadt-Prüfungen unten bewusst NICHT retrieval_query
        # (begrenztes Fenster, siehe retrieval_context_turns) verwenden,
        # sondern den kompletten bisherigen Gesprächsverlauf: Ein reiner
        # Text-Abgleich auf Stadtnamen verwässert nicht dadurch, dass man
        # ihm mehr Text gibt (anders als die Embedding-Suche), daher kann
        # er ruhig auch eine Stadt finden, die mehrere Nachrichten zuvor
        # genannt wurde (z. B. "dorthin" als Folgefrage).
        previous_user_messages_full = [
            message["content"]
            for message in self.history
            if message["role"] == "user"
        ]
        all_user_text = " ".join(previous_user_messages_full + [question])
        # Zusätzliche, code-garantierte Prüfung (unabhängig davon, ob das
        # LLM die allgemeine Anweisung im Prompt befolgt): Wird eine Stadt
        # genannt, die nachweislich NICHT in lieferung.txt steht, bekommt
        # das Modell eine kurze, isolierte Extra-Anweisung dazu.
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

        # Gegenstück: Wurde eine Stadt genannt, für die es SEHR WOHL eine
        # dokumentierte Lieferzeit gibt (evtl. schon vor mehreren
        # Nachrichten), wird die exakte Angabe hier explizit mitgegeben.
        # Hintergrund: In Tests hat llama3.2:3b diese Angabe nicht
        # zuverlässig selbst im vollständigen Dokumenttext gefunden und
        # stattdessen fälschlich behauptet, es gäbe keine dokumentierte
        # Lieferzeit für die Stadt.
        #
        # Bewusst nur ausgelöst, wenn die AKTUELLE Frage überhaupt nach
        # Dauer/Lieferzeit klingt (nicht bei jeder bloßen Erwähnung der
        # Stadt, siehe all_user_text oben) – sonst feuert der Block z. B.
        # schon, wenn der Nutzer nur seinen Zielort nennt, ohne etwas zu
        # fragen (Regel 6 im SYSTEM_PROMPT), und vermischt sich unnötig
        # mit der Retrieval-Einschätzung aus build_context_message().
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
            # Kleine, risikoarme Nachbearbeitung: llama3.2:3b erzeugt
            # gelegentlich ein isoliertes "Sie." als eigenes Fragment am
            # Anfang der Antwort (Sprachmodell-Unschärfe, kein Logikfehler).
            # Wird hier weggeschnitten, ohne Retrieval/Prompt anzufassen.
            answer = re.sub(r"^Sie\.\s+", "", answer)

            # Zweite Zusatzanforderung: neue Fragen in der Datenbank
            # protokollieren. Nutzt den Dokumentnamen aus retrieval_result,
            # um (falls vorhanden) die passende document_id nachzuschlagen.
            # Bei mehreren Treffern (exakter Code in mehreren Dokumenten,
            # siehe find_document_by_reference_code) wird der erste
            # verwendet – für den Log reicht ein Verweis, keine vollständige
            # Auflistung aller Treffer.
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
