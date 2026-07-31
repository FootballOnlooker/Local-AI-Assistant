import ollama

from teil1.retrieval import (
    encode_documents,
    extract_documented_cities,
    find_undocumented_delivery_city,
    load_documents,
    retrieve_document,
)

# Wissensdatenbank wird einmalig beim Start geladen.
DOCUMENTS = load_documents()
# Embeddings der Dokumente ebenfalls nur einmal berechnen, statt bei jeder
# einzelnen Chat-Nachricht neu (siehe Diskussion: ansonsten "Loading
# weights" bei jedem Aufruf, weil sonst auch das Modell neu geladen würde).
DOCUMENT_VECTORS = encode_documents(DOCUMENTS)
# Städte, für die lieferung.txt tatsächlich eine Lieferzeit dokumentiert.
# Wird für eine deterministische, code-basierte Prüfung genutzt (siehe
# ask() unten) – zusätzlich zur Anweisung im Prompt, nicht nur anstelle
# davon, weil reine Prompt-Anweisungen bei diesem konkreten Fall in
# Tests nicht zuverlässig genug eingehalten wurden.
DOCUMENTED_CITIES = extract_documented_cities(DOCUMENTS)

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
            "bisherigen Gespräch stehen.\n\n"
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
    def __init__(self, max_history_messages=20, retrieval_context_turns=2):
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

        return " ".join(previous_user_messages + [question])

    def ask(self, question):
        question = question.strip()

        if not question:
            return "Bitte geben Sie eine Frage ein."

        retrieval_query = self._build_retrieval_query(question)
        retrieval_result = retrieve_document(retrieval_query, DOCUMENTS, DOCUMENT_VECTORS)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "system",
                "content": build_context_message(retrieval_result),
            },
        ]

        # Zusätzliche, code-garantierte Prüfung (unabhängig davon, ob das
        # LLM die allgemeine Anweisung im Prompt befolgt): Wird eine Stadt
        # genannt, die nachweislich NICHT in lieferung.txt steht, bekommt
        # das Modell eine kurze, isolierte Extra-Anweisung dazu.
        undocumented_city = find_undocumented_delivery_city(question, DOCUMENTED_CITIES)
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
