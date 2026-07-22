from classifier import classify_text


def main() -> None:
    text = input("Enter text: ").strip()

    if not text:
        print("Bitte geben Sie einen Text ein.")
        return

    result, confidence = classify_text(text)

    print(f"Kategorie: {result}")
    print(f"Confidence: {confidence:.2f} %")


if __name__ == "__main__":
    main()