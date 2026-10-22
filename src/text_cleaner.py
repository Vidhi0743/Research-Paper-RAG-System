
import re
import unicodedata


def clean_text(text):
    """
    Clean extracted document text while preserving
    meaningful paragraph boundaries.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if not text.strip():
        return ""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Convert Windows/Mac line endings to Unix format.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking spaces with normal spaces.
    text = text.replace("\u00a0", " ")

    # Remove unwanted control characters, but preserve
    # newlines and tabs for further processing.
    text = "".join(
        char for char in text
        if char in ("\n", "\t")
        or unicodedata.category(char) != "Cc"
    )

    # Replace tabs and repeated horizontal spaces.
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove spaces around newline characters.
    text = re.sub(r" *\n *", "\n", text)

    # Reduce three or more consecutive newlines to two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace.
    return text.strip()


def clean_document(document):
    """
    Clean every extracted page of a document
    while preserving its metadata.
    """

    cleaned_document = document.copy()
    cleaned_pages = []

    for page in document["pages"]:

        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        cleaned_pages.append({
            **page,
            "text": cleaned_text
        })

    cleaned_document["pages"] = cleaned_pages

    return cleaned_document


def clean_documents(documents):
    """
    Clean multiple extracted documents.
    """

    cleaned_documents = []

    for document in documents:

        cleaned_document = clean_document(document)

        if cleaned_document["pages"]:
            cleaned_documents.append(cleaned_document)

            print(
                f"[CLEANED] {document['filename']}"
            )

        else:
            print(
                f"[SKIPPED] No usable text: "
                f"{document['filename']}"
            )

    return cleaned_documents
