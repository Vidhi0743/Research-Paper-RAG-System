
import hashlib
import re
from pathlib import Path


CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Common research-paper section headings.
SECTION_PATTERN = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVX]+)\s*[.:\-]?\s+)?"
    r"(Abstract|Introduction|Background|Related Work|"
    r"Literature Review|Methodology|Methods|Method|"
    r"Proposed Approach|Proposed Method|Model|Architecture|"
    r"Experimental Setup|Experiments|Results|"
    r"Results and Discussion|Discussion|Evaluation|"
    r"Conclusion|Conclusions|Future Work|"
    r"Limitations|References|Acknowledgements|Appendix)"
    r"\s*$",
    re.IGNORECASE
)


def identify_section(line):
    """Return a recognized section heading, if present."""
    candidate = " ".join(line.strip().split())

    if len(candidate) > 80:
        return None

    match = SECTION_PATTERN.fullmatch(candidate)

    if match:
        return candidate

    return None


def split_into_units(text, max_words):
    """
    Prefer paragraph and sentence boundaries.
    Fall back to word splitting for oversized sentences.
    """
    units = []

    paragraphs = re.split(r"\n\s*\n", text)

    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split())

        if not paragraph:
            continue

        if len(paragraph.split()) <= max_words:
            units.append(paragraph)
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            paragraph
        )

        for sentence in sentences:
            words = sentence.split()

            if not words:
                continue

            if len(words) <= max_words:
                units.append(sentence)
            else:
                for start in range(0, len(words), max_words):
                    units.append(
                        " ".join(words[start:start + max_words])
                    )

    return units


def make_text_chunks(text, chunk_size=300, overlap=50):
    """
    Recursive-style boundary-aware splitting with word overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    units = split_into_units(text, chunk_size)

    chunks = []
    current = []
    current_words = 0

    for unit in units:
        unit_words = unit.split()
        unit_length = len(unit_words)

        if current and current_words + unit_length > chunk_size:

            completed = " ".join(current)
            chunks.append(completed)

            # Carry words from the preceding chunk.
            previous_words = completed.split()

            tail = (
                previous_words[-overlap:]
                if overlap
                else []
            )

            # Avoid exceeding the size limit when adding
            # the next paragraph or sentence.
            available = chunk_size - unit_length
            tail = tail[-min(len(tail), available):] if available else []

            current = [" ".join(tail)] if tail else []
            current_words = len(tail)

        current.append(unit)
        current_words += unit_length

    if current:
        final_chunk = " ".join(current).strip()

        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def chunk_research_document(
    document,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Accepts the cleaned document structure produced
    by src/text_cleaner.py.

    Returns chunks with page and section metadata.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk configuration")

    filename = document["filename"]
    source_path = document["source_path"]

    chunks = []
    current_section = "Unclassified"
    chunk_index = 0

    for page in document["pages"]:

        page_number = page["page_number"]
        text = page["text"]

        # Separate lines that look like section headings.
        lines = text.splitlines()
        section_parts = []
        buffer = []

        for line in lines:
            heading = identify_section(line)

            if heading:
                if buffer:
                    section_parts.append(
                        (current_section, "\n".join(buffer))
                    )
                    buffer = []

                current_section = heading

            else:
                buffer.append(line)

        if buffer:
            section_parts.append(
                (current_section, "\n".join(buffer))
            )

        for section_name, section_text in section_parts:

            text_chunks = make_text_chunks(
                section_text,
                chunk_size=chunk_size,
                overlap=overlap
            )

            for chunk_text in text_chunks:

                chunk_index += 1

                raw_id = (
                    f"{source_path}|{page_number}|"
                    f"{chunk_index}|{chunk_text}"
                )

                chunk_id = hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()[:16]

                chunks.append({
                    "chunk_id": chunk_id,
                    "source": filename,
                    "source_path": source_path,
                    "file_type": Path(filename).suffix.lower(),
                    "page_number": page_number,
                    "section": section_name,
                    "chunk_index": chunk_index,
                    "chunking_method": "section_aware_recursive",
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })

    return chunks
