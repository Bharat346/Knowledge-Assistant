from app.ingestion.chunking import TextSection, chunk_sections


def test_recursive_chunking_respects_size_and_overlap():
    source = "\n\n".join(f"Paragraph {index}. " + "word " * 18 for index in range(8))
    chunks = chunk_sections(
        [TextSection(source, page=2)], document_id="doc", filename="notes.md",
        file_type="md", source_path="uploads/notes.md", chunk_size=120, chunk_overlap=20,
    )
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)


def test_code_metadata_and_duplicate_chunk_removal():
    repeated = "def answer():\n    return 42"
    chunks = chunk_sections(
        [TextSection(repeated + "\n\n" + repeated)], document_id="doc", filename="main.py",
        file_type="py", source_path="src/main.py", chunk_size=100, chunk_overlap=10,
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["language"] == "python"
    assert chunks[0].metadata["path"] == "src/main.py"