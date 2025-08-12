from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import loader as mod

# ----------------------------
# validate_pdf_file tests
# ----------------------------

def test_validate_pdf_file_nonexistent():
    with pytest.raises(FileNotFoundError):
        mod.validate_pdf_file("does/not/exist.pdf")


def test_validate_pdf_file_wrong_extension(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("just text")
    with pytest.raises(ValueError) as excinfo:
        mod.validate_pdf_file(str(p))
    assert "File is not a PDF" in str(excinfo.value)


def test_validate_pdf_file_invalid_header(tmp_path: Path):
    p = tmp_path / "bad.pdf"

    # Wrong header; must be exactly b'%PDF' in the first 4 bytes
    p.write_bytes(b"%PDX-1.7\nrest")
    with pytest.raises(ValueError) as excinfo:
        mod.validate_pdf_file(str(p))
    assert "does not appear to be a valid PDF" in str(excinfo.value)


def test_validate_pdf_file_valid_pdf(tmp_path: Path, capsys):
    p = tmp_path / "valid.pdf"

    # Correct header; function only checks first four bytes
    p.write_bytes(b"%PDF-1.4\nmore bytes")
    mod.validate_pdf_file(str(p))
    out = capsys.readouterr().out
    assert "PDF file validated" in out


# ----------------------------
# load_document tests
# ----------------------------

def test_load_document_pdf_success(tmp_path: Path):
    p = tmp_path / "file.pdf"
    p.write_bytes(b"%PDF-1.4\n")

    fake_loader_instance = Mock()
    fake_doc1 = Mock(page_content="page one")
    fake_doc2 = Mock(page_content="page two")
    fake_loader_instance.load.return_value = [fake_doc1, fake_doc2]

    with patch.object(mod, "PyPDFLoader", return_value=fake_loader_instance) as patched:
        docs = mod.load_document(str(p))

    patched.assert_called_once_with(str(p))
    assert len(docs) == 2
    assert docs[0].page_content == "page one"
    assert docs[1].page_content == "page two"


def test_load_document_txt_success(tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text("hello world")

    fake_loader_instance = Mock()
    fake_doc = Mock(page_content="hello world")
    fake_loader_instance.load.return_value = [fake_doc]

    with patch.object(mod, "TextLoader", return_value=fake_loader_instance) as patched:
        docs = mod.load_document(str(p))

    patched.assert_called_once_with(str(p))
    assert len(docs) == 1
    assert docs[0].page_content == "hello world"


def test_load_document_unsupported_extension(tmp_path: Path):
    p = tmp_path / "notes.md"
    p.write_text("# heading")
    with pytest.raises(ValueError) as excinfo:
        mod.load_document(str(p))
    assert "Unsupported file type" in str(excinfo.value)


def test_load_document_empty_load_raises_runtimeerror(tmp_path: Path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"%PDF-1.4\n")

    fake_loader_instance = Mock()
    fake_loader_instance.load.return_value = []

    with patch.object(mod, "PyPDFLoader", return_value=fake_loader_instance):
        with pytest.raises(RuntimeError) as excinfo:
            mod.load_document(str(p))
    assert "No content loaded" in str(excinfo.value)


def test_load_document_loader_raises_is_reraised(tmp_path: Path, capsys):
    p = tmp_path / "boom.pdf"
    p.write_bytes(b"%PDF-1.4\n")

    def _constructor(_path):
        raise RuntimeError("loader explode")

    with patch.object(mod, "PyPDFLoader", side_effect=_constructor):
        with pytest.raises(RuntimeError) as excinfo:
            mod.load_document(str(p))

    # The function prints a helpful message, then re-raises
    out = capsys.readouterr().out
    assert "Document loading failed for" in out
    assert "loader explode" in str(excinfo.value)


# ----------------------------
# preprocess_document tests
# ----------------------------

class SimpleDoc:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


def test_preprocess_document_cleans_whitespace_and_keeps_long_docs():
    # Note: the function collapses ALL whitespace (including newlines) into single spaces,
    # then tries to split on '\n' (which won't do much after collapsing). We test the observed behavior.
    noisy_text = "Line   1 \n\n with   extra   spaces \n and   newlines.\n"

    # Make it long enough (>50 chars) so it isn't filtered out
    noisy_text = (noisy_text + " ")*5

    doc = SimpleDoc(noisy_text)
    out = mod.preprocess_document([doc])

    assert len(out) == 1
    # In-place mutation expected
    assert out[0] is doc
    assert "  " not in doc.page_content  # no double spaces
    assert "\n" not in doc.page_content  # newlines collapsed
    assert len(doc.page_content.strip()) > 50


def test_preprocess_document_filters_short_docs():
    short_text = "Too short to keep."
    doc = SimpleDoc(short_text)
    out = mod.preprocess_document([doc])
    assert out == []


def test_preprocess_document_skips_very_short_lines_if_any_left():
    # Even though newlines are collapsed, we can still provide a long-enough content
    # and ensure it's kept; short (<4 char) lines would be excluded if present.
    text = "abc\nde\nfghi jkl mno pqr stu vwx yz " * 3  # length > 50 after collapsing
    doc = SimpleDoc(text)
    out = mod.preprocess_document([doc])
    assert len(out) == 1
    # Collapsed and trimmed
    assert "  " not in out[0].page_content
    assert "\n" not in out[0].page_content


# ----------------------------
# Integration-ish smoke tests
# ----------------------------

def test_end_to_end_pdf_path_then_preprocess(tmp_path: Path):
    # Create a valid-looking PDF file
    pdf = tmp_path / "ok.pdf"
    pdf.write_bytes(b"%PDF-1.5\nrest")
    # validate should pass
    mod.validate_pdf_file(str(pdf))

    # Mock loading two pages so preprocess has something to clean
    page1 = SimpleDoc("First    page \n with   noise.\n")
    page2 = SimpleDoc("Second   page \n with more   noise.\n" + "x" * 60)

    fake_loader_instance = Mock()
    fake_loader_instance.load.return_value = [page1, page2]

    with patch.object(mod, "PyPDFLoader", return_value=fake_loader_instance):
        docs = mod.load_document(str(pdf))

    processed = mod.preprocess_document(docs)
    # page1 likely gets filtered (<50 chars after cleaning), page2 kept
    assert len(processed) == 1
    assert processed[0] is page2
    assert "  " not in processed[0].page_content
    assert "\n" not in processed[0].page_content
