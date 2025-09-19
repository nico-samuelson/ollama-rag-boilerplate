import os
import io
import pytest

# ---------- Fakes ----------
class FakeDoc:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}

class FakePDFLoader:
    to_return = [FakeDoc("PDF page 1 content " + "x" * 80)]

    def __init__(self, path):
        self.path = path

    def load(self):
        return list(type(self).to_return)

class FakeTextLoader:
    to_return = [FakeDoc("TXT content " + "y" * 80)]

    def __init__(self, path):
        self.path = path

    def load(self):
        return list(type(self).to_return)


# ---------- Helpers ----------
def write_bytes(p, b):
    p.write_bytes(b)
    return str(p)


# ---------- Tests ----------
def test_validate_pdf_file_happy_path(tmp_path):
    from loader import DocumentLoader
    pdf = tmp_path / "doc.pdf"
    # Minimal valid header + some bytes
    write_bytes(pdf, b"%PDF" + b"\n%EOF")

    DocumentLoader(str(pdf)).validate_pdf_file()  # should not raise


def test_validate_pdf_file_missing_file(tmp_path):
    from loader import DocumentLoader
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError):
        DocumentLoader(str(missing)).validate_pdf_file()


def test_validate_pdf_file_wrong_extension(tmp_path):
    from loader import DocumentLoader
    txt = tmp_path / "doc.txt"
    write_bytes(txt, b"not a pdf")
    with pytest.raises(ValueError):
        DocumentLoader(str(txt)).validate_pdf_file()


def test_validate_pdf_file_invalid_header(tmp_path):
    from loader import DocumentLoader
    pdf = tmp_path / "doc.pdf"
    write_bytes(pdf, b"%PDX" + b"\nwhatever")
    with pytest.raises(ValueError):
        DocumentLoader(str(pdf)).validate_pdf_file()


def test_load_document_pdf_success(monkeypatch, tmp_path):
    import loader
    # Patch loaders with fakes
    monkeypatch.setattr(loader, "PyPDFLoader", FakePDFLoader)
    monkeypatch.setattr(loader, "TextLoader", FakeTextLoader)

    pdf = tmp_path / "doc.pdf"
    write_bytes(pdf, b"%PDF" + b"\n%EOF")

    dl = loader.DocumentLoader(str(pdf))
    docs = dl.load_document()
    assert isinstance(docs, list) and len(docs) == len(FakePDFLoader.to_return)
    assert all(hasattr(d, "page_content") for d in docs)


def test_load_document_txt_success(monkeypatch, tmp_path):
    import loader
    monkeypatch.setattr(loader, "PyPDFLoader", FakePDFLoader)
    monkeypatch.setattr(loader, "TextLoader", FakeTextLoader)

    txt = tmp_path / "doc.txt"
    txt.write_text("line 1\nline 2\n" + "z" * 100, encoding="utf-8")

    dl = loader.DocumentLoader(str(txt))
    docs = dl.load_document()
    assert isinstance(docs, list) and len(docs) == len(FakeTextLoader.to_return)
    assert all(hasattr(d, "page_content") for d in docs)


def test_load_document_unsupported_extension(tmp_path):
    from loader import DocumentLoader
    md = tmp_path / "doc.md"
    md.write_text("# not supported", encoding="utf-8")
    with pytest.raises(ValueError):
        DocumentLoader(str(md)).load_document()


def test_load_document_empty_results_raises(monkeypatch, tmp_path):
    import loader
    monkeypatch.setattr(loader, "PyPDFLoader", FakePDFLoader)
    monkeypatch.setattr(loader, "TextLoader", FakeTextLoader)

    # Force empty result from TextLoader
    FakeTextLoader.to_return = []

    txt = tmp_path / "empty.txt"
    txt.write_text("some content", encoding="utf-8")

    dl = loader.DocumentLoader(str(txt))
    with pytest.raises(RuntimeError):
        dl.load_document()

    # reset for other tests
    FakeTextLoader.to_return = [FakeDoc("TXT content " + "y" * 80)]


def test_load_document_loader_exception_propagates(monkeypatch, tmp_path):
    import loader

    class BoomTextLoader(FakeTextLoader):
        def load(self):
            raise IOError("disk error")

    monkeypatch.setattr(loader, "TextLoader", BoomTextLoader)

    txt = tmp_path / "boom.txt"
    txt.write_text("some content", encoding="utf-8")

    dl = loader.DocumentLoader(str(txt))
    with pytest.raises(IOError):
        dl.load_document()


def test_preprocess_document_filters_and_cleans():
    from loader import DocumentLoader

    # Short doc (<50) should be dropped
    short = FakeDoc("a b c  d")  # after cleanup still too short

    # Long doc with messy whitespace and short lines to remove
    long_text = (
        "Header\n"          # <= 6 chars but still >3, stays
        "a\n"               # <=3, gets removed
        "This   is   a      long   line   with   extra   spaces.   "
        "\n"
        "ok\n"
        + "content " * 20
    )
    long = FakeDoc(long_text)

    dl = DocumentLoader("ignored.txt")
    processed = dl.preprocess_document([short, long])

    # Only the long one should survive
    assert len(processed) == 1
    out = processed[0].page_content
    # Excess spaces collapsed
    assert "  " not in out
    # Very short lines removed
    assert "\na\n" not in out
    # Still substantial (>50 chars)
    assert len(out.strip()) > 50
