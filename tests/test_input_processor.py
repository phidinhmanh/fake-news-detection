import pytest
from sequential_adversarial.input_processor import InputProcessor

def test_strip_html_logic():
    ip = InputProcessor()
    html = """
    <html>
        <head><title>Test</title></head>
        <body>
            <nav>Menu</nav>
            <header>Banner</header>
            <main>
                <h1>Title</h1>
                <p>Hello world!</p>
                <script>alert('hidden');</script>
                <style>.css { color: red; }</style>
            </main>
            <footer>Copyright</footer>
        </body>
    </html>
    """
    stripped = ip._strip_html(html)
    # Nav, Header, Footer, Script, Style should be skipped
    assert "Menu" not in stripped
    assert "Banner" not in stripped
    assert "Copyright" not in stripped
    assert "alert" not in stripped
    assert "color: red" not in stripped
    # Content should be kept
    assert "Title" in stripped
    assert "Hello world!" in stripped

def test_detect_type_variants():
    ip = InputProcessor()
    assert ip._detect_type("HTTPS://GOOGLE.COM") == "url"
    assert ip._detect_type("not/a/file/path/that/exists") == "raw"

def test_handle_raw_metadata():
    ip = InputProcessor()
    text = "Tiếng Việt có dấu."
    _, meta = ip._handle_raw(text)
    assert meta["source_type"] == "raw_text"
    assert meta["char_count"] == len(text)

def test_process_strips_input():
    ip = InputProcessor()
    out = ip.process("   https://example.com   ")
    assert out["source"] == "https://example.com"
    assert out["input_type"] == "url"

def test_read_file_not_found(tmp_path):
    ip = InputProcessor()
    # Path is detected as file ONLY if it exists. 
    # If we pass a string that isn't a file, it's 'raw'.
    assert ip._detect_type("non_existent_file.txt") == "raw"

def test_max_chars_applied():
    ip = InputProcessor()
    text = "X" * 10000
    out = ip.process(text)
    assert len(out["raw_text"]) == InputProcessor.MAX_CHARS
