from rag_collection.html import extract_html


def test_extract_html_removes_scripts_and_resolves_links() -> None:
    title, text, links = extract_html(
        b"""
        <html>
          <head><title>SIST</title><script>bad()</script></head>
          <body><h1>School</h1><a href="/news/">News</a></body>
        </html>
        """,
        "https://sist.shanghaitech.edu.cn/",
    )

    assert title == "SIST"
    assert "bad" not in text
    assert "School" in text
    assert links == ["https://sist.shanghaitech.edu.cn/news/"]
