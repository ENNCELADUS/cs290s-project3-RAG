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


def test_extract_html_uses_declared_charset_before_utf8_replacement() -> None:
    html = """
    <html>
      <head><meta charset="gb2312"><title>学院新闻</title></head>
      <body><p>信息科学与技术学院</p></body>
    </html>
    """.encode("gb18030")

    title, text, _ = extract_html(html, "https://sist.shanghaitech.edu.cn/")

    assert title == "学院新闻"
    assert "信息科学与技术学院" in text
    assert "\ufffd" not in text
