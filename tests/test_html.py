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


def test_extract_html_drops_page_chrome_while_keeping_main_content() -> None:
    _, text, _ = extract_html(
        b"""
        <html>
          <body>
            <header>School logo English Navigation</header>
            <nav>Home About Research Faculty Academics</nav>
            <main>
              <article>
                <h1>2026 Academic Seminar</h1>
                <p>Speaker: ShanghaiTech SIST professor</p>
              </article>
            </main>
            <footer>Copyright ShanghaiTech address QR code</footer>
          </body>
        </html>
        """,
        "https://sist.shanghaitech.edu.cn/2026/0522/c11304a1122875/page.htm",
    )

    assert "2026 Academic Seminar" in text
    assert "Speaker: ShanghaiTech SIST professor" in text
    assert "School logo" not in text
    assert "Home About Research Faculty" not in text
    assert "Copyright ShanghaiTech" not in text
