from rag_collection.urls import canonicalize_url, infer_language, is_official_url


def test_canonicalize_url_drops_fragments_and_tracking_params() -> None:
    url = "HTTPS://WWW.SHANGHAITECH.EDU.CN/path/?utm_source=x&b=2&a=1#section"

    assert canonicalize_url(url) == "https://www.shanghaitech.edu.cn/path/?a=1&b=2"


def test_canonicalize_url_percent_encodes_path_control_characters() -> None:
    url = "http://sist.shanghaitech.edu.cn/office/Share/Entropy-SIST Yearbook,2020.pdf"

    assert canonicalize_url(url) == (
        "http://sist.shanghaitech.edu.cn/office/Share/Entropy-SIST%20Yearbook,2020.pdf"
    )


def test_official_url_accepts_subdomains_only() -> None:
    assert is_official_url("https://sist.shanghaitech.edu.cn/")
    assert is_official_url("https://faculty.sist.shanghaitech.edu.cn/")
    assert not is_official_url("https://shanghaitech.edu.cn.evil.example/")


def test_infer_language_prefers_chinese_when_present() -> None:
    assert infer_language("信息科学与技术学院负责本科生和研究生培养。") == "zh"
    assert infer_language("School of Information Science and Technology") == "en"
