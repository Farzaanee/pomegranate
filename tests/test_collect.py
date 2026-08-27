import json

import pytest
import requests

from investment_rag.collect import CollectionError, clean_html, collect_source, save_documents
from investment_rag.models import SourceDocument


def test_clean_html_keeps_main_content_and_removes_navigation() -> None:
    title, text = clean_html("""
        <html><head><title>Investing basics</title><script>ignore()</script></head>
        <body><nav>Menu links</nav><main><h1>Risk</h1><p>Diversification spreads risk.</p></main>
        <footer>Cookie settings</footer></body></html>
    """)

    assert title == "Investing basics"
    assert text == "Risk Diversification spreads risk."
    assert "Menu" not in text


def test_save_documents_preserves_required_provenance(tmp_path) -> None:
    document = SourceDocument("MoneyHelper", "UK", "https://example.test/investing", "Basics", "Text")

    path = save_documents([document], tmp_path)[0]

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["source_name"] == "MoneyHelper"
    assert loaded["region"] == "UK"
    assert loaded["url"] == "https://example.test/investing"


def test_collect_source_wraps_http_error_with_source_context(monkeypatch) -> None:
    class BlockedResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("403 Client Error: Forbidden")

    class BlockedSession:
        headers = {}

        def get(self, url, timeout):
            return BlockedResponse()

    monkeypatch.setattr(requests, "Session", BlockedSession)

    with pytest.raises(CollectionError, match="MoneyHelper.*403"):
        collect_source({"name": "MoneyHelper", "region": "UK", "url": "https://example.test"})

