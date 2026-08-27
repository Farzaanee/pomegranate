"""Download and clean trusted public educational pages."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

from .models import SourceDocument

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
REMOVABLE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript")


class CollectionError(RuntimeError):
    """A source could not be downloaded without weakening access controls."""


def load_sources(path: str | Path) -> list[dict[str, str]]:
    """Load source definitions that supply required provenance metadata."""
    sources = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "region", "url"}
    if not isinstance(sources, list) or any(required - item.keys() for item in sources):
        raise ValueError("Each source must include name, region, and url.")
    return sources


def clean_html(html: str) -> tuple[str, str]:
    """Extract readable, normalized page text and its title."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled"
    for tag in soup.find_all(REMOVABLE_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()
    return title, text


def _same_site_link(url: str, source_url: str) -> bool:
    parsed = urlparse(url)
    source = urlparse(source_url)
    return parsed.scheme in {"http", "https"} and parsed.netloc == source.netloc


def _discover_links(html: str, source_url: str, limit: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for anchor in soup.find_all("a", href=True):
        url = urldefrag(urljoin(source_url, anchor["href"]))[0]
        if _same_site_link(url, source_url) and url not in candidates:
            candidates.append(url)
        if len(candidates) >= limit:
            break
    return candidates


def collect_source(source: dict[str, str], pages_per_source: int = 1) -> list[SourceDocument]:
    """Fetch a seed page and, optionally, same-site pages linked from it."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Accept-Language"] = "en-GB,en;q=0.9"
    try:
        seed_response = session.get(source["url"], timeout=30)
        seed_response.raise_for_status()
    except requests.RequestException as error:
        raise CollectionError(f"{source['name']} ({source['url']}) could not be collected: {error}") from error
    urls = [source["url"]]
    if pages_per_source > 1:
        urls.extend(_discover_links(seed_response.text, source["url"], pages_per_source - 1))

    documents = []
    for url in urls:
        try:
            response = seed_response if url == source["url"] else session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            if url == source["url"]:
                raise
            continue
        title, text = clean_html(response.text)
        if text:
            documents.append(SourceDocument(source["name"], source["region"], url, title, text))
    return documents


def save_documents(documents: list[SourceDocument], output_dir: str | Path) -> list[Path]:
    """Persist clean documents as JSON for review and repeatable indexing."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for document in documents:
        digest = hashlib.sha256(document.url.encode()).hexdigest()[:12]
        path = directory / f"{digest}.json"
        path.write_text(json.dumps(document.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(path)
    return paths
