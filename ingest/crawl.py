"""Crawl ready.gov (HTML + PDF) into SQLite.

Lifted from the original notebook's crawler cell almost unchanged - same
logic, same schema - just made importable/runnable as a script (and driven
by a couple of CLI flags instead of editing constants in a notebook cell)
so Kestra can call it directly.
"""
import argparse
import sqlite3
import time
from collections import deque
from datetime import datetime
from io import BytesIO
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from markdownify import markdownify
from pypdf import PdfReader

from common.config import SQLITE_DB_PATH

USER_AGENT = "ReadyCrawler/1.0 (Educational Purposes)"
CRAWL_DELAY = 1.0


def setup_database(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            markdown_content TEXT,
            scraped_at TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def is_valid_url(url: str) -> bool:
    """Allow any domain, allow PDFs, block multimedia/data."""
    parsed = urlparse(url)
    if any(parsed.path.lower().endswith(ext) for ext in [".jpg", ".png", ".zip", ".mp4", ".xml", ".json"]):
        return False
    if any(block in parsed.path.lower() for block in ["/es/", "/espanol/", "/zh/"]):
        return False
    return True


def is_english_text(text: str) -> bool:
    clean_text = text.strip()
    if len(clean_text) < 50:
        return False
    try:
        return detect(clean_text) == "en"
    except LangDetectException:
        return False


def extract_pdf_text(response_content: bytes):
    try:
        reader = PdfReader(BytesIO(response_content))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        title = "Unknown PDF"
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title
        return title, text.strip()
    except Exception as e:
        print(f"      -> Failed to parse PDF: {e}")
        return None, None


def clean_html(soup: BeautifulSoup):
    for element in soup(["header", "footer", "nav", "aside", "script", "style", "noscript", "form", "iframe"]):
        element.decompose()
    main_content = soup.find("main") or soup.find(id="content") or soup.find(role="main")
    if not main_content:
        main_content = soup.body
    return main_content


def crawl(start_url: str, max_pages: int, db_path: str) -> int:
    """Runs the crawl; returns the number of pages saved."""
    conn = setup_database(db_path)
    cursor = conn.cursor()
    pages_scraped = 0

    try:
        visited = set()
        queue = deque([start_url])
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        robots_cache = {}

        print(f"Starting crawl at {start_url}...")

        while queue:
            if pages_scraped >= max_pages:
                print(f"\nReached limit of {max_pages} pages. Stopping.")
                break

            current_url = queue.popleft()
            normalized_url = current_url.split("#")[0]
            parsed_url = urlparse(normalized_url)
            domain = parsed_url.netloc

            if normalized_url in visited:
                continue

            if domain not in robots_cache:
                rp = RobotFileParser()
                rp.set_url(f"{parsed_url.scheme}://{domain}/robots.txt")
                try:
                    robots_resp = session.get(rp.url, timeout=5)
                    if robots_resp.status_code == 200:
                        rp.parse(robots_resp.text.splitlines())
                except Exception:
                    pass
                robots_cache[domain] = rp

            if not robots_cache[domain].can_fetch(USER_AGENT, normalized_url):
                print(f"  -> Blocked by robots.txt: {normalized_url}")
                visited.add(normalized_url)
                continue

            print(f"Fetching: {normalized_url}")
            visited.add(normalized_url)

            try:
                try:
                    response = session.get(normalized_url, timeout=15)
                    response.raise_for_status()
                except requests.RequestException as e:
                    print(f"  -> Network error: {e}")
                    continue

                content_type = response.headers.get("Content-Type", "").lower()
                title, final_text, is_pdf = "", "", False

                if "application/pdf" in content_type or normalized_url.lower().endswith(".pdf"):
                    is_pdf = True
                    pdf_title, pdf_text = extract_pdf_text(response.content)
                    if not pdf_text:
                        continue
                    title = pdf_title if pdf_title != "Unknown PDF" else normalized_url.split("/")[-1]
                    final_text = f"# {title}\n\n{pdf_text}"
                elif "text/html" in content_type:
                    soup = BeautifulSoup(response.text, "html.parser")
                    html_tag = soup.find("html")
                    if html_tag and html_tag.has_attr("lang") and not html_tag["lang"].lower().startswith("en"):
                        print("  -> Skipped (HTML lang is not English)")
                        continue
                    main_content = clean_html(soup)
                    if not main_content:
                        continue
                    final_text = markdownify(str(main_content), heading_style="ATX", strip=["a", "img"])
                    title_tag = soup.find("title")
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"
                else:
                    continue

                if not is_english_text(final_text):
                    print("  -> Skipped (Detected non-English text)")
                    continue

                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO pages (url, title, markdown_content, scraped_at) VALUES (?, ?, ?, ?)",
                        (normalized_url, title, final_text.strip(), datetime.now().isoformat()),
                    )
                    conn.commit()
                    pages_scraped += 1
                    print(f"      [+] Saved {'PDF' if is_pdf else 'HTML'}: {title}")
                except sqlite3.Error as e:
                    print(f"  -> Database error: {e}")

                if not is_pdf:
                    for link in soup.find_all("a", href=True):
                        absolute_url = urljoin(normalized_url, link["href"])
                        clean_url = absolute_url.split("#")[0]
                        if clean_url not in visited and is_valid_url(clean_url):
                            queue.append(clean_url)

            except Exception as e:
                print(f"  -> Unexpected error: {e}")

            time.sleep(CRAWL_DELAY)

    finally:
        conn.close()
        print("\nDatabase connection safely closed.")

    return pages_scraped


def main():
    parser = argparse.ArgumentParser(description="Crawl ready.gov into SQLite.")
    parser.add_argument("--start-url", default="https://www.ready.gov/hurricanes")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--db-path", default=SQLITE_DB_PATH)
    args = parser.parse_args()

    try:
        count = crawl(args.start_url, args.max_pages, args.db_path)
        print(f"\nDone. {count} pages saved to {args.db_path}.")
    except KeyboardInterrupt:
        print("\n\nCrawling interrupted by user. Shutting down gracefully...")


if __name__ == "__main__":
    main()
