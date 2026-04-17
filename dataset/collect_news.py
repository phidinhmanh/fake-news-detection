"""
collect_news.py — Thu thập tin tức từ báo VN (VnExpress, Tuổi Trẻ)
=====================================================================
Thu thập 300-500 mẫu tin thật từ các báo chính thống để bổ sung dataset.

Chạy:
    python dataset/collect_news.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Headers giả lập trình duyệt để tránh bị chặn
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Timeout cho mỗi request
REQUEST_TIMEOUT = 15


def _hash_text(text: str) -> str:
    """Tạo hash để deduplicate."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def crawl_vnexpress(
    max_articles: int = 250,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """Crawl bài viết từ VnExpress.net (tin thật).

    Args:
        max_articles: Số bài tối đa cần crawl.
        categories: Danh sách categories (mặc định: thời sự, kinh doanh, sức khỏe, xã hội).

    Returns:
        List of dicts với keys: text, title, url, category, source.
    """
    if categories is None:
        categories = [
            "thoi-su",
            "kinh-doanh",
            "suc-khoe",
            "the-gioi",
            "giao-duc",
            "khoa-hoc",
        ]

    articles: list[dict] = []
    seen_hashes: set[str] = set()

    for category in categories:
        if len(articles) >= max_articles:
            break

        logger.info(f"  📰 Crawling VnExpress/{category}...")
        base_url = f"https://vnexpress.net/{category}"

        try:
            resp = requests.get(base_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Tìm các link bài viết
            article_links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if (
                    href.startswith("https://vnexpress.net/")
                    and href.count("-") >= 3
                    and href[-1].isdigit()
                    and "/tag/" not in href
                    and "/video/" not in href
                ):
                    article_links.append(href)

            article_links = list(dict.fromkeys(article_links))[:30]  # Deduplicate, max 30

            for url in article_links:
                if len(articles) >= max_articles:
                    break

                article = _crawl_single_article(url, category, "vnexpress")
                if article:
                    text_hash = _hash_text(article["text"])
                    if text_hash not in seen_hashes:
                        seen_hashes.add(text_hash)
                        articles.append(article)

                time.sleep(0.5)  # Rate limiting

        except Exception as exc:
            logger.warning(f"⚠️ Lỗi crawl VnExpress/{category}: {exc}")

    logger.info(f"✅ VnExpress: {len(articles)} bài viết")
    return articles


def crawl_tuoitre(
    max_articles: int = 250,
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """Crawl bài viết từ Tuổi Trẻ Online (tin thật).

    Args:
        max_articles: Số bài tối đa cần crawl.
        categories: Danh sách categories.

    Returns:
        List of dicts với keys: text, title, url, category, source.
    """
    if categories is None:
        categories = [
            "thoi-su",
            "the-gioi",
            "kinh-doanh",
            "suc-khoe",
            "giao-duc",
        ]

    articles: list[dict] = []
    seen_hashes: set[str] = set()

    for category in categories:
        if len(articles) >= max_articles:
            break

        logger.info(f"  📰 Crawling Tuổi Trẻ/{category}...")
        base_url = f"https://tuoitre.vn/{category}.htm"

        try:
            resp = requests.get(base_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            article_links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if not href.startswith("http"):
                    href = "https://tuoitre.vn" + href

                if (
                    "tuoitre.vn" in href
                    and href.endswith(".htm")
                    and href.count("-") >= 2
                    and "/tag/" not in href
                    and "/video/" not in href
                ):
                    article_links.append(href)

            article_links = list(dict.fromkeys(article_links))[:30]

            for url in article_links:
                if len(articles) >= max_articles:
                    break

                article = _crawl_single_article(url, category, "tuoitre")
                if article:
                    text_hash = _hash_text(article["text"])
                    if text_hash not in seen_hashes:
                        seen_hashes.add(text_hash)
                        articles.append(article)

                time.sleep(0.5)

        except Exception as exc:
            logger.warning(f"⚠️ Lỗi crawl Tuổi Trẻ/{category}: {exc}")

    logger.info(f"✅ Tuổi Trẻ: {len(articles)} bài viết")
    return articles


def _crawl_single_article(url: str, category: str, source: str) -> Optional[dict]:
    """Crawl nội dung 1 bài viết.

    Args:
        url: URL bài viết.
        category: Category (thoi-su, suc-khoe, etc.)
        source: Tên nguồn (vnexpress, tuoitre).

    Returns:
        Dict chứa thông tin bài viết hoặc None nếu lỗi.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract body text
        text_parts: list[str] = []

        # Phương pháp 1: Tìm article body
        body_selectors = [
            "article.fck_detail",         # VnExpress
            "div.fck_detail",              # VnExpress fallback
            "div.detail-content",          # Tuổi Trẻ
            "div#main-detail-body",        # Tuổi Trẻ fallback
            "div.content-detail",          # Generic
            "article",                      # Generic article
        ]

        for selector in body_selectors:
            body = soup.select_one(selector)
            if body:
                for p in body.find_all("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 20:  # Skip short fragments
                        text_parts.append(text)
                break

        # Fallback: all paragraphs
        if not text_parts:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 30:
                    text_parts.append(text)

        full_text = " ".join(text_parts)

        # Validate: phải có đủ nội dung
        if len(full_text) < 100:
            return None

        return {
            "text": full_text,
            "title": title,
            "url": url,
            "category": category,
            "source": source,
            "label": "real",  # Báo chính thống → real
        }

    except Exception:
        return None


def main() -> None:
    """Thu thập tin tức từ báo VN."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import COLLECTED_DIR

    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU THU THẬP TIN TỨC TỪ BÁO VIỆT NAM")
    logger.info("=" * 60)

    all_articles: list[dict] = []

    # Crawl VnExpress
    vnexpress_articles = crawl_vnexpress(max_articles=250)
    all_articles.extend(vnexpress_articles)

    # Crawl Tuổi Trẻ
    tuoitre_articles = crawl_tuoitre(max_articles=250)
    all_articles.extend(tuoitre_articles)

    if not all_articles:
        logger.warning("⚠️ Không thu thập được bài viết nào!")
        return

    # Save to CSV
    df = pd.DataFrame(all_articles)
    csv_path = COLLECTED_DIR / "collected_news.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"\n✅ Saved: {csv_path} ({len(df)} articles)")

    # Save as JSONL (for reference)
    jsonl_path = COLLECTED_DIR / "collected_news.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for article in all_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    # Stats
    logger.info(f"\n📊 Stats:")
    logger.info(f"   VnExpress: {len(vnexpress_articles)}")
    logger.info(f"   Tuổi Trẻ:  {len(tuoitre_articles)}")
    logger.info(f"   Total:     {len(all_articles)}")
    logger.info(f"   Sources:   {df['source'].value_counts().to_dict()}")
    logger.info(f"   Categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
