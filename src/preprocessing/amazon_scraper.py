"""
Live Amazon Review Scraper — P1 Real Data
Tries Playwright (JS-rendered) first, falls back to requests+BS4.
Usage: scrape_amazon_reviews("B0FNJS4N2H", max_reviews=10)
Returns List[Dict] with keys: review_id, reviewer_id, product_id, rating, verified_purchase, timestamp, text
"""
import re, time, random
from typing import List, Dict, Any
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _parse_rating(text: str) -> int:
    m = re.search(r"([1-5])\.0 out of 5", text)
    if m: return int(m.group(1))
    m = re.search(r"([1-5])", text)
    return int(m.group(1)) if m else 4

def scrape_via_requests(product_id: str, max_reviews: int = 10, domain: str = "amazon.in") -> List[Dict[str, Any]]:
    """Fallback: static HTTP fetch of product-reviews page."""
    import requests
    from bs4 import BeautifulSoup
    urls = [
        f"https://www.{domain}/product-reviews/{product_id}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews",
        f"https://www.{domain}/dp/{product_id}",
    ]
    reviews = []
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            blocks = soup.select('[data-hook="review"]')
            for idx, blk in enumerate(blocks[:max_reviews]):
                try:
                    text_el = blk.select_one('[data-hook="review-body"]')
                    text = text_el.get_text(" ", strip=True) if text_el else ""
                    if len(text) < 20:
                        continue
                    rating_el = blk.select_one('[data-hook="review-star-rating"]')
                    rating_text = rating_el.get_text() if rating_el else "5.0 out of 5"
                    rating = _parse_rating(rating_text)
                    verified = bool(blk.select_one('[data-hook="avp-badge"]'))
                    # reviewer
                    reviewer_el = blk.select_one('.a-profile-name')
                    reviewer = reviewer_el.get_text(strip=True) if reviewer_el else f"u{idx}"
                    # date
                    date_el = blk.select_one('[data-hook="review-date"]')
                    date_text = date_el.get_text() if date_el else ""
                    # Try parse "Reviewed in India on 5 March 2024"
                    timestamp = datetime.now().isoformat()
                    reviews.append({
                        "review_id": f"{product_id}_live_r{idx}",
                        "reviewer_id": re.sub(r'\W+', '_', reviewer)[:20] or f"u{idx}",
                        "product_id": product_id,
                        "rating": rating,
                        "verified_purchase": verified,
                        "timestamp": timestamp,
                        "text": text[:800]
                    })
                except Exception:
                    continue
            if reviews:
                print(f"[Scraper][requests] {len(reviews)} reviews from {url}")
                return reviews[:max_reviews]
        except Exception as e:
            print(f"[Scraper][requests] failed {url}: {e}")
    return reviews

def scrape_via_playwright(product_id: str, max_reviews: int = 10, domain: str = "amazon.in") -> List[Dict[str, Any]]:
    """JS-rendered via Playwright (if installed)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[Scraper] playwright not installed, skip")
        return []
    reviews = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = browser.new_context(user_agent=HEADERS["User-Agent"], locale="en-US")
            page = ctx.new_page()
            # Try reviews page first
            for url in [
                f"https://www.{domain}/product-reviews/{product_id}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&sortBy=recent",
                f"https://www.{domain}/dp/{product_id}",
            ]:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    page.wait_for_timeout(2500)
                    # Try dismiss popup
                    for sel in ['button:has-text("Dismiss")', '#sp-cc-accept', 'input[data-action-type="DISMISS"]']:
                        try: page.locator(sel).first.click(timeout=1000)
                        except: pass
                    blocks = page.locator('[data-hook="review"]').all()
                    if not blocks:
                        continue
                    for idx, blk in enumerate(blocks[:max_reviews]):
                        try:
                            text = blk.locator('[data-hook="review-body"]').inner_text(timeout=1000) if blk.locator('[data-hook="review-body"]').count() else ""
                            if len(text.strip()) < 20:
                                # fallback
                                text = blk.inner_text()[:500]
                                if len(text) < 20: continue
                            rating_text = blk.locator('[data-hook="review-star-rating"]').first.inner_text(timeout=1000) if blk.locator('[data-hook="review-star-rating"]').count() else "5.0 out of 5"
                            rating = _parse_rating(rating_text)
                            verified = blk.locator('[data-hook="avp-badge"]').count() > 0
                            reviewer = blk.locator('.a-profile-name').first.inner_text(timeout=1000) if blk.locator('.a-profile-name').count() else f"u{idx}"
                            reviews.append({
                                "review_id": f"{product_id}_live_r{idx}",
                                "reviewer_id": re.sub(r'\W+', '_', reviewer)[:20] or f"u{idx}",
                                "product_id": product_id,
                                "rating": rating,
                                "verified_purchase": verified,
                                "timestamp": datetime.now().isoformat(),
                                "text": text.strip()[:800]
                            })
                        except Exception:
                            continue
                    if reviews:
                        print(f"[Scraper][playwright] {len(reviews)} reviews from {url}")
                        browser.close()
                        return reviews[:max_reviews]
                except Exception as e:
                    print(f"[Scraper][playwright] {url} failed: {e}")
            browser.close()
    except Exception as e:
        print(f"[Scraper][playwright] error: {e}")
    return reviews

def scrape_amazon_reviews(product_id: str, max_reviews: int = 10, domain: str = "amazon.in") -> List[Dict[str, Any]]:
    """Public: try live scrape, return [] if blocked/empty (caller falls back to synthetic)."""
    # 1. Playwright
    reviews = scrape_via_playwright(product_id, max_reviews, domain)
    if len(reviews) >= 3:
        return reviews
    # 2. Requests
    reviews2 = scrape_via_requests(product_id, max_reviews, domain)
    if len(reviews2) >= 3:
        return reviews2
    # 3. Try .com
    if domain != "amazon.com":
        reviews3 = scrape_via_requests(product_id, max_reviews, "amazon.com")
        if len(reviews3) >= 3:
            return reviews3
    print(f"[Scraper] Live failed for {product_id} (got {len(reviews)}+{len(reviews2) if 'reviews2' in locals() else 0}), fallback to synthetic")
    return []
