import requests
import json
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# NHK article URL from user's screenshot
# Title: 中国 軍民両用品目の対日輸出禁止を発表 日本の20企業など追加
# We need to find the article ID for this specific article

# First, get the page and find ALL API URLs referenced
r = requests.get("https://www3.nhk.or.jp/news/html/20260629/k10014733501000.html", headers=headers, timeout=10)
r.encoding = r.apparent_encoding

# Extract ALL JSON API URLs from the page source
api_urls = re.findall(r'["\']([^"\']*?\.json[^"\']*?)["\']', r.text)
print(f"Found {len(api_urls)} JSON URLs in page source")
for url in api_urls[:20]:
    print(f"  {url}")

# Try fetching some of the API URLs with the right referer
print("\n=== Testing API URLs ===")
for url in api_urls[:5]:
    full_url = url
    if url.startswith("/"):
        full_url = f"https://www3.nhk.or.jp{url}"
    elif not url.startswith("http"):
        continue
    
    try:
        api_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www3.nhk.or.jp/news/html/20260629/k10014733501000.html",
            "Origin": "https://www3.nhk.or.jp",
        }
        ar = requests.get(full_url, headers=api_headers, timeout=10)
        print(f"  {full_url}: status={ar.status_code}, len={len(ar.text)}")
        if ar.status_code == 200 and len(ar.text) > 100:
            try:
                data = json.loads(ar.text)
                print(f"  JSON keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                print(f"  Content: {json.dumps(data, ensure_ascii=False)[:300]}")
            except:
                pass
    except Exception as e:
        print(f"  {full_url}: error={e}")

# Also check if NHK has a different URL pattern for static content
# Some sites serve article JSON at /news/json/ or /news/data/ 
print("\n=== Testing alternative NHK JSON endpoints ===")
alt_urls = [
    "https://www3.nhk.or.jp/news/json/k10014733501000.json",
    "https://www3.nhk.or.jp/news/data/k10014733501000.json",
    "https://www3.nhk.or.jp/news/contents/k10014733501000.json",
    "https://news.web.nhk/api/v1/articles/k10014733501000",
    "https://api.web.nhk/r8/t/newsarticle/na/na-k10014733501000.json",
    "https://napi.web.nhk/r8/t/newsarticle/na/na-k10014733501000.json",
]
for url in alt_urls:
    try:
        ar = requests.get(url, headers=headers, timeout=5)
        print(f"  {url}: status={ar.status_code}, len={len(ar.text)}")
        if ar.status_code == 200 and len(ar.text) > 100:
            print(f"  Content: {ar.text[:300]}")
    except Exception as e:
        print(f"  {url}: error={e}")
