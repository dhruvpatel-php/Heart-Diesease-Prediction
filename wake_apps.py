"""
=============================================================================
CardioScreen AI | Streamlit Keep-Alive Automation
Wakes up sleeping Streamlit Community Cloud apps via Headless Chromium.
=============================================================================
"""

import os
import sys
import time
from playwright.sync_api import sync_playwright

# Monitored Streamlit Applications (All 5 user apps)
DEFAULT_URLS = [
    "https://heart-diesease-prediction-by-dhruvv.streamlit.app/",
    "https://texas-salary-prediction.streamlit.app",
    "https://customer-transaction-prediction.streamlit.app",
    "https://home-loan-default-prediction-by-dhruv.streamlit.app/",
    "https://incident-intelligence.streamlit.app/"
]

def get_target_urls():
    env_url = os.environ.get("STREAMLIT_APP_URL", "").strip()
    urls = list(DEFAULT_URLS)
    if env_url:
        for u in env_url.splitlines():
            u = u.strip()
            if u and u not in urls:
                urls.append(u)
    return urls

def probe_and_wake(page, url: str):
    print("\n" + "=" * 60)
    print(f"[PROBING] {url}")
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        status_code = response.status if response else "Unknown"
        print(f"  HTTP Status Code: {status_code}")
        
        # Give page time to execute Streamlit client scripts
        page.wait_for_timeout(6000)
        
        # Inspect page content for sleep indicator phrases
        content = page.content().lower()
        is_asleep = any(phrase in content for phrase in [
            "gone to sleep", "zzzz", "due to inactivity", "wake it back up"
        ])
        
        if is_asleep:
            print("  [STATUS] App is currently ASLEEP ('Zzzz' screen detected).")
            print("  [ACTION] Locating 'Yes, get this app back up!' button...")
            
            wake_btn = None
            selectors = [
                "button:has-text('get this app back up')",
                "button:has-text('Yes, get this app back up!')",
                "button:has-text('wake')",
                "button[kind='primary']",
                "button"
            ]
            
            for sel in selectors:
                locator = page.locator(sel)
                if locator.count() > 0:
                    wake_btn = locator.first
                    break
            
            if wake_btn:
                wake_btn.click()
                print("  [ACTION] Clicked 'Yes, get this app back up!' button.")
                print("  [WAITING] Allowing 15 seconds for Streamlit container spin-up...")
                page.wait_for_timeout(15000)
                print("  [SUCCESS] Wake-up signal dispatched successfully!")
            else:
                print("  [WARNING] Sleep text found, but wake-up button not located.")
        else:
            print("  [STATUS] App is already ACTIVE and AWAKE.")
            
    except Exception as e:
        print(f"  [PROBE RESULT] Finished checking {url}: {e}")

def main():
    urls = get_target_urls()
    print("[START] Streamlit Keep-Alive Automation Initialized")
    print(f"[CONFIG] Monitoring {len(urls)} Streamlit applications:")
    for u in urls:
        print(f"  - {u}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        page = context.new_page()
        
        for url in urls:
            probe_and_wake(page, url)
            
        browser.close()
        
    print("\n" + "=" * 60)
    print("[COMPLETED] All 5 Streamlit applications checked and kept alive.")

if __name__ == "__main__":
    main()
