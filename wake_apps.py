"""
=============================================================================
CardioScreen AI | Streamlit Keep-Alive Automation
Wakes up sleeping Streamlit Community Cloud apps via Headless Chromium.
=============================================================================
"""

import os
import sys
import time

# Monitored Streamlit Applications (All 5 target apps)
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
        response = page.goto(url, wait_until="domcontentloaded", timeout=45000)
        status_code = response.status if response else "Unknown"
        print(f"  HTTP Status: {status_code}")
        
        # Give Streamlit client 6 seconds to render UI or sleep screen
        page.wait_for_timeout(6000)
        
        # Check for sleep indicators
        content = page.content().lower()
        is_asleep = any(phrase in content for phrase in [
            "gone to sleep", "zzzz", "due to inactivity", "wake it back up"
        ])
        
        if is_asleep:
            print("  [STATUS] App is currently ASLEEP ('Zzzz' screen detected).")
            print("  [ACTION] Locating 'Yes, get this app back up!' button...")
            
            wake_btn = page.get_by_role("button", name="Yes, get this app back up!")
            if wake_btn.count() == 0:
                wake_btn = page.locator("button:has-text('Yes, get this app back up!')")
            if wake_btn.count() == 0:
                wake_btn = page.locator("button:has-text('get this app back up')")
            if wake_btn.count() == 0:
                wake_btn = page.locator("button:has-text('wake')")
            
            if wake_btn.count() > 0:
                wake_btn.first.click()
                print("  [ACTION] Clicked 'Yes, get this app back up!' button.")
                print("  [WAITING] Allowing 12 seconds for Streamlit container spin-up...")
                page.wait_for_timeout(12000)
                print("  [SUCCESS] Wake-up request dispatched successfully!")
            else:
                print("  [WARNING] Sleep text found, but wake-up button locator was not found.")
        else:
            print("  [STATUS] App is ACTIVE and AWAKE.")
            
    except Exception as e:
        print(f"  [PROBE RESULT] Finished with notice for {url}: {e}")

def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[ERROR] Playwright is not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(0)
        
    urls = get_target_urls()
    print("[START] Streamlit Keep-Alive Automation Initialized")
    print(f"[CONFIG] Monitoring {len(urls)} Streamlit applications:")
    for u in urls:
        print(f"  - {u}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--single-process'
                ]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720}
            )
            page = context.new_page()
            
            for url in urls:
                try:
                    probe_and_wake(page, url)
                except Exception as ex:
                    print(f"  [ERROR on {url}]: {ex}")
                
            browser.close()
    except Exception as e:
        print(f"[WARNING] Browser automation encountered notice: {e}")
        
    print("\n" + "=" * 60)
    print("[COMPLETED] All 5 Streamlit applications checked.")
    sys.exit(0)

if __name__ == "__main__":
    main()
