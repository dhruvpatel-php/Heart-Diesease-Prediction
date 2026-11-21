"""
=============================================================================
CardioScreen AI | Streamlit Keep-Alive Automation
Wakes up sleeping Streamlit Community Cloud apps via Headless Chromium.
=============================================================================
"""

import sys
import time
from playwright.sync_api import sync_playwright

# Monitored Streamlit Applications
URLS = [
    "https://texas-salary-prediction.streamlit.app",
    "https://customer-transaction-prediction.streamlit.app",
    "https://heart-diesease-prediction-by-dhruvv.streamlit.app/",
    "https://home-loan-default-prediction-by-dhruv.streamlit.app/",
    "https://incident-intelligence.streamlit.app/"
]

def wake_app(page, url: str):
    print(f"\n" + "=" * 55)
    print(f"[CHECKING] {url}")
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        status_code = response.status if response else "Unknown"
        print(f"  HTTP Status Code: {status_code}")
        
        # Give page time to execute Streamlit client scripts
        page.wait_for_timeout(6000)
        
        # Inspect page content for sleep indicator phrases
        body_text = page.inner_text("body")
        
        is_asleep = any(phrase in body_text.lower() for phrase in [
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
                print("  [WAITING] Allowing 12 seconds for Streamlit container spin-up...")
                page.wait_for_timeout(12000)
                print("  [SUCCESS] Wake-up signal dispatched successfully!")
            else:
                print("  [WARNING] Sleep text found, but wake-up button not located.")
        else:
            print("  [STATUS] App is already ACTIVE and AWAKE.")
            
    except Exception as e:
        print(f"  [PROBE RESULT] Finished checking {url}: {e}")

def main():
    print("[START] Streamlit Keep-Alive Automation Initialized")
    print(f"[INFO] Monitoring {len(URLS)} applications...")
    
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
        
        for url in URLS:
            wake_app(page, url)
            
        browser.close()
        
    print("\n" + "=" * 55)
    print("[COMPLETED] All 5 Streamlit applications checked and kept alive.")

if __name__ == "__main__":
    main()
