import asyncio
import requests
from playwright.async_api import async_playwright

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

async def main():
    p = await async_playwright().start()  # <-- manual start

    browser = await p.chromium.launch(
        headless=False,
        executable_path=CHROME_PATH,
        args=["--remote-debugging-port=9222"]
    )

    resp = requests.get("http://127.0.0.1:9222/json/version")
    ws_url = resp.json()["webSocketDebuggerUrl"]
    print("CDP URL:", ws_url)

    # connect over CDP
    cdp_browser = await p.chromium.connect_over_cdp(ws_url)
    context = cdp_browser.contexts[0]
    page = context.pages[0] if context.pages else await context.new_page()

    await page.goto("https://example.com")
    print("🌐 Browser opened. Close the browser window to exit the script.")

    try:
        # Set up event handler for browser disconnection
        browser_closed = asyncio.Event()
        
        def on_disconnected():
            print("🔴 Browser disconnected - exiting script...")
            browser_closed.set()
        
        # Listen for browser disconnection
        cdp_browser.on("disconnected", on_disconnected)
        
        # Wait until browser is closed
        await browser_closed.wait()
        
    except Exception as e:
        print(f"⚠️  Connection error: {e}")
    finally:
        try:
            await cdp_browser.close()
        except:
            pass
        try:
            await browser.close()
        except:
            pass
        await p.stop()
        print("✅ Script ended.")

asyncio.run(main())
