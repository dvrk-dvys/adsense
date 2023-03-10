from playwright.sync_api import Playwright, sync_playwright, expect
def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    # Open new page
    page = context.new_page()
    # Go to https://www.wikipedia.org/
    page.goto("https://www.wikipedia.org/")
    # Click text=Wikipedia The Free Encyclopedia English 6 458 000+ articles 日本語 1 314 000+ 記事 Es
    page.locator("text=Wikipedia The Free Encyclopedia English 6 458 000+ articles 日本語 1 314 000+ 記事 Es").click()
    # Click text=Wikipedia The Free Encyclopedia >> span
    page.locator("text=Wikipedia The Free Encyclopedia >> span").click()
    # Click text=Wikipedia The Free Encyclopedia English 6 458 000+ articles 日本語 1 314 000+ 記事 Es
    page.locator("text=Wikipedia The Free Encyclopedia English 6 458 000+ articles 日本語 1 314 000+ 記事 Es").click()
    # Click strong:has-text("English")
    page.locator("strong:has-text(\"English\")").click()
    page.wait_for_url("https://en.wikipedia.org/wiki/Main_Page")
    # Click text=View source
    page.locator("text=View source").click()
    page.wait_for_url("https://en.wikipedia.org/w/index.php?title=Main_Page&action=edit")
    # Click text=View history
    page.locator("text=View history").click()
    page.wait_for_url("https://en.wikipedia.org/w/index.php?title=Main_Page&action=history")
    # Click text=Log in
    page.locator("text=Log in").click()
    page.wait_for_url("https://en.wikipedia.org/w/index.php?title=Special:UserLogin&returnto=Main+Page&returntoquery=action%3Dhistory")
    # ---------------------
    context.close()
    browser.close()
with sync_playwright() as playwright:
    run(playwright)