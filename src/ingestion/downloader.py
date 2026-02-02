import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


def setup_pdf_driver(download_dir: Path):
    """Configure Chrome WebDriver for PDF downloads."""

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,
            "plugins.plugins_disabled": ["Chrome PDF Viewer"],
        },
    )
    return webdriver.Chrome(options=options)


def download_pdf(driver, pdf_url: str):
    """Download a single PDF using Selenium."""
    driver.get(pdf_url)
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(5)


def download_pdfs(article_metadata: pd.DataFrame, output_path: Path):
    driver = setup_pdf_driver(output_path)

    # Download PDFs
    success_count = 0
    for _, article in tqdm(
        article_metadata.iterrows(),
        total=len(article_metadata),
        desc="Downloading PDFs...",
    ):
        try:
            download_pdf(driver, article.pdf_url)
            success_count += 1
        except Exception:
            pass

    driver.quit()
