import logging
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


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


def download_pdf(pdf_url: str, driver: webdriver.Chrome) -> bool:
    """Download a single PDF using Selenium."""
    try:
        driver.get(pdf_url)
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(5)
        return True
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        return False
