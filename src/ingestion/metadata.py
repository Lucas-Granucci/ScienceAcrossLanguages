import random
import time
from typing import Dict, List

import requests


def fetch_openalex_metadata(lang_code: str, max_articles: int) -> List[Dict]:
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"language:{lang_code},type:article",
        "select": "primary_location,title,doi,publication_date",
        "mailto": "example@email.com",
        "per-page": 100,
        "cursor": "*",
    }

    session = requests.Session()
    article_data = []
    total_articles = 0
    next_cursor = 0

    while total_articles < max_articles:
        response = session.get(url, params=params)

        try:
            if "next_cursor" not in response.json()["meta"]:
                break

            next_cursor = response.json()["meta"]["next_cursor"]
            results = response.json()["results"]

            for result in results:
                primary_location = result["primary_location"]
                pdf_url = primary_location.get("pdf_url", "")

                if not pdf_url:
                    continue

                article_data.append(
                    {
                        "title": result["title"],
                        "pdf_url": pdf_url,
                        "doi": result["doi"],
                        "publication_date": result["publication_date"],
                    }
                )

                total_articles += 1
                if total_articles >= max_articles:
                    break

        except Exception as e:
            print(f"Error downloading article info: {str(e)}")

        params["cursor"] = next_cursor
        time.sleep(random.randint(4, 6))

    return article_data
