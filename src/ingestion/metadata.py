import time
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""

    word_indeces = []
    for word, indeces in inverted_index.items():
        word_indeces.extend([(idx, word) for idx in indeces])

    sorted_indeces = sorted(word_indeces, key=lambda x: x[0])
    return " ".join([index[1] for index in sorted_indeces])


def download_metadata(lang_code: str, max_articles: int, output_dir: Path):
    """Download article metadata from OpenAlex API."""
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"language:{lang_code},type:article",
        "select": "abstract_inverted_index,primary_location,title,doi,publication_date",
        "mailto": "example@email.com",
        "per-page": 100,
        "cursor": "*",
    }

    session = requests.Session()
    article_data = []
    total_articles = 0
    next_cursor = 0

    with tqdm(
        total=max_articles,
        desc=f"Collecting {lang_code} articles",
    ) as pbar:
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

                    abstract = reconstruct_abstract(result["abstract_inverted_index"])

                    article_data.append(
                        {
                            "title": result["title"],
                            "abstract": abstract,
                            "pdf_url": pdf_url,
                            "doi": result["doi"],
                            "publication_date": result["publication_date"],
                        }
                    )

                    pbar.update(1)
                    total_articles += 1
                    if total_articles >= max_articles:
                        break

            except Exception as e:
                print(f"Error downloading article info: {str(e)}")

            params["cursor"] = next_cursor
            time.sleep(random.randint(4, 6))

    df = pd.DataFrame(article_data)
    metadata_path = output_dir / f"{lang_code}_article_data.csv"
    df.to_csv(metadata_path, index=False, encoding="utf-8")
    return df
