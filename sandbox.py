import requests
from typing import Optional, Dict


# class WikidataGlossary:
#     def __init__(self):
#         self.endpoint_url = "https://query.wikidata.org/sparql"
#         self.headers = {
#             "User-Agent": "ScientificTranslationBot/1.0 (your_email@example.com)"
#         }

#     def get_translation(
#         self, term: str, target_lang_code: str = "vi"
#     ) -> Optional[Dict[str, str]]:
#         """
#         Searches Wikidata for an English label and tries to find the target language label.
#         """
#         # SPARQL query: Find item with English label 'term', get its target language label and description
#         query = f"""
#         SELECT ?item ?itemLabel ?itemDescription ?targetLabel WHERE {{
#             SERVICE wikibase:mwapi {{
#                 bd:serviceParam wikibase:endpoint "www.wikidata.org";
#                     wikibase:api "EntitySearch";
#                     mwapi:search "{term}";
#                     mwapi:language "en".
#                 ?item wikibase:apiOutputItem mwapi:item.
#             }}

#             OPTIONAL {{
#                 ?item rdfs:label ?targetLabel.
#                 FILTER(LANG(?targetLabel) = "{target_lang_code}")
#             }}

#           SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{target_lang_code},en". }}
#         }}
#         LIMIT 1
#         """

#         try:
#             r = requests.get(
#                 self.endpoint_url,
#                 params={"format": "json", "query": query},
#                 headers=self.headers,
#             )
#             data = r.json()
#             print(data)

#             results = data.get("results", {}).get("bindings", [])
#             if not results:
#                 return None

#             result = results[0]

#             # If we found a specific target label (e.g., Vietnamese)
#             if "targetLabel" in result:
#                 return {
#                     "translation": result["targetLabel"]["value"],
#                     "definition": result.get("itemDescription", {}).get(
#                         "value", "No definition found."
#                     ),
#                 }

#             return None

#         except Exception as e:
#             print(f"Wikidata lookup failed for {term}: {e}")
#             return None


# g = WikidataGlossary()

# t = g.get_translation("Mitochondria", "vi")
# print(t)

query = "gravitational lensing"
target_lang_code = "vi"
params = {
    "action": "wbsearchentities",
    "format": "json",
    "language": "en",
    "search": query,
    "limit": 1,
}

headers = {"User-Agent": "ScientificTranslationBot/1.0 (your_email@example.com)"}

try:
    r = requests.get(
        "https://www.wikidata.org/w/api.php", params=params, headers=headers
    )
    r.raise_for_status()

    data = r.json()
    sr = data.get("search", [])

    if sr:
        fr = sr[0]
        entity_id = fr["id"]

    entity_params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "props": "labels|descriptions",
    }

    er = requests.get(
        "https://www.wikidata.org/w/api.php", params=entity_params, headers=headers
    )
    er.raise_for_status()
    ed = er.json()
    entity = ed.get("entities", {}).get(entity_id, {})
    labels = entity.get("labels", {})
    descriptions = entity.get("descriptions", {})

    if target_lang_code in labels:
        translation = labels[target_lang_code]["value"]
        description = descriptions.get(target_lang_code, {}).get(
            "value", descriptions.get("en", {}).get("value", "No definition found.")
        )

        print(
            {
                "translation": translation,
                "definition": description,
                "entity_id": entity_id,
            }
        )
    else:
        print(f"No {target_lang_code} translation available")
        print(f"Available languages: {', '.join(list(labels.keys())[:10])}...")

except Exception as e:
    print(e)
