import json
from typing import Dict, List, Set

import requests


WIKIPEDIA_REQUEST_URL = "http://en.wikipedia.org/w/api.php?action=query"


def _get_wikipedia_title(request_token: List[str]) -> Dict[str, str]:
    '''
    Returns Dict with <pageid>: title
    '''
    out_format = "json"
    srprop = "sectiontitle"
    srlimit = 1  # just the most relevant only
    srsearch = " ".join(request_token)

    request_url = f"{WIKIPEDIA_REQUEST_URL}&list=search&format={out_format}&srprop={srprop}&srlimit={srlimit}&srsearch={srsearch}"
    
    results: Dict[int, str] = {}
    try:
        for r in requests.get(request_url).json()["query"]["search"]:
            results[str(r["pageid"])] = r["title"]
    except Exception:
        pass

    return results


def _get_wikipedia_categories(pageids: Set[int]) -> List[str]:
    out_format = "json"
    prop = "categories"
    pageids = "|".join(list(pageids))

    request_url = f"{WIKIPEDIA_REQUEST_URL}&format={out_format}&prop={prop}&pageids={pageids}"
    print(request_url)
    categories: List[str] = []
    try:
        for page_id, r in requests.get(request_url).json()["query"]["pages"].items():
            for category in r["categories"]:
                categories.append(category["title"].replace("Category:", ""))
    except Exception:
        pass

    return categories


def request_wikipedia(top_words: List[str]) -> List[str]:
    print(top_words)
    top_words = [x.replace("_", " ") for x in top_words]  # get rid of _ in phrase
    
    page_ids_titles = _get_wikipedia_title(top_words)
    categories = _get_wikipedia_categories(page_ids_titles.keys())

    return categories


if __name__ == "__main__":
    token = [
        "artes",
        "august",
        "appointment",
        "beaufort",
        "be"
    ]
    page_ids_titles = _get_wikipedia_title(token)  # (["henry", "viii"])
    categories = _get_wikipedia_categories(page_ids_titles.keys())
    
    print(page_ids_titles)
    print()
    print(categories)