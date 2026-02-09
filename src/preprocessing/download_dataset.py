import requests
from xml.etree import ElementTree as ET
import json
import os
import time

EMAIL = "your_real_email@example.com"  # REQUIRED by NCBI

SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

search_params = {
    "db": "pubmed",
    "term": "clinical trial diabetes",
    "retmax": 200,
    "retmode": "json",
    "email": EMAIL
}

search_response = requests.get(SEARCH_URL, params=search_params)
search_response.raise_for_status()

pmids = search_response.json()["esearchresult"]["idlist"]
print(f"PMIDs fetched: {len(pmids)}")


def fetch_abstracts(pmids):
    ids = ",".join(pmids)

    fetch_params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml",
        "email": EMAIL
    }

    time.sleep(0.4)  # NCBI rate limit

    response = requests.get(FETCH_URL, params=fetch_params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")

        abstract = " ".join(
            t.text for t in article.findall(".//AbstractText") if t.text
        )

        if not abstract:
            continue

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract
        })

    return articles


articles = fetch_abstracts(pmids)
print(f"Articles fetched: {len(articles)}")

os.makedirs("data/raw/pubmed", exist_ok=True)

with open("data/raw/pubmed/pubmed_articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

print("PubMed data saved successfully.")