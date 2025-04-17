import requests
import os
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime
import wikipedia
from dotenv import load_dotenv

load_dotenv()

# 1. Scrape Wikipedia
def get_wikipedia_article(topic):
    try:
        # Get top search result that matches the topic
        title = wikipedia.search(topic)[0]
        page = wikipedia.page(title)
        return [{
            "title": page.title,
            "source": "Wikipedia",
            "content": page.content,
            "url": page.url
        }]
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"DisambiguationError: {e} — picking first option.")
        page = wikipedia.page(e.options[0])
        return [{
            "title": page.title,
            "source": "Wikipedia",
            "content": page.content,
            "url": page.url
        }]
    except Exception as ex:
        print(f"❌ Failed to fetch Wikipedia page for '{topic}': {ex}")
        return []

# 2. Scrape Reddit using Pushshift API
def get_reddit_posts(keyword, limit=5):
    url = f"https://api.pushshift.io/reddit/search/submission/?q={keyword}&subreddit=stuttering&size={limit}"
    res = requests.get(url)
    posts = res.json().get("data", [])
    return [{
        "title": p.get("title", ""),
        "source": "Reddit",
        "content": p.get("selftext", ""),
        "url": f"https://www.reddit.com{p.get('permalink', '')}"
    } for p in posts if p.get("selftext")]

# 3. Scrape Semantic Scholar
def get_semantic_scholar_papers(query="stuttering", limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,abstract,url"}
    headers = {"Accept": "application/json"}
    res = requests.get(url, params=params, headers=headers).json()
    return [{
        "title": p.get("title", ""),
        "source": "SemanticScholar",
        "content": p.get("abstract", ""),
        "url": p.get("url", "")
    } for p in res.get("data", []) if p.get("abstract")]

# 4. Chunk, Embed and Save to FAISS
def process_and_store(docs, db_path="rag_expanded_index"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = []
    for doc in docs:
        if not doc["content"]:
            continue
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc["source"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "date": datetime.now().isoformat()
                }
            ))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(db_path)

# 5. Execute the pipeline
if __name__ == "__main__":
    wiki_docs = get_wikipedia_article("Stuttering")
    reddit_docs = get_reddit_posts("stuttering")
    semantic_docs = get_semantic_scholar_papers("stuttering treatment")

    all_docs = wiki_docs + reddit_docs + semantic_docs
    process_and_store(all_docs)

    print("✅ Knowledge base updated and stored in FAISS.")