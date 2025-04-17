import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from newspaper import Article
from serpapi.google_search import GoogleSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# 1. Google Search via SerpAPI
def google_search(query, num_results=10):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    links = [res['link'] for res in results.get("organic_results", []) if 'link' in res]
    return links

# 2. Scrape article content using newspaper3k
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return None, None

# 3. Preprocess & store into FAISS
def append_to_faiss(docs, db_path="rag_expanded_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = []
    for doc in docs:
        if not doc["content"]:
            continue
        chunks = splitter.split_text(doc["content"])
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
    vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    vectordb.add_documents(documents)
    vectordb.save_local(db_path)

# 4. Main function to tie it all together
if __name__ == "__main__":
    queries = ["stuttering", "stuttering exercises", "famous people who stutter", "stuttering therapy"]
    all_links = []
    for q in queries:
        all_links.extend(google_search(q, num_results=20))

    scraped_docs = []
    for url in set(all_links):
        title, text = scrape_article(url)
        if text:
            scraped_docs.append({
                "title": title,
                "source": "Google Search",
                "content": text,
                "url": url
            })

    append_to_faiss(scraped_docs)
    print(f"✅ {len(scraped_docs)} pages scraped and added to FAISS.")