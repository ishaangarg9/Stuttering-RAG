
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime

# 1. Scrape VeryWellHealth stuttering exercises article
def get_verywell_exercises():
    url = "https://www.verywellhealth.com/speech-exercises-to-reduce-stuttering-5195070"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    paras = soup.find_all("p")
    text = "\n".join(p.get_text() for p in paras)
    return [{
        "title": "Stuttering Exercises - VeryWellHealth",
        "source": "VeryWellHealth",
        "content": text,
        "url": url
    }]

# 2. Scrape StutteringHelp.org therapy and exercise pages
def get_stutteringhelp_therapy():
    base_url = "https://www.stutteringhelp.org"
    urls = [
        "/therapy", 
        "/content/classic-speech-therapy-techniques",
        "/content/speech-exercises"
    ]
    results = []
    for path in urls:
        full_url = base_url + path
        res = requests.get(full_url)
        soup = BeautifulSoup(res.text, "html.parser")
        body = soup.find("div", {"class": "field-item even"})
        if body:
            text = body.get_text(separator="\n").strip()
            results.append({
                "title": f"Therapy Content from {path}",
                "source": "StutteringHelp.org",
                "content": text,
                "url": full_url
            })
    return results

# 3. Pull YouTube transcript
def get_youtube_transcript(video_id, title_hint="YouTube Video"):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item["text"] for item in transcript])
        return [{
            "title": f"{title_hint}",
            "source": "YouTube",
            "content": full_text,
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }]
    except Exception as e:
        print(f"Transcript not available for {video_id}: {e}")
        return []

# 4. Append new content to existing vector DB
def append_to_faiss(docs, db_path="rag_expanded_index"):
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
    vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    vectordb.add_documents(documents)
    vectordb.save_local(db_path)

# 5. Run pipeline and update DB
if __name__ == "__main__":
    verywell_docs = get_verywell_exercises()
    stutteringhelp_docs = get_stutteringhelp_therapy()
    youtube_docs = get_youtube_transcript("Lwz7gkXhG2Y", "Speech Therapy Exercises for Stuttering")

    new_docs = verywell_docs + stutteringhelp_docs + youtube_docs
    append_to_faiss(new_docs)

    print("✅ New knowledge sources added to FAISS database.")
