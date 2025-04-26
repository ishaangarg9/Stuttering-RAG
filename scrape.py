import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai.client import MistralClient
from langchain_mistralai import ChatMistralAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI

load_dotenv()  # Load environment variables from .env

mistral_key = os.getenv("MISTRAL_API_KEY")

# ✅ Mistral LLM Client
llm = ChatMistralAI(
    mistral_api_key=mistral_key,
    model="mistral-medium"
)

# # ✅ Scraping PubMed
# def scrape_pubmed(query="stuttering"):
#     url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query}"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     results = []

#     for article in soup.find_all("article", class_="full-docsum"):
#         title = article.find("a", class_="docsum-title")
#         abstract = article.find("div", class_="full-view-snippet")
#         results.append({
#             "title": title.text.strip() if title else "N/A",
#             "link": "https://pubmed.ncbi.nlm.nih.gov" + title["href"] if title else "N/A",
#             "abstract": abstract.text.strip() if abstract else "N/A"
#         })
#     return results

# # ✅ Scraping Success Stories
# def scrape_success_stories():
#     url = "https://www.stutteringhelp.org/category/tags/success-stories"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     stories = []
#     for node in soup.select("div.view-content .views-row"):
#         title = node.find("h3")
#         summary = node.find("div", class_="field-content")
#         link = node.find("a")
#         stories.append({
#             "title": title.text.strip() if title else "",
#             "abstract": summary.text.strip() if summary else "",
#             "link": "https://www.stutteringhelp.org" + link['href'] if link else ""
#         })
#     return stories


# # ✅ Preprocess and Embed
# def preprocess_and_embed(documents, output_path="rag_stutter_index"):
#     texts = [doc["title"] + "\n" + doc["abstract"] for doc in documents]
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.create_documents(texts)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(chunks, embeddings)
#     vectordb.save_local(output_path)


# ✅ Query RAG with Mistral
def query_rag(question, index_path="rag_expanded_index"):
    vectordb = FAISS.load_local(
        index_path,
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )
    return qa_chain.run(question)


# ✅ Main Execution
if __name__ == "__main__":
    # pubmed_docs = scrape_pubmed("stuttering treatment")
    # stories = scrape_success_stories()
    # all_docs = pubmed_docs + stories

    # preprocess_and_embed(all_docs)
    print(query_rag("Who are some famous people who stutter?"))