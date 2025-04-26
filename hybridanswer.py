import requests
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import Together
from dotenv import load_dotenv

load_dotenv()

# Load FAISS Database
db_path = "rag_expanded_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# Load Mistral model via Together AI
mistral_llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.3,
    max_tokens=512,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# Prompt Template for Hybrid RAG
prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
You are an expert assistant. You are kind and explain everything in detail. Use the following context and previous conversation history to answer the question.
If the context is not enough, answer based on your knowledge.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
)

chat_history = []

def hybrid_answer(user_query):
    results = vectordb.similarity_search_with_score(user_query, k=4)
    context = "\n\n".join(doc.page_content for doc, score in results if score > 0.5)

    history_text = ""
    for user, bot in chat_history:
        history_text += f"User: {user}\nAssistant: {bot}\n"

    final_prompt = prompt_template.format(history=history_text, context=context, question=user_query)

    response = mistral_llm.invoke(final_prompt)

    chat_history.append((user_query, response))

    return response

if __name__ == "__main__":
    while True:
        user_question = input("You: ")
        if user_question.lower() in ("exit", "quit"):
            break
        answer = hybrid_answer(user_question)
        print("\nBot:", answer)
