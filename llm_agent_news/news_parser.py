import feedparser

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Или GigaChat (если через API)

import numpy as np
import os



url = "https://www.vedomosti.ru/rss/news.xml"
feed = feedparser.parse(url)

news_items = [
    entry.get("title", "") + ". " + 
    (entry.get("summary") or entry.get("description") or "")
    for entry in feed.entries
]


keywords = ["банкрот","риск"]

pre_filtered_news = [
    n for n in news_items
    if any(k in n.lower() for k in keywords)
]



joined = "\n\n".join(pre_filtered_news[:5])
prompt = f"""
Проанализируй следующие новости и найди те, которые могут свидетельствовать о банкротстве, санкциях, уголовных делах или репутационных рисках компаний:

{joined}

Отметь, какие из них вызывают подозрение, и почему.
"""




embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vector_db = FAISS.from_texts(pre_filtered_news, embedding=embeddings)
retriever = vector_db.as_retriever()
retriever.get_relevant_documents("Какие компании под риском?")
llm = OpenAI(
    temperature=0.3,
    openai_api_key= os.getenv("openai_token")

)


qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
query = "Есть ли среди компаний банкротства или признаки риска?"

result = qa_chain.run(prompt)
print(result)


# Векторизация ответа LLM
query_embedding = embeddings.embed_query(result)

query_vector = np.array([query_embedding]).astype("float32")
D, I = vector_db.index.search(query_vector, k=1)



# I[0][0] — индекс самой релевантной новости
best_match_index = I[0][0]
best_news = pre_filtered_news[best_match_index]

print("🔍 📰 Самая близкая по смыслу новость:")
print(best_news)



