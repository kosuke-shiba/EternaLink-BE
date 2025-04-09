from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from sqlalchemy import select
from config import settings
from db import SessionLocal
from models import Memorial

def update_vector_store():
    db = SessionLocal()
    documents = []

    try:
        memorials = db.execute(select(Memorial)).scalars().all()

        for row in memorials:
            text = f"""日記ID: {row.memorials_id}
日時: {row.timestamp}
場所: {row.location}
本文: {row.diary_text}"""

            documents.append(Document(
                page_content=text,
                metadata={"memorials_id": row.memorials_id}
            ))

        embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-large",
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version="2023-05-15"
        )

        faiss_db = FAISS.from_documents(documents, embeddings)
        FAISS.save_local(faiss_db, "vector_store")

        print("ベクトルストア更新完了 ✅")
        return "ベクトルストア更新完了 ✅"

    except Exception as e:
        db.rollback()
        print(f"❌ エラー: {e}")
        return f"エラー: {e}"

    finally:
        db.close()

