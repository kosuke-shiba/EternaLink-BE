from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from config import settings

def inspect_vector_store():
    # 同じ埋め込みモデル（読み込みに必要）
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-large",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version="2023-05-15"
    )

    # 保存されたベクトルストアの読み込み
    faiss_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

    # 中身（ドキュメント）を全部取り出す
    documents = faiss_db.docstore._dict.values()

    print("📦 ベクトルストアの中身（上位5件まで表示）:")
    for i, doc in enumerate(documents):
        print(f"\n--- Doc {i+1} ---")
        print(f"ID: {doc.metadata.get('id')}")
        print(doc.page_content)

        if i >= 4:  # 上位5件まで表示
            break

if __name__ == "__main__":
    inspect_vector_store()
