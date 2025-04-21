import os
import base64  # 追加
from typing import List
from sqlalchemy.orm import Session
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings
from db import FamilyRelationship
from models import Memorial  # 追加

def get_family_data(family_id: int, user_id: int, db: Session) -> List[dict]:
    records = db.query(FamilyRelationship).filter_by(
        family_id=family_id, user_id=user_id
    ).all()
    return [{"relation": r.relation, "calling_name": r.calling_name} for r in records]

def generate_response(question: str, family_id: int, user_id: int, db: Session) -> dict:
    embedding_model = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-large",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version="2023-05-15"
    )

    db_vector = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    retriever = db_vector.as_retriever(search_kwargs={"k": 1})
    docs_and_scores = retriever.vectorstore.similarity_search_with_score(question, k=1)

    print("▼ 検索でヒットした日記とスコア（スコアが小さいほど類似）:", flush=True)

    context = ""
    memorials_id = None
    photo_path = None

    for i, (doc, score) in enumerate(docs_and_scores):
        print(f"\n--- Top {i+1} ---", flush=True)
        print(f"スコア: {score:.4f}", flush=True)

        memorials_id = doc.metadata.get("memorials_id")
        print(f"memorials_id from metadata: {memorials_id}", flush=True)

        memorial = db.query(Memorial).filter_by(memorials_id=memorials_id).first()
        if memorial:
            print(f"日記ID: {memorial.memorials_id}", flush=True)
            print(f"日時: {memorial.timestamp}", flush=True)
            print(f"場所: {memorial.location}", flush=True)
            print(f"本文: {memorial.diary_text}", flush=True)

            context = memorial.diary_text

            # BLOB画像対応（Base64エンコード）
            if isinstance(memorial.photo, (bytes, bytearray)):
                encoded = base64.b64encode(memorial.photo).decode("utf-8")
                photo_path = f"data:image/jpeg;base64,{encoded}"
            elif isinstance(memorial.photo, str) and memorial.photo.strip():
                photo_path = memorial.photo
            else:
                print("⚠ photo の形式が不明。スキップ", flush=True)
                photo_path = None
        else:
            context = doc.page_content

    family_data = get_family_data(family_id, user_id, db)
    default_info = [{"relation": f"関係{i+1}", "calling_name": f"名前{i+1}"} for i in range(10)]
    while len(family_data) < 10:
        family_data.append(default_info[len(family_data)])

    info = {}
    for i in range(10):
        info[f"calling_name{i+1}"] = family_data[i]["calling_name"]
        info[f"relation_name{i+1}"] = family_data[i]["relation"]
        info[f"birthday_name{i+1}"] = f"2000-01-{i+1:02d}"

    family_lines = "\n".join([
        f"{info[f'relation_name{i+1}']}のことを{info[f'calling_name{i+1}']}（誕生日: {info[f'birthday_name{i+1}']}）"
        for i in range(10)
    ])

    prompt = PromptTemplate.from_template(
        """
        あなたは天国にいて子供を見守っています。
        あなたに語りかけるあなたの子供のコメント{question}に対して、
        家族の素晴らしい思い出を一緒に回想してコメントを作成してください。
        全てのコメントは親子間のフランクな口調で、優しく語りかけるような表現にしてください。
        あなたが残した日記のコメントは下記3件あります。{context}
        日記の内容を含めて回想してください。
        ただし、子供のコメントに対して、ロケーション・場所・年月日が大きく異なる日記の内容は回想に含める必要はありません。
        以下にあなたの家族構成と呼称を最大10件記載します。文脈に合えば使ってください。
        {family_lines}
        200文字以内にしてください。
        """
    ).partial(**info)

    final_prompt = prompt.format(question=question, context=context, family_lines=family_lines)

    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment_name="gpt-4o-mini",
        temperature=0.7
    )

    return {
        "response": llm.invoke(final_prompt).content,
        "photo": photo_path or "画像なし"
    }