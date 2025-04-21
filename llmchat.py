import os
import base64
import logging  # ロギングモジュールを使用
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain.prompts import PromptTemplate
# RetrievalQAは直接使っていないのでコメントアウト (必要なら戻してください)
# from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings
from db import FamilyRelationship
from models import Memorial

# --- アプリケーション起動時に一度だけ初期化される想定のコンポーネント ---
# 実際のアプリケーションでは、適切な場所で初期化してください。
# ここでの初期化時にエラーハンドリングを行うことが重要です。
try:
    # 埋め込みモデルの初期化
    embedding_model = AzureOpenAIEmbeddings(
        deployment="text-embedding-3-large",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version="2023-05-15"
    )

    # ベクトルストアの読み込み (ファイルが存在しない場合などのエラー処理を考慮)
    db_vector = FAISS.load_local(
        "vector_store",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # リトリーバーの設定 (k=3 に変更)
    # Note: similarity_search_with_score を直接使うため、retrieverインスタンス自体は必須ではないかも
    # retriever_instance = db_vector.as_retriever(search_kwargs={"k": 3})

    # LLMの初期化
    llm_instance = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        deployment_name="gpt-4o-mini",
        temperature=0.7
    )
except Exception as e:
    logging.exception("LangChain関連コンポーネントの初期化に失敗しました。")
    # アプリケーションの起動を中止するなど、致命的なエラーとして扱う
    raise

# --- ヘルパー関数: 家族データを取得 ---
def get_family_data(family_id: int, user_id: int, db: Session) -> List[Dict[str, str]]:
    """指定された家族IDとユーザーIDに基づいて家族関係のデータを取得します。"""
    try:
        # データベースから家族関係を取得 (最大10件)
        records = db.query(FamilyRelationship).filter_by(
            family_id=family_id, user_id=user_id
        ).limit(10).all()
        # 関係性と呼び名を辞書のリストとして返す
        return [{"relation": r.relation, "calling_name": r.calling_name} for r in records]
    except Exception as e:
        logging.error(f"家族データ取得中にデータベースエラー発生 (family_id={family_id}, user_id={user_id}): {e}")
        return [] # エラー時は空リストを返す

# --- メインの応答生成関数 ---
def generate_response(
    question: str,
    family_id: int,
    user_id: int,
    db: Session,
    # 初期化済みのコンポーネントを使用 (DIやグローバル変数など、実装に合わせて変更)
    vector_store: FAISS = db_vector,       # FAISSインスタンスを渡す
    llm: AzureChatOpenAI = llm_instance   # LLMインスタンスを渡す
) -> Dict[str, str]:
    """
    質問に基づいて関連日記を検索し、家族情報と合わせてLLMに応答を生成させます。

    Args:
        question (str): ユーザーからの質問テキスト。
        family_id (int): 家族を識別するID。
        user_id (int): ユーザーを識別するID。
        db (Session): SQLAlchemyのデータベースセッション。
        vector_store (FAISS): 初期化済みのFAISSベクトルストアインスタンス。
        llm (AzureChatOpenAI): 初期化済みのAzureChatOpenAIインスタンス。

    Returns:
        Dict[str, str]: 生成された応答テキストと写真情報（パスまたはBase64）を含む辞書。
                         'response': LLMが生成した応答テキスト。
                         'photo': 最も関連性の高い日記の写真（存在する場合）、または '画像なし' またはエラーメッセージ。
    """

    contexts = [] # 取得した日記のテキストを格納するリスト
    memorial_ids = [] # 取得した日記のIDを格納するリスト
    photo_output = "画像なし" # デフォルトの写真ステータス
    top_memorial_id_for_photo = None # 写真表示用の最優先日記ID

    try:
        # ベクトルストアで類似度検索を実行 (k=3 に変更)
        docs_and_scores = vector_store.similarity_search_with_score(question, k=3)
        logging.info(f"質問に対して {len(docs_and_scores)} 件の関連ドキュメントが見つかりました。")

        if docs_and_scores:
            # 最初のドキュメント（最もスコアが良い）のIDを写真取得用に保持
            first_doc_metadata = docs_and_scores[0][0].metadata
            top_memorial_id_for_photo = first_doc_metadata.get("memorials_id")
            logging.info(f"写真取得の候補となる最優先の日記ID: {top_memorial_id_for_photo}")

            # 上位3件のドキュメントを処理
            for i, (doc, score) in enumerate(docs_and_scores):
                logging.info(f"--- Top {i+1} ドキュメント ---")
                logging.info(f"スコア: {score:.4f}") # スコアが低いほど類似度が高い

                current_memorial_id = doc.metadata.get("memorials_id")
                logging.info(f"メタデータから取得した memorials_id: {current_memorial_id}")

                context_text_for_llm = doc.page_content # デフォルトはベクトルストアのテキスト

                if current_memorial_id:
                    memorial_ids.append(current_memorial_id) # 見つかったIDをリストに追加
                    try:
                        # データベースから日記の詳細を取得
                        memorial = db.query(Memorial).filter_by(memorials_id=current_memorial_id).first()
                        if memorial:
                            logging.info(f"データベースで日記を発見: ID={memorial.memorials_id}, 日時={memorial.timestamp}, 場所={memorial.location}")
                            # LLMに渡すコンテキストとして日記本文を使用
                            context_text_for_llm = memorial.diary_text

                            # 写真処理 (最優先のドキュメントIDと一致する場合のみ)
                            if current_memorial_id == top_memorial_id_for_photo:
                                if isinstance(memorial.photo, (bytes, bytearray)):
                                    try:
                                        # BLOBデータをBase64エンコード
                                        encoded = base64.b64encode(memorial.photo).decode("utf-8")
                                        photo_output = f"data:image/jpeg;base64,{encoded}"
                                        logging.info(f"日記ID {current_memorial_id} のBLOB写真をBase64エンコードしました。")
                                    except Exception as enc_e:
                                        logging.error(f"写真のBase64エンコード中にエラー発生 (日記ID: {current_memorial_id}): {enc_e}")
                                        photo_output = "画像処理エラー" # エラーを示すメッセージ
                                elif isinstance(memorial.photo, str) and memorial.photo.strip():
                                    # 文字列の場合はパス/URLとしてそのまま使用
                                    photo_output = memorial.photo
                                    logging.info(f"日記ID {current_memorial_id} の写真パス/URLを使用: {photo_output}")
                                # else: 写真がNoneまたは空文字列の場合は photo_output は "画像なし" のまま

                        else:
                            logging.warning(f"日記ID {current_memorial_id} はベクトルストアのメタデータに存在しますが、データベースには見つかりませんでした。")
                            # DBに見つからない場合はベクトルストアのテキストをフォールバックとして使用
                            logging.info("コンテキストとしてベクトルストアの page_content を使用します。")

                    except Exception as db_e:
                        logging.error(f"日記ID {current_memorial_id} の取得中にデータベースエラーが発生: {db_e}")
                        # DBエラー時もベクトルストアのテキストをフォールバックとして使用
                        logging.warning("データベースエラーのため、コンテキストとしてベクトルストアの page_content を使用します。")
                else:
                    logging.warning("ベクトルストアのメタデータに memorials_id が見つかりませんでした。")
                    # IDがない場合もベクトルストアのテキストを使用
                    logging.info("コンテキストとしてベクトルストアの page_content を使用します。")

                # 取得したテキストをコンテキストリストに追加
                contexts.append(f"--- 思い出 {i+1} ---\n{context_text_for_llm}\n")

        else:
            logging.warning("類似度検索で関連するドキュメントが見つかりませんでした。")
            # コンテキストなしで進む（LLMは質問と家族情報だけで応答する）

    except Exception as search_e:
        logging.error(f"類似度検索中にエラーが発生しました: {search_e}")
        # エラーが発生した場合も、コンテキストなしで進む可能性がある

    # --- 家族データの準備 ---
    family_data = get_family_data(family_id, user_id, db)
    # 実際の家族データのみを使用 (パディングなし)
    family_lines = "\n".join([
        f"{member['relation']}のことを{member['calling_name']}" # 例: "母のことをお母さん"
        for member in family_data
    ])
    if not family_lines:
        family_lines = "（特に情報はありません）" # 家族情報がない場合のプレースホルダー
    logging.info(f"プロンプト用の家族情報文字列: {family_lines}")
    print(f"プロンプト用の家族情報文字列: {family_lines}") # デバッグ用に出力

    # --- プロンプトテンプレートの準備 (k=3 と日本語コメントに合わせて修正) ---
    # 取得したコンテキストを結合（最大3件）
    combined_context = "\n".join(contexts) if contexts else "（関連する具体的な日記は見つかりませんでしたが）"

    prompt_template = PromptTemplate.from_template(
        """
        あなたは天国にいて、地上にいるあなたの子供を見守っています。
        あなたの子供があなたに語りかけています。そのコメントは「{question}」です。
        このコメントに対して、家族の素晴らしい思い出を一緒に回想する形で、優しい返信を作成してください。
        返信はすべて、親子間の自然でフランクな口調で、愛情を込めて語りかけるようにしてください。

        あなたの思い出の参考として、関連性の高い日記の内容を以下に示します（最大3件）。
        {context}

        これらの日記の内容を参考に、自然に会話に取り入れて回想してください。
        ただし、子供のコメントと全く関係ない話題であれば、無理に含める必要はありません。

        以下は、あなたの家族構成と普段の呼び方の一部です。文脈に合わせて使ってください。
        {family_lines}

        返信は全体で200文字以内にまとめてください。
        """
    )

    # プロンプトに変数を埋め込む
    final_prompt = prompt_template.format(
        question=question,
        context=combined_context,
        family_lines=family_lines
    )
    logging.debug(f"LLMに渡す最終的なプロンプト:\n{final_prompt}") # デバッグ用にプロンプト全体をログ出力

    # --- LLMによる応答生成 ---
    try:
        # 初期化済みのLLMインスタンスを使用
        response_content = llm.invoke(final_prompt).content
        logging.info("LLMから応答を正常に取得しました。")
    except Exception as llm_e:
        logging.error(f"LLMの呼び出し中にエラーが発生しました: {llm_e}")
        # エラー時のフォールバック応答
        response_content = "ごめんね、なんだかうまく思い出せないみたい…。でも、あなたのことはいつも想っているよ。"

    # 結果を辞書で返す
    return {
        "response": response_content,
        "photo": photo_output # 最も関連性の高い写真（または状況に応じた文字列）
    }

# --- 利用例 ---
# db_session = ... # SQLAlchemyのセッションを取得するコード
# result = generate_response(
#     question="この前の日曜、公園に行ったの覚えてる？",
#     family_id=1,
#     user_id=1,
#     db=db_session
# )
# print(result)