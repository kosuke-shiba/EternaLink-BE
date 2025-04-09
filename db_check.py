from sqlalchemy import text
from db import engine

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ DB接続成功！結果:", result.scalar())
    except Exception as e:
        print("❌ DB接続エラー:", e)

if __name__ == "__main__":
    test_connection()