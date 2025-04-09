from sqlalchemy import text
from db import engine

def column_exists(conn, table, column):
    result = conn.execute(text(
        f"SHOW COLUMNS FROM {table} LIKE :column"
    ), {"column": column}).fetchone()
    return result is not None

def add_timestamp_column_and_copy():
    with engine.connect() as conn:
        if not column_exists(conn, "memorials", "timestamp"):
            conn.execute(text(
                "ALTER TABLE memorials ADD COLUMN timestamp DATETIME AFTER family_id;"
            ))
            print("✅ timestampカラムを追加")
        else:
            print("ℹ️ timestampカラムはすでに存在します")

        conn.execute(text(
            "UPDATE memorials SET timestamp = created_at WHERE created_at IS NOT NULL;"
        ))
        print("✅ created_atのデータをtimestampにコピー")

        conn.commit()

if __name__ == "__main__":
    add_timestamp_column_and_copy()

