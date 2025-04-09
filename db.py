import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ssl_cert_path = os.path.join(BASE_DIR, "azure-ca.pem")

DB_URL = (
    f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:3306/{settings.DB_NAME}"
    f"?ssl_ca={ssl_cert_path}&ssl_verify_cert=true"
)

engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class FamilyRelationship(Base):
    __tablename__ = "family_relationships"

    relation_id = Column(Integer, primary_key=True, index=True)
    family_id = Column(Integer, index=True)
    user_id = Column(Integer, index=True)
    relation = Column(String)
    calling_name = Column(String)

