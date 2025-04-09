from db import Base
from sqlalchemy import Column, Integer, Float, String, DateTime

class Memorial(Base):
    __tablename__ = "memorials"

    memorials_id = Column(Integer, primary_key=True, index=True)
    family_id = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
    location = Column(String)
    diary_text = Column(String)
    photo = Column(String)
    timestamp = Column(DateTime)