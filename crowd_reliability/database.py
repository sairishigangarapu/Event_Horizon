import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from crowd_reliability.config import settings
from crowd_reliability.models.schema import Base

# Ensure data directory exists
data_dir = os.path.dirname(settings.database_url.replace("sqlite:///", ""))
if data_dir and not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
