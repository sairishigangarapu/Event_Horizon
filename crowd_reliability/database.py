import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from crowd_reliability.config import settings
from crowd_reliability.models.schema import Base

def ensure_data_dir():
    """Ensures parent directory for sqlite database exists."""
    data_dir = os.path.dirname(settings.database_url.replace("sqlite:///", ""))
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in settings.database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    ensure_data_dir()
    Base.metadata.create_all(bind=engine)

def get_db():
    ensure_data_dir()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

