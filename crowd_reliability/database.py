import os
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session
from crowd_reliability.config import settings
from crowd_reliability.models.schema import Base

def ensure_data_dir():
    """Ensures parent directory for sqlite database exists."""
    url=make_url(settings.database_url)
    if url.get_backend_name()=='sqlite' and url.database and url.database!=':memory:':
        directory=os.path.dirname(url.database)
        if directory:os.makedirs(directory,exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if engine.dialect.name=="sqlite":
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    ensure_data_dir()
    Base.metadata.create_all(bind=engine)
    upgrade_forecast_columns(engine)

def get_db():
    ensure_data_dir()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def upgrade_forecast_columns(bind):
    """Idempotent, additive schema upgrade; preserves all pre-audit forecast rows."""
    if 'forecasts' not in inspect(bind).get_table_names():return []
    existing={c['name'] for c in inspect(bind).get_columns('forecasts')}
    changes={'channel_mask':'JSON','applicability':'FLOAT DEFAULT 1.0','missing_data_mask':'JSON'}
    added=[]
    with bind.begin() as connection:
        for name,sql_type in changes.items():
            if name not in existing:
                connection.execute(text(f'ALTER TABLE forecasts ADD COLUMN {name} {sql_type}'))
                added.append(name)
    return added
