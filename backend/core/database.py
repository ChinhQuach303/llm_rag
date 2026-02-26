from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.config import DATABASE_URL

# Setup SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, 
    # check_same_thread false is needed for sqlite in fastapi with dependency injection
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory bound to engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

def init_db():
    """Create all tables stored in Base metadata."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency generator for FastAPI routers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
