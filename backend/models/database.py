from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, inspect, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
print(f"Using database URL: {DATABASE_URL}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


from sqlalchemy import Column, Integer, String


class Videos(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    detected_objects = Column(Text, nullable=True)
    frame_timestamps = Column(String, nullable=True)
    embeddings = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))


class Audios(Base):
    __tablename__ = "audios"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    transcriptions = Column(Text)
    timestamps = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.now())


# Create tables
def create_tables():
    print("Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        # Check created tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Created tables: {tables}")
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        raise


# get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
