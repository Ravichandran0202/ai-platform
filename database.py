from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
ENV = os.getenv("ENV", "local")

if ENV == "production":
    # Supabase (LIVE)
    DATABASE_URL = "postgresql://postgres:RaviTesting%40123@db.wrsasiaelfpqiuiptsrc.supabase.co:5432/postgres"
else:
    # Local MySQL
    DATABASE_URL = "mysql+pymysql://root:Ravi%401234@localhost:3306/ai_universal_platform"

engine = create_engine(DATABASE_URL)
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
