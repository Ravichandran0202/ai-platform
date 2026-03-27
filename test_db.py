# test_db.py
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres.wrsasiaelfpqiuiptsrc:Ravichandran0202@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres?sslmode=require"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
