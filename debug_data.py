import pandas as pd
from sqlalchemy import create_engine

# Database credentials
db_username = 'root'
db_password = 'root'
db_host = 'localhost'
db_name = 'steam_data'

# Connect and read data
try:
    print("Connecting to database...")
    engine = create_engine(f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}/{db_name}')
    query = 'SELECT * FROM steamout LIMIT 50'
    df = pd.read_sql(query, engine)

    print(f"\nSuccessfully loaded {len(df)} rows.")
    
    print("\n--- COLUMN DTYPES ---")
    print(df.dtypes)
    
    print("\n--- RAW PRICE SAMPLES ---")
    print(df['Price'].head(20).tolist())
    
    print("\n--- RAW YEAR SAMPLES ---")
    print(df['Year'].head(20).tolist())
    
    print("\n--- RAW DISCOUNT SAMPLES ---")
    print(df['Discount'].head(20).tolist())

    print("\n--- RAW REVIEWNUM SAMPLES ---")
    print(df['ReviewNum'].head(20).tolist())

except Exception as e:
    print(f"Error: {e}")
