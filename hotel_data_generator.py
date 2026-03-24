import random
from faker import Faker
from sqlalchemy import create_engine, text

fake = Faker()

DATABASE_URL = "mysql+pymysql://root:Ravi%401234@localhost:3306/ai_universal_platform"
engine = create_engine(DATABASE_URL)

cities = ["Goa", "Chennai", "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune"]
airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa"]
categories = ["Travel", "Electronics", "Accessories", "Gadgets"]

with engine.connect() as conn:

    # ===============================
    # ✈ Generate Flights (300)
    # ===============================
    for _ in range(300):
        conn.execute(text("""
            INSERT INTO Flights
            (FlightName, Source, Destination, BasePrice, DepartureTime)
            VALUES (:name, :src, :dest, :price, :time)
        """), {
            "name": random.choice(airlines) + " " + str(random.randint(100, 999)),
            "src": "Chennai",
            "dest": random.choice(cities),
            "price": random.randint(2500, 9000),
            "time": fake.time(pattern="%I:%M %p")
        })

    # ===============================
    # 🛒 Generate Products (600)
    # ===============================
    for _ in range(600):
        conn.execute(text("""
            INSERT INTO Products
            (ProductName, Category, BasePrice, Rating)
            VALUES (:name, :cat, :price, :rating)
        """), {
            "name": fake.word().title() + " Product",
            "cat": random.choice(categories),
            "price": random.randint(200, 7000),
            "rating": round(random.uniform(3.0, 4.9), 1)
        })

    # ===============================
    # 👤 Generate UserBehavior (1500) ⭐ VERY IMPORTANT
    # ===============================
    domains = ["Hotel", "Flight", "Product"]
    actions = ["View", "Book", "Search", "Cancel"]
    times = ["Morning", "Afternoon", "Evening", "Night"]

    for _ in range(1500):
        conn.execute(text("""
            INSERT INTO UserBehavior
            (UserId, Domain, ItemId, ActionType, PriceRange, TimeOfDay)
            VALUES (:uid, :domain, :item, :action, :price, :time)
        """), {
            "uid": random.randint(1, 5),
            "domain": random.choice(domains),
            "item": random.randint(1, 200),
            "action": random.choice(actions),
            "price": random.randint(500, 8000),
            "time": random.choice(times)
        })

    conn.commit()

print("✅ Bulk data generated successfully!")
