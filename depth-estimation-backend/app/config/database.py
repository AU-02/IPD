from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb+srv://anoukudumalagala:XjuPC6JLuTG0hpNr@cluster0.peynz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB Atlas
client = AsyncIOMotorClient(MONGO_URI)
database = client["D3MSD"]  
users_collection = database["users"]  # Users collection
