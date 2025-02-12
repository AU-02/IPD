from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from app.config.database import users_collection
from app.core.security import hash_password, verify_password, create_access_token
from bson import ObjectId

router = APIRouter()

# Pydantic Models
class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    terms_accepted: bool

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
async def register(user: UserCreate):
    """Handles user registration."""
    existing_user = await users_collection.find_one({"email": user.email})

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if not user.terms_accepted:
        raise HTTPException(status_code=400, detail="You must accept the terms and conditions")

    hashed_password = hash_password(user.password)
    
    user_data = {
        "full_name": user.full_name,
        "email": user.email,
        "password": hashed_password,
        "terms_accepted": user.terms_accepted
    }

    new_user = await users_collection.insert_one(user_data)
    
    return {"message": "User registered successfully", "user_id": str(new_user.inserted_id)}

@router.post("/login")
async def login(user: UserLogin):
    """Handles user login."""
    existing_user = await users_collection.find_one({"email": user.email})

    if not existing_user or not verify_password(user.password, existing_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(data={"sub": str(existing_user["_id"])})

    return {"access_token": access_token, "token_type": "bearer"}
