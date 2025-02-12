from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from app.routes import auth, home, depth  # Import depth route

app = FastAPI()

# Enable CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(home.router, prefix="", tags=["Home"])
app.include_router(depth.router, prefix="/depth", tags=["Depth"])  # Add depth route

# Serve static files for the 'uploads' directory
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}
