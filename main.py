import uvicorn
from api.server import app


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1, # Set to 1 for development, increase for production
        reload=True # Only for development, remove in production
    )
