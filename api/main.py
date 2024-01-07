from fastapi import FastAPI
import uvicorn
from api.api.router import api_router

app = FastAPI()

app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True, log_level="info")
