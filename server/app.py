"""
server/app.py — FastAPI app for Vyapar-RL OpenEnv environment.
"""
import sys
import os
import uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from models import GSTAction, GSTObservation
from server.environment import GSTEnvironment

app = create_fastapi_app(GSTEnvironment, GSTAction, GSTObservation)

@app.get("/health")
def health_status():
    return {"status": "ok"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
