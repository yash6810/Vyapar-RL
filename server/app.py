"""
server/app.py — FastAPI app using OpenEnv's create_app helper.
"""
import uvicorn
from openenv.core.env_server import create_app

from models import GSTAction, GSTObservation
from server.environment import GSTEnvironment

app = create_app(GSTEnvironment, GSTAction, GSTObservation,
                 env_name="Vyapar-RL")

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
