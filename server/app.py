"""
server/app.py — FastAPI app using OpenEnv's create_app helper.
"""
from openenv.core.env_server.http_server import create_app

from models import GSTAction, GSTObservation
from server.environment import GSTEnvironment

app = create_app(GSTEnvironment, GSTAction, GSTObservation,
                 env_name="vyapar-gst-env")
