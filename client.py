"""
client.py — HTTP client for connecting to the vyapar-gst-env Space.
Uses GenericEnvClient from openenv for dict-based interaction.
"""
from openenv import GenericEnvClient


class GSTEnvClient(GenericEnvClient):
    """Thin wrapper around GenericEnvClient for vyapar-gst-env."""
    pass
