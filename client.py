"""
client.py — HTTP client for connecting to the Vyapar-RL Space.
Uses GenericEnvClient from openenv for dict-based interaction.
"""
from openenv import GenericEnvClient


class GSTEnvClient(GenericEnvClient):
    """Thin wrapper around GenericEnvClient for Vyapar-RL."""
    pass
