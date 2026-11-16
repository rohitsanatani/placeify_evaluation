"""WSGI entry point: gunicorn --workers 1 --threads 8 wsgi:app."""
import json

from app import create_app
from common import HERE

with (HERE / "config.json").open() as config_file:
    app = create_app(json.load(config_file))

if __name__ == "__main__":
    app.run()
