import os
from flask import Flask


def create_app():
    app = Flask(__name__)

    # Secret key for sessions
    app.secret_key = os.urandom(24)

    # Upload folder configuration
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = upload_folder

    # Registering Blueprints
    from .routes import main
    app.register_blueprint(main)

    return app