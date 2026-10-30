"""Local image and video generation application."""


def create_app(config=None):
    """Create the catalog, generation forms, and job UI without GPU imports."""
    from pathlib import Path
    from secrets import token_hex

    from flask import Flask

    from .web.web import catalog_bp
    from .web.generation_routes import generation_bp
    from .web.job_routes import jobs_bp

    app = Flask(__name__, static_folder=None)
    app.config.from_mapping(
        CATALOG_ROOT=Path(__file__).resolve().parents[2] / "Models/Model_Families",
        JOB_DATABASE=Path(__file__).resolve().parents[2] / "data/jobs.sqlite3",
        JOB_OUTPUT_ROOT=Path(__file__).resolve().parents[2] / "data/outputs",
    )
    app.config.from_prefixed_env(prefix="IMAGE_GENERATOR")
    if config is not None:
        app.config.update(config)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = token_hex(32)
    app.register_blueprint(catalog_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(jobs_bp)
    return app
