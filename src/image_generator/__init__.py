"""Local image and video generation application."""


def create_app(config=None):
    """Create the catalog and form-preview app; catalog-only imports need no Flask."""
    from pathlib import Path

    from flask import Flask

    from .web.web import catalog_bp
    from .web.generation_routes import generation_bp

    app = Flask(__name__, static_folder=None)
    app.config.from_mapping(
        CATALOG_ROOT=Path(__file__).resolve().parents[2] / "Models/Model_Families",
    )
    app.config.from_prefixed_env(prefix="IMAGE_GENERATOR")
    if config is not None:
        app.config.update(config)
    app.register_blueprint(catalog_bp)
    app.register_blueprint(generation_bp)
    return app
