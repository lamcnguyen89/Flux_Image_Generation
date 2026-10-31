"""Run the local Flask catalog and generation forms with `python main.py`."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from image_generator import create_app


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    create_app().run()
