"""Bounded image validation and immutable, retry-stable job input storage."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from io import BytesIO
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import warnings

from flask import current_app

from image_generator.jobs import SubmissionConflict


IMAGE_EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tiff", "GIF": "gif"}


def validate_image_upload(upload):
    """Decode for validation only; keep original bytes, orientation, and colors."""
    from PIL import Image, UnidentifiedImageError

    payload = upload.stream.read(current_app.config["INPUT_MAX_BYTES"] + 1)
    if len(payload) > current_app.config["INPUT_MAX_BYTES"]:
        raise ValueError("Source image exceeds the upload size limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                extension = IMAGE_EXTENSIONS.get(image.format)
                if extension is None:
                    raise ValueError("Choose a PNG, JPEG, WebP, BMP, TIFF, or GIF image.")
                if image.width * image.height > current_app.config["INPUT_MAX_PIXELS"]:
                    raise ValueError("Source image exceeds the decoded pixel limit.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Choose a single-frame source image.")
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Source image is invalid, damaged, or too large to decode.") from error
    return payload, extension


@dataclass
class StagedSource:
    path: Path
    retained: bool = False

    def retain(self):
        self.retained = True


@contextmanager
def stage_source(submission_key, validated):
    """Hold the submission lock through enqueue and roll back new files on failure.

    Existing sources are never replaced or removed. Per-key locks serialize
    retries across web processes so cleanup cannot remove a concurrent job input.
    """
    if not re.fullmatch(r"[0-9a-f]{32}", submission_key):
        raise ValueError("Invalid submission key")
    payload, extension = validated
    root = Path(current_app.config["JOB_INPUT_ROOT"]).expanduser().resolve()
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    if locks.is_symlink():
        raise ValueError("Input lock directory must not be a symlink")
    with (locks / submission_key).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = root / submission_key
        if directory.is_symlink():
            raise ValueError("Input directory must not be a symlink")
        directory.mkdir(exist_ok=True)
        path = directory / f"source.{extension}"
        existing = list(directory.iterdir())
        created = False
        staged = StagedSource(path)
        try:
            if existing:
                if existing != [path] or path.is_symlink() or path.read_bytes() != payload:
                    raise SubmissionConflict("This form already has a different source image. Reload to create another job.")
            else:
                with NamedTemporaryFile(dir=directory, prefix=".source-", delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        stream.write(payload)
                        stream.flush()
                        temporary.chmod(0o400)
                        temporary.replace(path)
                        created = True
                    finally:
                        temporary.unlink(missing_ok=True)
            yield staged
        finally:
            if created and not staged.retained:
                path.unlink(missing_ok=True)
            if not any(directory.iterdir()):
                directory.rmdir()


def managed_source_path(job):
    """Only a job's registered source inside the configured input root is public."""
    raw = job.request.get("inputs", {}).get("input_image")
    if not raw:
        return None
    root = Path(current_app.config["JOB_INPUT_ROOT"]).expanduser().resolve()
    path = Path(raw)
    if (path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(root)
            or path.parent.parent != root or not re.fullmatch(r"[0-9a-f]{32}", path.parent.name)
            or path.name not in {f"source.{extension}" for extension in IMAGE_EXTENSIONS.values()}
            or not path.is_file()):
        return None
    return path
