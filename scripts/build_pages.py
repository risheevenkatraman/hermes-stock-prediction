"""Create the small public Pages artifact; never copy the repository wholesale."""

import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit


def build(destination: Path, api_url: str) -> None:
    url = urlsplit(api_url.strip())
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "PUBLIC_API_URL must be an HTTPS origin, e.g. https://api.example.com"
        )
    # Also reject malformed port numbers before publishing.
    _ = url.port
    root = Path(__file__).resolve().parents[1]
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(root / name, destination / name)
    (destination / "config.js").write_text(
        "window.HERMES_CONFIG = "
        + json.dumps({"apiBase": api_url.strip().rstrip("/")})
        + ";\n",
        encoding="utf-8",
    )
    (destination / ".nojekyll").touch()


if __name__ == "__main__":
    build(Path("dist/pages"), os.environ.get("PUBLIC_API_URL", ""))
