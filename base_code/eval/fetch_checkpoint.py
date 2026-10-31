"""Download the team's public Drive checkpoint and verify its immutable SHA-256."""
import argparse
from html.parser import HTMLParser
from pathlib import Path

import requests

from sample_t2i import sha256

DAVID_FILE_ID = "1j-Y6_RWIFsyos7pAT5eBBhAcPtEjJ95a"
DAVID_SHA256 = "f693fd7391979856e7059994643e755239276ba5d66a31becf1ebf1e18e93c6b"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}
        self.active = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs["action"]
            self.active = True
        if tag == "input" and self.active and attrs.get("name"):
            self.params[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def download_checkpoint(destination, *, file_id=DAVID_FILE_ID, expected_sha256=DAVID_SHA256):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError("Existing checkpoint hash differs; it was not overwritten")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    with requests.Session() as session:
        response = session.get("https://drive.google.com/uc", params={"export": "download", "id": file_id},
                               stream=True, timeout=60)
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            form = DownloadForm()
            form.feed(response.text)
            response.close()
            if form.action != "https://drive.usercontent.google.com/download":
                raise RuntimeError("Drive did not supply the public download form; check sharing/access")
            response = session.get(form.action, params=form.params, stream=True, timeout=60)
            response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Drive returned a web page rather than model weights")
        with response, temporary.open("wb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                stream.write(chunk)
    if sha256(temporary) != expected_sha256:
        temporary.unlink()
        raise ValueError("Downloaded checkpoint does not match the inspected team file")
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = download_checkpoint(args.output)
    print(f"Verified checkpoint: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
