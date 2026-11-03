"""Image-build-only pinned Chrome/driver acquisition; never run on the host."""
import hashlib
import json
from pathlib import Path
import stat
import time
import urllib.request
import zipfile

VERSION = "156.0.8078.80"
ROOT = Path("/opt/cft")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("unexpected redirect for version-pinned browser archive")


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(NoRedirect)
    receipts = {"version": VERSION, "archives": {}, "binaries": {}, "files": {}}
    extracted_bytes = 0
    for artifact in ("chrome-linux64.zip", "chromedriver-linux64.zip"):
        url = "https://storage.googleapis.com/chrome-for-testing-public/" + VERSION + "/linux64/" + artifact
        destination = ROOT / artifact
        started, size = time.monotonic(), 0
        digest = hashlib.sha256()
        with opener.open(url, timeout=30) as response, destination.open("xb") as output:
            if response.status != 200 or response.geturl() != url:
                raise RuntimeError("unexpected browser archive origin/status")
            while True:
                if time.monotonic() - started >= 300:
                    raise TimeoutError("browser archive download deadline")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 400 * 1024 * 1024:
                    raise ValueError("browser archive exceeds 400 MiB")
                digest.update(chunk)
                output.write(chunk)
        receipts["archives"][artifact] = {"url": url, "bytes": size, "sha256": digest.hexdigest()}
        with zipfile.ZipFile(destination) as archive:
            if len(archive.infolist()) > 10000:
                raise ValueError("too many browser archive members")
            for member in archive.infolist():
                relative = Path(member.filename)
                mode = member.external_attr >> 16
                if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                    raise ValueError("unadmitted archive path/link")
                path = ROOT / relative
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    extracted_bytes += member.file_size
                    if extracted_bytes > 1024 * 1024 * 1024:
                        raise ValueError("browser extraction exceeds 1 GiB")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, path.open("xb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    path.chmod(mode & 0o777 or 0o644)
    for relative in ("chrome-linux64/chrome", "chromedriver-linux64/chromedriver"):
        path = ROOT / relative
        path.chmod(0o755)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        receipts["binaries"][str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    for folder in (ROOT / "chrome-linux64", ROOT / "chromedriver-linux64"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                receipts["files"][str(path)] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    Path("/opt/miniwob-browser-build-receipt.json").write_text(json.dumps(receipts, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
