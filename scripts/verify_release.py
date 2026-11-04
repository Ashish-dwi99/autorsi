"""Validate release identity and a narrow distribution inventory without uploads."""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {"README.md", "PACKAGE_README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md",
             "pyproject.toml", "MANIFEST.in", "PKG-INFO", "setup.cfg"}
EGG_INFO = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt"}
DOCUMENTS = {"docs/integration.md", "docs/releasing.md"}
SCRIPTS = {"scripts/render_package_readme.py", "scripts/verify_release.py"}


def version(root=ROOT):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    value = project["version"]
    if project["name"] != "autorsi" or project.get("license") != "MIT":
        raise ValueError("release must declare package autorsi and SPDX license MIT")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?", value):
        raise ValueError("release version must be a canonical numeric release or prerelease")
    tree = ast.parse((root / "src/autorsi/__init__.py").read_text())
    declared = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    if declared != [value]:
        raise ValueError("package __version__ and pyproject.toml version disagree")
    return value


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check_identity(tag, release_version, expected_sha=None, git=git_output):
    if tag != "v" + release_version:
        raise ValueError("release tag must exactly equal v + the package version")
    if expected_sha is None:
        return
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("expected commit must be a complete lowercase 40-character SHA")
    for ref in ("HEAD", "refs/remotes/origin/main", "refs/tags/" + tag + "^{commit}"):
        if git("rev-parse", "--verify", ref) != expected_sha:
            raise ValueError(f"{ref} does not match the immutable dispatched main commit")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked release files were modified after checkout")


def source_allowed(path):
    parts = PurePosixPath(path).parts
    if path in TOP_LEVEL | DOCUMENTS | SCRIPTS:
        return True
    if len(parts) == 3 and parts[:2] == ("src", "autorsi") and parts[-1].endswith(".py"):
        return True
    if len(parts) == 3 and parts[:2] == ("src", "autorsi.egg-info") and parts[-1] in EGG_INFO:
        return True
    return len(parts) == 3 and parts[:2] in {("examples", "autorsi"), ("tests", "autorsi"), ("tests", "release")} and parts[-1].endswith(".py")


def check_metadata(raw, release_version):
    metadata = BytesParser().parsebytes(raw)
    if metadata["Name"] != "autorsi" or metadata["Version"] != release_version:
        raise ValueError("distribution name/version does not match the release")
    if metadata["License-Expression"] != "MIT" or "LICENSE" not in metadata.get_all("License-File", []):
        raise ValueError("distribution must include SPDX MIT metadata and LICENSE")
    if "Development Status :: 3 - Alpha" not in metadata.get_all("Classifier", []):
        raise ValueError("distribution must declare its alpha status")
    if "```mermaid" in metadata.get_payload():
        raise ValueError("package README must use the index-compatible text workflow")


def check_distributions(directory, release_version):
    directory = Path(directory)
    expected = {f"autorsi-{release_version}-py3-none-any.whl", f"autorsi-{release_version}.tar.gz"}
    files = {path.name for path in directory.iterdir()}
    if files != expected:
        raise ValueError(f"dist directory must contain exactly {sorted(expected)}; found {sorted(files)}")
    wheel = directory / f"autorsi-{release_version}-py3-none-any.whl"
    info = f"autorsi-{release_version}.dist-info/"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        allowed_metadata = {info + name for name in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD", "licenses/LICENSE")}
        for item in names:
            parts = PurePosixPath(item).parts
            if item not in allowed_metadata and not (len(parts) == 2 and parts[0] == "autorsi" and parts[1].endswith(".py")):
                raise ValueError(f"unexpected wheel member: {item}")
        check_metadata(archive.read(info + "METADATA"), release_version)
        if archive.read(info + "licenses/LICENSE") != (ROOT / "LICENSE").read_bytes():
            raise ValueError("wheel license differs from the reviewed license")
    sdist = directory / f"autorsi-{release_version}.tar.gz"
    prefix = f"autorsi-{release_version}/"
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix) or not source_allowed(member.name[len(prefix):]):
                raise ValueError(f"unexpected source distribution member: {member.name}")
        check_metadata(archive.extractfile(prefix + "PKG-INFO").read(), release_version)
        if archive.extractfile(prefix + "LICENSE").read() != (ROOT / "LICENSE").read_bytes():
            raise ValueError("source license differs from the reviewed license")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-sha", help="require HEAD, fetched main and tag to equal this immutable SHA")
    parser.add_argument("--dist", type=Path, help="check built wheel/sdist inventories and metadata")
    args = parser.parse_args()
    release_version = version()
    check_identity(args.tag, release_version, args.expected_sha)
    result = {"version": release_version, "tag": args.tag, "commit": args.expected_sha}
    if args.dist:
        result["sha256"] = check_distributions(args.dist, release_version)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
