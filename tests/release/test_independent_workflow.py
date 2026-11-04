"""Independent archive-level checks of the release admission boundary."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("independent_verify_release", ROOT / "scripts/verify_release.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class DistributionAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.version = verify.version()
        self.metadata = (
            "Metadata-Version: 2.4\nName: autorsi\n"
            f"Version: {self.version}\nLicense-Expression: MIT\nLicense-File: LICENSE\n"
            "Classifier: Development Status :: 3 - Alpha\n\nOffline archive fixture.\n"
        ).encode()

    def distributions(self, *, wheel_extra=None, sdist_extra=None, sdist_link=None):
        license_bytes = (ROOT / "LICENSE").read_bytes()
        info = f"autorsi-{self.version}.dist-info/"
        with zipfile.ZipFile(self.directory / f"autorsi-{self.version}-py3-none-any.whl", "w") as archive:
            archive.writestr(info + "METADATA", self.metadata)
            archive.writestr(info + "licenses/LICENSE", license_bytes)
            if wheel_extra is not None:
                archive.writestr(wheel_extra, b"must never be admitted")
        prefix = f"autorsi-{self.version}/"
        with tarfile.open(self.directory / f"autorsi-{self.version}.tar.gz", "w:gz") as archive:
            entries = {prefix + "PKG-INFO": self.metadata, prefix + "LICENSE": license_bytes}
            if sdist_extra is not None:
                entries[sdist_extra] = b"must never be admitted"
            for path, data in entries.items():
                member = tarfile.TarInfo(path)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if sdist_link is not None:
                member = tarfile.TarInfo(prefix + sdist_link)
                member.type = tarfile.SYMTYPE
                member.linkname = "/outside-the-release"
                archive.addfile(member)

    def test_wheel_private_member_is_refused_at_archive_boundary(self):
        self.distributions(wheel_extra="evidence/private.json")
        with self.assertRaisesRegex(ValueError, "unexpected wheel member"):
            verify.check_distributions(self.directory, self.version)

    def test_source_private_member_is_refused_at_archive_boundary(self):
        self.distributions(sdist_extra=f"autorsi-{self.version}/reviews/private.json")
        with self.assertRaisesRegex(ValueError, "unexpected source distribution member"):
            verify.check_distributions(self.directory, self.version)

    def test_source_link_cannot_impersonate_an_allowed_python_file(self):
        self.distributions(sdist_link="src/autorsi/injected.py")
        with self.assertRaisesRegex(ValueError, "unexpected source distribution member"):
            verify.check_distributions(self.directory, self.version)

    def test_source_member_outside_version_root_is_refused(self):
        self.distributions(sdist_extra="outside-version-root/LICENSE")
        with self.assertRaisesRegex(ValueError, "unexpected source distribution member"):
            verify.check_distributions(self.directory, self.version)


if __name__ == "__main__":
    unittest.main()
