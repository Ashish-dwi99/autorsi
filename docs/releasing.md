# Releasing AutoRSI

**0.1.0a1 is published on [PyPI](https://pypi.org/project/autorsi/0.1.0a1/)** under the MIT license. [Release run 41761677232](https://github.com/geniusgeek/autorsi/actions/runs/41761677232) published tag `v0.1.0a1` at commit `78af2b26899e8352117d2f083791c54db398fe9a` after all 35 core/model tests and 11 release checks passed. A clean PyPI install and installed demo passed. The repository remains private; PyPI installation is public, while source links require repository access.

The `pypi` environment is configured for `main` only, and release tag ruleset `26502764` prevents tag updates and deletion. The instructions below describe future releases: choose a new version and tag. Do not rebuild, reupload or move the published `v0.1.0a1` release.

## Local preparation for the next version

1. Choose a new version, keep `pyproject.toml` and `src/autorsi/__init__.py` identical, and update `CHANGELOG.md`. Replace `v0.1.0a1` in the example commands below with that new version's tag.
2. Edit `README.md`, then run `python scripts/render_package_readme.py`. This preserves the GitHub Mermaid diagram while generating `PACKAGE_README.md` with a text workflow and absolute documentation links for package indexes.
3. Create a development environment, install the build tools and optional model-test dependencies, and run the checks below. No provider keys or model downloads are required by the tests.

```sh
python -m pip install -e '.[hf-demo]' 'build>=1.2' 'twine>=6'
python scripts/render_package_readme.py --check
python scripts/verify_release.py --tag v0.1.0a1
python -m unittest discover -s tests/autorsi -v
python -m unittest discover -s tests/release -v
python -m build
python -m twine check --strict dist/*
python scripts/verify_release.py --tag v0.1.0a1 --dist dist
```

Start with an empty `dist/` directory: the verifier refuses extra or stale artifacts. Inspect the wheel and source distribution before publishing. The narrow inventory excludes historical model experiments, private artifacts, saved checkpoints and local review records. `LICENSE` must appear in both distributions and metadata must declare the SPDX expression `MIT`.

Install the built wheel into a fresh virtual environment with `--no-index --no-deps`, change to a directory outside the checkout, and run `autorsi-demo`. This verifies the installed package rather than a source import. The release workflow performs the same checks, runs all optional model tests without skips and builds the wheel from the source distribution.

## One-time publisher configuration

The publisher and environment are already configured for this repository. For a new repository or configuration change, these are separate maintainer account/repository actions:

- Configure a PyPI pending trusted publisher (or add one to the existing project) for project `autorsi`, GitHub owner `geniusgeek`, repository `autorsi`, workflow **`release.yml`**, and environment **`pypi`**. No API token or password belongs in repository secrets.
- Create the GitHub environment `pypi` and configure a selected **Branch** rule for `main`, with no tag or other branch deployment rules. Recommend a required maintainer reviewer where the repository visibility and plan support it. GitHub Pro supports environment branch restrictions for private repositories, but its required-reviewer rules are available only for public repositories. In a private Pro repository, the authorized manual workflow dispatch is the explicit publication authorization: after validation succeeds, publishing proceeds without a second approval prompt. Merely naming an environment in YAML does **not** configure branch restrictions or reviewers. See [GitHub's environment availability and configuration](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).
- Protect release tags from deletion or replacement using the available repository controls. Tag protection is an external setting; this workflow verifies source identity but does not configure repository rules.

The publisher must be registered on PyPI before uploading. See [PyPI's trusted-publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) and [publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Explicit publication

Publishing is not triggered by a push, tag or GitHub Release. For a new version, replace the example tag `v0.1.0a1` below with its new tag. After review and explicit maintainer authorization:

1. Commit the reviewed release preparation to `main`. Create and push the exact tag `v0.1.0a1` on that same commit. Do not change a version after its distributions have been uploaded.
2. Manually run **Manual alpha release**, selecting branch `main` and entering tag `v0.1.0a1`. This dispatch explicitly authorizes publication after all checks and configured environment rules pass; it is not a dry run. If required reviewers are unavailable or not configured, no second approval will occur. Use **Framework checks** or the local commands for validation without publication.
3. Confirm the checked commit and distribution checks. The workflow checks out the immutable dispatch SHA, and refuses unless `HEAD`, the fetched `origin/main`, and the peeled release tag all match it. The input tag must exactly equal `v` plus the package version. If `main` moved before checkout, dispatch again from the intended reviewed release commit on `main` instead of relaxing the check.
4. If a supported required-reviewer rule is configured, review and approve the `pypi` deployment. Otherwise, publication proceeds under the authorization given by the manual dispatch once checks and the main-only branch restriction pass. The isolated publishing job downloads only the immutable artifact ID produced by this workflow run and fails on a digest mismatch. It does not check out repository code, build, install the package or run tests with publishing privileges.
5. Verify the actual PyPI version, files and attestations. Only then update the changelog/README to say the version is published and document the exact installation command for the alpha.

The publishing job alone has `id-token: write`. It uses PyPI trusted publishing with attestations and refuses existing distributions. A failed or partially completed upload is not proof that nothing was published: inspect PyPI before retrying. Do not enable `skip-existing` to conceal an inconsistent release.

The workflow pins official actions to reviewed full commit SHAs. Updating those pins and build dependencies is a separate reviewed change. Its design follows [PyPA's isolated publishing guidance](https://github.com/pypa/gh-action-pypi-publish#trusted-publishing), [GitHub's manual workflow semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch) and [artifact identity/integrity controls](https://github.com/actions/download-artifact#inputs).
