# Releasing ASTScribe

ASTScribe publishes with PyPI Trusted Publishing. No long-lived API token or GitHub
secret is required.

## One-time setup

Create accounts on both [PyPI](https://pypi.org/) and
[TestPyPI](https://test.pypi.org/). They are separate services and require separate
accounts.

Register a pending GitHub publisher on each index with these exact values:

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| PyPI project name | `astscribe` | `astscribe` |
| Owner | `edujbarrios` | `edujbarrios` |
| Repository | `astscribe` | `astscribe` |
| Workflow | `release.yml` | `test-release.yml` |
| Environment | `pypi` | `testpypi` |

Create matching GitHub environments named `pypi` and `testpypi`. Configure required
reviewers on `pypi` if production publication should require manual approval.

The pending publisher creates the project on first publication. It does not reserve
the name in advance.

## TestPyPI rehearsal

1. Update the package version if that version already exists on TestPyPI.
2. Run the `test-release` workflow manually from the GitHub Actions page.
3. Install the uploaded build in a clean environment:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ --no-deps astscribe
   python -m astscribe --version
   ```

## Production release

1. Confirm `main` is green.
2. Add the dated release section to `CHANGELOG.md`.
3. Confirm `astscribe.__version__`, `CITATION.cff`, and `CHANGELOG.md` agree.
4. Open **Actions → release → Run workflow**, select `main`, and run it.
5. Approve the `pypi` environment deployment if reviewers are configured.
6. Confirm the matching GitHub Release and PyPI release were created.
7. Verify installation and the CLI from a fresh environment.

The production workflow reads the package version from `astscribe.__version__`,
rejects releases from branches other than `main`, validates metadata, refuses to
overwrite an existing version tag, builds and smoke-tests the distributions, creates
the annotated `vX.Y.Z` tag, publishes those exact artifacts to PyPI with Trusted
Publishing, and finally creates the matching GitHub Release.

A guarded push trigger exists only to bootstrap the workflow itself: a change to
`.github/workflows/release.yml` on `main` publishes only when that commit message
contains `[publish-pypi]`. Normal source-code pushes never publish a release.
