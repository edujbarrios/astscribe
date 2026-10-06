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
reviewers on `pypi` so production publication always needs manual approval.

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

1. Confirm `main` is green and the worktree is clean.
2. Add the dated release section to `CHANGELOG.md`.
3. Confirm `astscribe.__version__`, `CITATION.cff`, and `CHANGELOG.md` agree.
4. Create and push an annotated version tag:

   ```bash
   git tag -a vX.Y.Z -m "ASTScribe X.Y.Z"
   git push origin vX.Y.Z
   ```

5. Approve the `pypi` environment deployment in GitHub Actions.
6. Confirm that the workflow created the matching GitHub Release.
7. Verify the project page, installation, CLI, and import from a fresh environment.

The production workflow rejects a tag that does not match the package version, builds
the wheel and source distribution once, validates both with Twine, and publishes those
same artifacts with attestations.
