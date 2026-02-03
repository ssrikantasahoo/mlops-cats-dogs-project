# Final Submission Checklist (Assignment 2)

Use this checklist before uploading the final ZIP.

## Required files in ZIP

- [ ] Source code (`src/`, `scripts/`, `tests/`)
- [ ] CI/CD config (`.github/workflows/ci-cd.yml`)
- [ ] Docker artifacts (`Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`)
- [ ] Deployment manifests (`kubernetes/`)
- [ ] Trained model artifact (`models/model.pt`) or hosted link
- [ ] DVC/Git-LFS metadata (`.dvc/`, `.dvcignore`, `dvc.yaml`, `data/*.dvc`, `.gitattributes`)
- [ ] Monitoring config/artifacts (`prometheus.yml`, monitoring scripts)
- [ ] Rubric evidence map (`EVIDENCE_INDEX.md`)

## Runtime evidence to capture (outside code)

- [ ] CI run URL showing test/build success
- [ ] Registry page URL showing pushed Docker image tags/digest
- [ ] CD run URL showing deploy + smoke test pass
- [ ] Monitoring artifact download from deploy job (`production-monitoring-evidence`)

## Mandatory recording (< 5 minutes)

- [ ] Show repo structure and key files
- [ ] Show CI run (tests + docker build/push)
- [ ] Show CD run on `main` (deployment + smoke tests)
- [ ] Show live prediction request and response
- [ ] Show monitoring outputs/logs

## Build ZIP

Run:

```bash
python scripts/create_submission_bundle.py
```

Output:

- `submission/mlops_assignment2_submission.zip`
