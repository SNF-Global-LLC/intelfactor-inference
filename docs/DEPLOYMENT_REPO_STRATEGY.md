# Deployment Repo Strategy

`intelfactor-inference` is the reusable edge runtime. It should stay generic:
camera adapters, inference runtime, evidence writer, SQLite storage, local
dashboard/API, cloud sync agent, and generic Docker Compose templates.

Customer and site deployments belong in separate deployment overlay repos, for
example `deploy-asiaone`, `deploy-wiko`, and `deploy-demo`. Those repos own
station manifests, environment examples, Docker Compose overrides, camera/ROI
settings, cloud mappings, and runbooks.

## Repository Split

```text
Core product repos
  intelbase
    cloud platform, API, dashboard, auth, AWS infrastructure modules
  intelfactor-inference
    reusable edge runtime

Deployment overlay repos
  deploy-asiaone
    Asia One station manifests, Docker overrides, AWS env mapping, camera config
  deploy-wiko
    Wiko station manifests, model config, factory-specific line setup
  deploy-demo
    sales/demo configs, synthetic data, public-safe examples

Infrastructure repos or folders
  infra-aws-prod
    shared AWS production account infrastructure
  infra-aws-asiaone
    Asia One AWS-heavy deployment infrastructure
  infra-aliyun-hk
    Alibaba Cloud / HK regional control-plane experiments
```

## Runtime Boundary

The edge station remains edge-authoritative:

```text
Camera -> edge runtime -> PASS/REVIEW/FAIL -> local SQLite/evidence -> local dashboard
                                                |
                                                v
                                      optional outbound cloud sync
```

The edge runtime decides inspection verdicts locally. Cloud services store,
sync, analyze, supervise, and scale. Cloud failure must not block camera
capture, local inference, evidence writing, SQLite persistence, or the local
operator dashboard.

## What Stays In This Repo

- Camera adapters and capture code.
- Vision/language provider resolution.
- Inference and verdict logic.
- Evidence writer and local retention.
- SQLite-backed local storage.
- Local API and dashboard.
- Cloud sync agent.
- Generic Docker Compose templates.
- Generic deployment and integration documentation.

## What Must Stay Out

- Real customer secrets.
- Customer RTSP credentials.
- AWS access keys or production API keys.
- Customer-specific LAN maps.
- Station passwords.
- Customer-specific ROI calibration as live deployment truth.
- Customer-specific station IDs as runtime defaults.

## Deployment Overlay Responsibilities

Deployment overlay repos should pin a runtime image tag and commit SHA, then
provide only site-specific configuration around that generic runtime.

Example manifest fragment:

```yaml
edge_runtime:
  repo: tonesgainz/intelfactor-inference
  image: ghcr.io/tonesgainz/intelfactor-inference
  sync_image: ghcr.io/tonesgainz/intelfactor-inference-sync
  version: edge-2026.05.05
  commit: REPLACE_WITH_RUNTIME_COMMIT_SHA
```

Deployment repos should use customer-agnostic images:

```text
ghcr.io/tonesgainz/intelfactor-inference:edge-2026.05.05
ghcr.io/tonesgainz/intelfactor-inference-sync:edge-2026.05.05
```

Do not build customer-specific runtime images such as
`intelfactor-inference-asiaone` or `intelfactor-inference-wiko`.

## Workspace Names

Use workspace as the tenant/customer/site boundary:

```text
intelfactor-internal
asiaone-staging
asiaone-prod
wiko-staging
wiko-prod
demo-public
```

## Station IDs

Use this convention:

```text
{customer}-{site}-{line}-{station}
```

Examples:

```text
asiaone-hk-line01-station01
asiaone-hk-line01-station02
wiko-yangjiang-finalinspection-station01
wiko-yangjiang-polishing-station01
demo-local-station01
```

## Manifest Schema

Each deployment repo should include `stations/station_manifest.yaml` with:

```yaml
workspace_id: asiaone-prod
customer: asiaone
site: hk
line: line01
station_id: asiaone-hk-line01-station01
camera_uri_env_var: CAMERA_URI
camera_type: usb
model_version: edge-2026.05.05
evidence_dir: /opt/intelfactor/data/evidence
sqlite_path: /opt/intelfactor/data/local.db
cloud_api_url: https://api.intelfactor.ai
s3_bucket: intelfactor-evidence-prod
s3_prefix: asiaone/hk/line01/station01
sync_interval_sec: 300
deployment_mode: hybrid
environment: prod
```

## Security Rules

- No real secrets committed.
- `.env` files ignored; only `.env.example` files committed.
- Station secrets loaded locally or from AWS Secrets Manager.
- Customer deployment repos are private.
- Demo repos may be public only if they use synthetic data.
- Camera RTSP URLs in git must be examples without real credentials.

## Promotion

Core repos use:

```text
main
feature/*
fix/*
release/*
```

Deployment repos use customer/environment branches:

```text
main
asiaone-staging
asiaone-prod
wiko-staging
wiko-prod
```

Promote deployment changes from staging to prod only after station validation.
