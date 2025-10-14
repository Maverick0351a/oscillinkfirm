# Production Readiness Checklist

Use this checklist to gate releases and deployments of Oscillink Firm. Each item should be satisfied or explicitly accepted as a risk with an owner and date.

## 1. Versioning and Releases
- [ ] SemVer tagging (X.Y.Z) with CHANGELOG.md updated
- [ ] PyPI package build reproducible (locked deps, hashes)
- [ ] Docker image built with pinned base image and digest
- [ ] SBOM generated and stored (e.g., syft)

## 2. CI/CD Quality Gates
- [ ] Lint (ruff) passes
- [ ] Type checks (mypy) pass
- [ ] Tests (pytest) 100% passing
- [ ] Minimal coverage threshold agreed and met (e.g., 80%)
- [ ] Platform matrix: Windows + Linux (and macOS if relevant)

## 3. Security & Compliance
- [ ] Secrets: none in repo; use env/KeyVault/Secret Manager
- [ ] Image hardening (rootless, non-root user, read-only FS where possible)
- [ ] Dependency vulnerability scan (pip-audit or Safety)
- [ ] License policy checked; third-party notices updated
- [ ] Egress policy: app binds to 127.0.0.1 by default; outbound blocked unless enabled
- [ ] Receipts never include PHI/PII values; only counts and hashes

## 4. Data Governance
- [ ] Ingest receipts persisted with parser/OCR/model hashes
- [ ] Index metadata includes ACL facets (matter_id, client_id, department)
- [ ] Role/matter filters enforced at query time
- [ ] Data retention policy documented (indexes and receipts)

## 5. Determinism & Observability
- [ ] Determinism tests pass across seeds and runs
- [ ] Receipts chained (ingest → settle) with candidate_set_hash
- [ ] Structured logging with request IDs and receipt hashes
- [ ] Metrics: latency p50/p95 for ingest, recall, settle; abstain rate

## 6. Performance & Sizing
- [ ] Benchmarks on target hardware documented
- [ ] Memory/CPU sizing guidance provided (README/docs)
- [ ] Backpressure or queueing strategy defined for large ingests

## 7. Operations
- [ ] Health endpoints return ready/live
- [ ] Graceful shutdown and idempotent work replays
- [ ] Backup/restore of indexes and receipts documented
- [ ] Runbooks for common incidents (OCR errors, bad docs, OOM)
- [ ] Rollback plan for releases (can pin prior image/tag)

## 8. Billing (Cloud/Stripe only)
- [ ] Webhook idempotency verified; payload_sha256 captured on first event
- [ ] Price map validated against docs/billing/PRICING.md
- [ ] Quotas/feature flags documented and enforced

## 9. Deployment
- [ ] Docker Compose and/or Helm values reviewed for security (ports, volumes)
- [ ] Bind addresses restricted; TLS termination documented if exposed
- [ ] Example configs sanitized; defaults safe

## 10. Documentation
- [ ] README reflects branding and capabilities
- [ ] API docs up to date (docs/API.md)
- [ ] Troubleshooting section covers top issues and their receipts

Owner:
- Checklist owner: ____
- Release owner: ____
- Security review owner: ____
