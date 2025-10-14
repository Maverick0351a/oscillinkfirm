# Kubernetes deployment (licensed, local-first)

This guide deploys the query service on Kubernetes using the included Helm chart with a licensed, no‑egress posture. It mirrors the Docker Compose setup.

## Prereqs
- A Kubernetes cluster and `kubectl`/`helm`
- A Secret containing:
  - `oscillink.lic` — Ed25519-signed JWT
  - `jwks.json` — JWKS (mirrored internally for air‑gapped)

Example Secret:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: oscillink-license
  namespace: default
stringData:
  oscillink.lic: |
    <paste-your-license-jwt>
  jwks.json: |
    {"keys": [ ... ]}
```

## Install chart

```sh
helm upgrade --install oscillink deploy/helm/oscillink \
  --set image.repository=ghcr.io/oscillink/oscillink \
  --set image.tag=v0.1.13 \
  --set env.OSCILLINK_JWKS_URL=file:///run/secrets/jwks.json \
  --set secretMounts[0].name=license \
  --set secretMounts[0].mountPath=/run/secrets \
  --set secretMounts[0].secretName=oscillink-license \
  --set serviceAccount.create=true \
  --set serviceAccount.automountServiceAccountToken=false \
  --set networkPolicy.enabled=true \
  --set networkPolicy.allowDNS=false \
  --set networkPolicy.egressCIDR=0.0.0.0/0 \
  --set networkPolicy.egressPorts[0]=443
```

## Probes and readiness
- Readiness probes target `/license/status` and will fail until entitlements are present.
- Liveness probes target `/health`.

## Security posture
- Pod runs as non‑root, read‑only root fs, dropped capabilities, RuntimeDefault seccomp, no token automount.
- NetworkPolicy is off by default. Enable it and scope egress to your JWKS location (or disable egress when using file:// JWKS).

## Verify
```sh
kubectl get pods
kubectl port-forward deploy/oscillink 8080:8080 &
curl -fsS http://127.0.0.1:8080/license/status
curl -fsS http://127.0.0.1:8080/health
```

## Uninstall
```sh
helm uninstall oscillink
```
