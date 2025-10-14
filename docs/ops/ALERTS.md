# Oscillink Firm — Prometheus Alert Rules

If you use Prometheus, add these alerting rules to monitor OCR quality and abstains. If you don’t ship Prometheus, add equivalent patterns to your log alerts.

```yaml
groups:
- name: oscillink-firm-quality
  rules:
  - alert: OscLowOCRAbstainSpike
    expr: rate(osc_query_abstain_total{reason="low_ocr"}[15m]) > 3
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Low-quality OCR causing abstains"
      description: "Over the last 15m, more than 3 queries/min abstained due to low OCR quality. Investigate flagged sources in receipts."

  - alert: OscLowOCRDocsAccumulating
    expr: increase(osc_ocr_low_conf_total[24h]) > 25
    for: 5m
    labels:
      severity: info
    annotations:
      summary: "Many docs flagged with low OCR confidence in 24h"
      description: "Users are ingesting poor scans. Advise re-scan at 300–600 DPI. See OPERATIONS.md → OCR governance."
```
