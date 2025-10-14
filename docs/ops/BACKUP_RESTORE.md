# Backup and Restore

This guide covers simple backup and restore for Oscillink indexes and receipts.

## What to back up
- JSONL indexes and any FAISS index files (.faiss + .meta.jsonl)
- Ingest receipts (*.ingest.json) sidecars
- Usage logs/receipts (if you persist response receipts)

## Simple backup script
You can use the provided utility to bundle indexes and receipts into a timestamped tar archive.

```bash
# Windows PowerShell users can invoke python directly
python scripts/backup_indexes.py --source /data/indexes --out /backups
```

## Restore
- Extract the archive into the target directory
- Point the query or OPRA services at the restored index path

## Notes
- Consider encrypting archives at rest (e.g., with OS-provided encryption)
- Maintain retention per your data governance policy
