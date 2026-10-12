#!/usr/bin/env bash
# Convert the markdown report to docx (for editing) and pdf (submission).
set -e
cd "$(dirname "$0")/../report"
pandoc report.md -o report.docx --resource-path=. 2>&1
soffice --headless --convert-to pdf --outdir . report.docx >/dev/null 2>&1
echo "wrote report/report.docx and report/report.pdf"
