#!/bin/sh
# Full Round 1 paper pipeline: Python stdlib -> ODT -> LibreOffice -> PDF.
# Requires: python3 (stdlib only), soffice.
# Output: research/round1/R1_SUBMISSION.pdf (the form upload, < 10 MB).
set -e
cd "$(dirname "$0")"
python3 build_odt.py
soffice --headless --convert-to pdf --outdir . R1_SUBMISSION.odt
ls -la R1_SUBMISSION.pdf
