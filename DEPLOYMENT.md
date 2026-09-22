# Deployment Guide

## Streamlit Community Cloud
1. Create a GitHub repository and upload the project folder.
2. Confirm `requirements.txt` is in the repository root.
3. Open Streamlit Community Cloud and create a new app.
4. Select the repository, branch and `app.py` as the entry point.
5. Deploy.

## Local deployment
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Production notes
- The included CSV authentication is suitable for a college demonstration, not production identity management.
- Do not commit real passwords, API keys or private resumes.
- Session-saved jobs are intentionally ephemeral in this academic implementation.
- Scanned/image-only PDFs need OCR before text extraction.