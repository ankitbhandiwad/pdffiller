# final

Minimal upload-ready version of the `webapp` app.

What is included:

- the self-contained `webapp` package
- static assets and templates
- a root `requirements.txt`
- an `env.example.sh` template for optional integrations

What is intentionally excluded:

- local sample PDFs and generated images
- runtime data under `webapp/data/`
- secret export files and credentials
- old `webapp1/` and `release/` copies

Run locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn webapp.app:app --reload --port 8000
```

Optional integrations:

- Google Document AI:
  `DOC_AI_PROJECT_ID`, `DOC_AI_LOCATION`, `DOC_AI_PROCESSOR_ID`, `GOOGLE_APPLICATION_CREDENTIALS`
- OpenAI / hosted LLM features:
  `OPENAI_API_KEY` or `LLM_API_KEY`

The app creates `webapp/data/` at runtime. That directory is ignored by git.
