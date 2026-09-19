# AI Tutor 16.22

AI Tutor 16.22 is a course-grounded tutoring system using Retrieval-Augmented Generation (RAG), course-specific Chroma knowledge stores, configurable AI providers, and fail-closed runtime validation.

The validated deployment supports two courses:

- `electronics`
- `mathematics`

The tested runtime baseline is Python 3.11.1 on Windows.

---

## 1. Release Baseline

Frozen functional source release:

- Tag: `v16.22-rc2`
- Commit: `75006ea3ac359c70c19f4fda0966846537cab3ec`
- Static regression: `507 / 507 PASS`
- Live integration: `8 / 8 PASS`

Deployment packaging is maintained on the `deployment/16.22-packaging` branch after the frozen functional release.

The production application architecture was not modified during deployment packaging.

---

## 2. Runtime Architecture

When:

```text
AI_PROVIDER=ollama
```

the runtime uses:

```text
Chat       -> Ollama
Embedding  -> Ollama
```

When:

```text
AI_PROVIDER=groq
```

the runtime uses:

```text
Chat       -> Groq
Embedding  -> Ollama
```

Ollama is therefore required for embeddings even when Groq is selected as the chat provider.

Validated models:

```text
Ollama chat:       qwen3:8b
Embedding:         nomic-embed-text
Groq chat:         openai/gpt-oss-20b
```

The validated Chroma stores use 768-dimensional embeddings.

---

## 3. Supported Courses

### Electronics

```text
Course ID:        electronics
Chroma path:      ./data/chroma/electronics
Indexed records:  17
Source:           Lecture_Transistor.pdf
```

### Mathematics

```text
Course ID:        mathematics
Chroma path:      ./data/chroma/mathematics
Indexed records:  124
Source:           mathematics.pdf
```

Course profiles are stored in:

```text
app/courses/electronics/course.json
app/courses/mathematics/course.json
```

The course profile is authoritative for runtime Chroma selection.

---

## 4. Requirements

Required:

- Python 3.11.x
- Ollama
- Prebuilt course-specific Chroma stores

Tested Python version:

```text
Python 3.11.1
```

Check Ollama:

```powershell
ollama list
```

For the validated deployment, the following models are required:

```text
nomic-embed-text
qwen3:8b
```

`nomic-embed-text` is required for RAG embeddings for both Ollama and Groq chat modes.

---

## 5. Create a Python Environment

For a new installation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Do not recreate `.venv` while that same environment is currently activated.

Check Python:

```powershell
python --version
```

---

## 6. Install Runtime Dependencies

For a reproducible runtime installation:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
```

Expected:

```text
No broken requirements found.
```

The direct runtime dependency list is:

```text
requirements.txt
```

The frozen runtime dependency closure is:

```text
requirements-lock.txt
```

---

## 7. Environment Configuration

The repository provides:

```text
.env.example
```

The real `.env` file is excluded from Git.

Create `.env` only when it does not already exist:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
} else {
    Write-Host ".env already exists - leaving it unchanged."
}
```

Never overwrite an existing `.env` without checking whether it contains deployment-specific settings or credentials.

Default configuration:

```dotenv
APP_NAME=AI Tutor
ACTIVE_COURSE=electronics

AI_PROVIDER=ollama

GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

EMBEDDING_MODEL=nomic-embed-text
```

Never commit the real `.env`.

---

## 8. Groq Mode

To use Groq for chat:

```dotenv
AI_PROVIDER=groq
GROQ_API_KEY=<your Groq API key>
GROQ_MODEL=openai/gpt-oss-20b
```

Do not put a real API key in:

- `.env.example`
- source code
- README files
- Git commits
- diagnostic scripts

Even in Groq mode, Ollama must be running because embeddings are generated through Ollama.

---

## 9. Runtime Chroma Assets

The runtime release artifact must contain:

```text
data/
└── chroma/
    ├── electronics/
    └── mathematics/
```

Required databases:

```text
data/chroma/electronics/chroma.sqlite3
data/chroma/mathematics/chroma.sqlite3
```

Do not include the legacy root store:

```text
data/chroma/chroma.sqlite3
data/chroma/<legacy UUID>/
```

Runtime retrieval uses existing-only Chroma access and must not silently create a missing store or collection.

---

## 10. Deployment Preflight

Before starting the Tutor, run:

```powershell
python deployment_preflight.py
```

A successful deployment ends with:

```text
PREFLIGHT                    PASS
```

The preflight validates:

- Python version
- configuration
- AI provider
- Groq credential presence when required
- course profile
- course-specific Chroma path
- Chroma database
- required collection
- non-empty knowledge store
- provider construction

The preflight does not make live chat or embedding requests.

It also does not create a missing Chroma store or collection.

---

## 11. Switch Course

Use:

```dotenv
ACTIVE_COURSE=electronics
```

or:

```dotenv
ACTIVE_COURSE=mathematics
```

Then run:

```powershell
python deployment_preflight.py
```

Validated record counts:

```text
electronics    17
mathematics    124
```

An unknown course ID fails startup.

---

## 12. Dual-Course Deployment Smoke Test

Run:

```powershell
python deployment_dual_course_smoke.py
```

The test validates:

- both Chroma stores
- 768-dimensional stored embeddings
- course-local RAG retrieval
- context construction
- citation construction
- cross-course isolation

Successful completion ends with:

```text
NETWORK_EMBEDDING_CALLS=0 (embedding boundary intercepted)
DUAL_COURSE_RAG_SMOKE=PASS
```

The smoke test does not make an external embedding request.

---

## 13. Start AI Tutor

After the preflight passes:

```powershell
python -m app.main
```

The active course and AI provider are read from `.env` or from configuration defaults.

---

## 14. Optional Document Indexing

Source PDFs are not required for normal Tutor runtime when validated prebuilt Chroma stores are supplied.

Canonical source documents for rebuilding indexes are:

```text
docs/electronics/Lecture_Transistor.pdf
docs/mathematics/mathematics.pdf
```

For indexing, install:

```powershell
python -m pip install -r requirements-indexing-lock.txt
```

The indexing dependency set additionally contains:

```text
pymupdf==1.28.2
```

PyMuPDF is intentionally excluded from the normal runtime dependency lock.

The duplicate legacy PDF:

```text
docs/Lecture_Transistor.pdf
```

must not be used as the canonical electronics source.

---

## 15. Dependency Files

Runtime:

```text
requirements.txt
requirements-lock.txt
```

Indexing:

```text
requirements-indexing.txt
requirements-indexing-lock.txt
```

Use the lock files for reproducible release installation.

---

## 16. Fail-Closed Startup

Startup fails when:

- `ACTIVE_COURSE` is unknown
- a course profile is invalid
- the active course has no valid Chroma path
- the Chroma directory is missing
- `chroma.sqlite3` is missing
- the required collection is missing
- the collection is empty
- `AI_PROVIDER` is unsupported
- Groq is selected without `GROQ_API_KEY`

Missing runtime knowledge stores are not automatically created.

---

## 17. Security

Never commit:

```text
.env
.venv/
API keys
credentials
local secrets
```

Verify that `.env` is not tracked:

```powershell
git ls-files .env
```

Expected:

```text
<no output>
```

Verify that `.env` is ignored:

```powershell
git check-ignore -v .env
```

---

## 18. Runtime Release Artifact

The runtime release should contain:

```text
.env.example
.gitignore
README.md

requirements.txt
requirements-lock.txt
requirements-indexing.txt
requirements-indexing-lock.txt

deployment_preflight.py
deployment_dual_course_smoke.py

app/

data/
└── chroma/
    ├── electronics/
    └── mathematics/
```

Do not include:

```text
.env
.venv/
legacy root Chroma
duplicate root PDF
developer diagnostics
temporary development files
```

Source PDFs may be distributed separately as an optional indexing bundle.

---

## 19. Deployment Validation

The current deployment validation has passed:

```text
P1  Repository hygiene                  PASS
P2  Dependency freeze                   PASS
P3  Environment configuration           PASS
P4  Chroma packaging strategy           PASS
P5  Startup preflight                   PASS
P6  Clean-environment installation      PASS
P7  Dual-course deployment smoke        PASS
P8  Provider deployment smoke           PASS
```
