# Decision Companion

A multi-agent decision-making system combining AI research with the Fuzzy TOPSIS algorithm for ranked, explainable recommendations. Users define options and weighted criteria. Crucially, if the LLM lacks the pre-trained knowledge to evaluate the criteria, users can upload custom PDFs to trigger local decision-making—forcing the AI to score, rank, and explain the best choice based strictly on your provided documents.

## Architecture

```
Streamlit (Frontend)  ──HTTP──>  Django (Backend API)
       │                              │
       │                         ┌────┴─────┐
       │                    LangGraph       ChromaDB
       │                   (Agents)      (Vector Store)
       │                   ┌───┴───┐
       │              Research   Synthesis
       │              Agent      Agent
       │                │
       │           Groq / Llama 3
       │
  Fuzzy TOPSIS
  (Deterministic Math)
```

At first glance, using Django and LangGraph might look like overkill, but it solves a lot of real-world problems.I used django, because it can designed as an api and very easily integrated to any frontend. For the AI side, LangGraph is essential because getting accurate numbers for the Fuzzy TOPSIS math requires more than a simple linear prompt and also to implement RAG it was necessary. Building it as a graph handles the orchestration right now, and it lays the groundwork for a future feature: adding a 'Critic Agent' that can dynamically loop back and fix LLM hallucinations on the fly.


**Frontend** — Streamlit multipage app with a five-phase workflow.
**Backend** — Django REST API exposing `/api/research/` and `/api/calculate/`.
**AI Agents** — LangGraph orchestrates a Research Agent (scores options against criteria using LLM + RAG context) and a Synthesis Agent (explains the mathematical result in plain language).
**Core Algorithm** — Fuzzy TOPSIS using triangular fuzzy numbers, implemented in pure Python with no AI dependency.
**RAG Pipeline** — Per-option PDF uploads are chunked, embedded locally with a HuggingFace model, stored in an ephemeral ChromaDB collection, and retrieved per (option, criterion) pair to ground the LLM's scoring in real evidence.

## Workflow

1. **Define** — Enter a decision goal, options (with optional PDF documents each), and weighted criteria.
2. **Research** — The AI Research Agent scores every (option, criterion) pair as triangular fuzzy numbers (l, m, u) with justifications. When PDFs are uploaded, RAG context is injected so scores are evidence-driven.
3. **Review** — Inspect and edit the AI-generated fuzzy scores and confirm criteria classification (benefit / cost).
4. **Calculate** — Deterministic Fuzzy TOPSIS computes closeness coefficients and ranks all options.
5. **Results** — View the winner, closeness coefficients, and a two-part explanation (Algorithmic Breakdown + Contextual Fit).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Pandas |
| Backend API | Django, Django CORS Headers |
| Data Models | Pydantic v2 |
| AI Orchestration | LangGraph |
| LLM | Groq (Llama 3.3 70B) via LangChain |
| RAG Embeddings | sentence-transformers (all-MiniLM-L6-v2), ChromaDB |
| PDF Processing | PyPDF, LangChain Text Splitters |

## Setup

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/SJSIO/Decison_Companion.git
cd Decison_Companion

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or export directly:

```
GROQ_API_KEY=gsk_your_key_here
```

Optional RAG configuration:

```
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2    # HuggingFace model id (default)
RAG_EMBEDDING_DEVICE=cpu                  # cpu or cuda (default: cpu)
```

### Run

Start both servers (two terminals):

```bash
# Terminal 1 — Django backend
python manage.py runserver

# Terminal 2 — Streamlit frontend
streamlit run streamlit_app.py
```

Open the Streamlit URL shown in Terminal 2 (default: `http://localhost:8501`).

## Project Structure

```
Decsion_Companion/
├── config/                  # Django project settings & URL config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── decision_comp/           # Django app — core logic
│   ├── models.py            # Pydantic schemas (options, criteria, scores, TFN)
│   ├── views.py             # API endpoints (/api/research/, /api/calculate/)
│   ├── graph.py             # LangGraph agents, Fuzzy TOPSIS implementation
│   ├── llm_services.py      # LLM prompts, structured output, deterministic guardrails
│   ├── rag.py               # PDF loading, chunking, embedding, ChromaDB retrieval
│   ├── urls.py              # API URL routing
│   └── tests/               # Unit tests (API research, RAG)
├── pages/
│   └── How_this_Algorithm_works.py   # Streamlit page explaining Fuzzy TOPSIS
├── streamlit_app.py         # Main Streamlit frontend (5-phase workflow)
├── manage.py                # Django management
├── requirements.txt         # Python dependencies
├── BUILD_PROCESS.md         # Development journal & design decisions
└── RESEARCH_LOG.md          # Research notes
```

## API Endpoints

### POST `/api/research/`

Runs the AI Research Agent to score all (option, criterion) pairs.

**Request body:**
```json
{
  "problem_description": "Hiring the best candidate",
  "options": [
    {
      "name": "Candidate A",
      "description": "Software developer",
      "documents": [{ "filename": "resume.pdf", "content_base64": "..." }]
    }
  ],
  "criteria": [
    { "name": "Communication", "weight": 8, "description": "...", "kind": "benefit" }
  ]
}
```

**Response:** Array of fuzzy scores (l, m, u) with justifications, plus classified criteria metadata.

### POST `/api/calculate/`

Runs Fuzzy TOPSIS on provided scores and returns the ranked result with an explanation.

**Request body:** Same structure plus a `scores` array of `{ option_name, criterion_name, l, m, u, justification }`.

**Response:** Winner, loser, closeness coefficients per option, explanation, and calculation intermediates.

## How Fuzzy TOPSIS Works

1. **Normalize** — Raw fuzzy scores are normalized per criterion type (benefit: divide by max; cost: invert and scale).
2. **Weight** — Each criterion's weight (1–10) is applied to the normalized matrix.
3. **Ideal Solutions** — The Fuzzy Positive Ideal Solution (FPIS) and Fuzzy Negative Ideal Solution (FNIS) are computed from the weighted matrix.
4. **Distances** — Each option's distance to FPIS and FNIS is calculated using the vertex method.
5. **Closeness Coefficient** — CC = distance_to_FNIS / (distance_to_FPIS + distance_to_FNIS). Higher CC means closer to the ideal.
6. **Rank** — Options are ranked by CC; the highest wins.

## RAG Pipeline

When PDFs are uploaded per option:

1. Each PDF is loaded and split into overlapping text chunks.
2. Chunks are embedded locally using a HuggingFace SentenceTransformer model.
3. Chunks are stored in an ephemeral ChromaDB collection with `option_name` metadata.
4. For each (option, criterion) pair, a criterion-specific similarity search retrieves the most relevant chunks (filtered by option).
5. Retrieved context is annotated with keyword-match markers and injected into the LLM prompt.
6. Deterministic guardrails boost scores when keyword evidence is found and ensure evidence-bearing options outscore those without evidence.
