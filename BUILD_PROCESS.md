# Build Process

A chronological account of how this project was designed, built, broken, fixed, and refined.

---

## How I Started

**23/02/2026**

Initially I thought of building a domain specific decision companion system which helps a person with multiple offers in different companies, decide which offer to chose based on different criteria's then after the session held to clear doubts on the home assignment, I thought of building a general decision companion system where the user can ask for help in making decisions in any topics concerned and I also made sure that the, in the design, system followed  the core constraints like accept multiple options, process weighted criteria, and provide a ranked recommendation.

Through the researches I done on the topic, I planned to implement a pure mathematical backend using the Weighted Sum Model (WSM). Before introducing any AI, I built a basic Python script where a user manually inputted scores (1-10) for various options.

How it works:
1.Define Criteria & Weights: The user defines criteria (e.g., Cost, Performance, Learning Curve) and assigns a weight to each (e.g., 1 to 5, or percentages summing to 100%).
2.Score Options: The user scores each option (e.g., React, Vue, Angular) against each criterion on a standard scale (e.g., 1 to 10).
3.Calculate: For each option, multiply the score of each criterion by its weight, and sum them up.

The option with highest value is suggested by the program to the user

Then I asked gemini LLM to simulate a CLI based on the idea provided:

"$ python decision_maker.py

Welcome to the Decision Companion System!
What are you trying to decide today?
> Choosing a database for my next project

Great. Let's add your criteria.
Enter a criterion name (or type 'done'):
> Performance
Enter a weight for 'Performance' (1-10):
> 9

Enter a criterion name (or type 'done'):
> Ease of Use
Enter a weight for 'Ease of Use' (1-10):
> 6

... [User enters options like MySQL, ChromaDB, etc. and scores them] ...

--- FINAL RESULTS ---
1. ChromaDB (Score: 85)
2. MySQL (Score: 72)

Recommendation: You should choose ChromaDB because its high performance score strongly aligned with your heavily weighted criteria."


---

## How My Thinking Evolved



**23/02/2026**

Once the mathematical engine model was fixed, I realized a significant flaw, that the user itself is scoring the different options and  If a user wants to compare options, forcing them to manually research and input the exact scores defeats the purpose of a "Decision Companion.". So I thought by integrating an Ai llm model would be a good idea.

Alternative Approach Considered: I initially considered letting an LLM calculate the final decision entirely by providing inputs to the llm along with passing a prompt asking the AI to "choose the best option based on these weights."
Why I Rejected It: I quickly realized LLMs are notoriously unreliable when we use the same system prompt for differnt scenarios with only the inputs changed as I have experience with llms in building my final year project and also this approach would voilate the assignment's requirement that the logic must be explainable and not rely entirely on an AI model.

The Pivot (The Hybrid Architecture):
I pivoted to a multi-agent architectural approach. I decided to use the LLM strictly as a Research Agent (to gather data and estimate initial scores) and a Synthesis Agent (to explain the final math). The core calculation would remain pure mathematical model implemented in Python code.


**24/02/2026**

Implemented the project using weight sum model as the core algorithm and used python Django with simple command line interface using Cursor Ai.

I initially went with the Weighted Sum Model (WSM) for the core engine because it’s completely transparent. You multiply the score by the weight, add it up, and you're done. It easily satisfied the "explainable logic" requirement.

But while I was testing the project with different domains the results I got didn't satisfy me.Throuogh various reseach I realized it has a massive blind spot. WSM just averages things out. That means a terrible score in one critical area can get completely hidden if the other scores are high enough.

When I tested evaluating laptops for someone who travels constantly.

The Weights: Performance (9), Display (8), Build Quality (8), Battery Life (8).

I fed two hypothetical laptops into the math engine:

The one laptop which  Scored a solid 8 across the board.

Total WSM Score: 264.

And a gaming laptop Scored a perfect 10 in Performance, Display, and Build, but got a 1 in Battery.

Total WSM Score: 258.

The first laptop won, but it was way too close. I realized that if the user just increased the "Performance" weight from a 9 up to a 10, the Gaming laptop suddenly jumps to 268 and wins the entire evaluation.

That is a real-world system failure. Recommending a laptop with a 30-minute battery life to a traveler is useless. A battery score of 1 is very bad deal , and the WSM couldn't understand that.
So I thought of changing the core algorithm and started researching for it


### Discovering Fuzzy TOPSIS

Through research into MCDM methods, I found **Fuzzy TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution). Instead of single-point scores, it uses **Triangular Fuzzy Numbers** (l, m, u) — a lower bound, most likely value, and upper bound. It then computes how close each option is to the ideal solution and how far it is from the worst solution.

This solved the WSM problem: an option with an extreme weakness gets pushed far from the ideal, and its closeness coefficient drops accordingly, no matter how strong its other scores are.


**25/02/2026**

Researched for a better algorithm and decided to implement fuzzy topsis algorithm. Changed the weigh sum algorithm to fuzzy TOPSIS algorithm using cursor. Faced with many value error and fixed it, then found out that the implementation of the fuzzy normalization was wrong as it divided the three set of values generated by ai with the maximum value in the set every time but
the logic was to divide by the largest value if it is a benefit criteria like mileage for a car (better the mileage better) and divide by the lowest value for criteria like cost (lower the cost better).Made the AI to decide if a particular criteria is a cost or benefit criteria and the user is asked whether the criteria classification is satisfied by teh user if not,the user can change it

Changed the command line interface to streamlit frontend for better user experience.


**26/02/2026**

Was in the process of simultaneously building my final year college project where 
RAG was implemented so thought of implementing RAG to do local decision making where the LLM model would have no knowledge to score the options based on criterion.For example uploading resume pdfs of different students and selecting a candidate for a particular role. Implemented a global pdf upload with goal description and decided to change it to file uploads per option, which would help LLM to more efficiently score the criterions based on a particular option. 


---


## Alternative Approaches Considered

| Approach | Why Considered | Why Rejected |
|----------|---------------|--------------|

| **Weighted Sum Model** | Transparent math, easy to explain | Cannot handle extreme weaknesses; a score of 1 can be masked by high scores elsewhere |
| **Fuzzy TOPSIS** | Handles uncertainty via fuzzy numbers; penalizes extreme weaknesses; explainable math | Selected as final approach |

For the frontend:

| Approach | Why Considered | Why Rejected |
|----------|---------------|--------------|
| **Django CLI** | Quick to build, no frontend overhead | Poor UX for editing scores, no file upload support, hard to visualize results |
| **Streamlit** | Python built in,Very Simple, Had Experience | Selected — matched the project's needs perfectly |

For RAG embeddings:

| Approach | Why Considered | Why Rejected |
|----------|---------------|--------------|
| **ChromaDB default (ONNX)** | Zero config | Failed on Windows with load errors |
| **OpenAI embeddings API** | High quality | Requires API key, network dependency, cost |
| **HuggingFace SentenceTransformers (local)** | Runs locally, no API key, free, good quality | Selected — `all-MiniLM-L6-v2` provides fast inference |

---

## Refactoring Decisions

### CLI to Streamlit (commit `a59439b`)

The original CLI frontend (Django management command with `input()` loops) worked but providing inputs felt harder and CLI was implemented just as a prototype and planned to change the frontend.Adding file upload for RAG was also impossible. I replaced it with a Streamlit multipage app that provides:
- Dynamic input forms for options and criteria
- Per-option PDF file uploaders
- Editable data tables for reviewing AI scores
- Visual results with closeness coefficients

### WSM to Fuzzy TOPSIS (commit `d5c3df4`)

So I ended up replacing the entire scoring and calculation engine. This basically touched every layer of the app:
- **Models** — I added all the new schemas like `TriangularFuzzyNumber`, `NormalizedTriangularFuzzyNumber`, `WeightedTriangularFuzzyNumber`, `FuzzyOptionResult`, and `FuzzyTopsisResult`.
- **LLM prompts** — Changed it so instead of asking for a single integer score, we now request the (lower_bound, most_likely, upper_bound) as floats.
- **Graph** — Totally rewrote the calculation node. It now handles the actual math like normalization, weighting, computing the ideal solutions (FPIS/FNIS), vertex distances, and getting the final closeness coefficient.
- **Synthesis** — Updated the final explanation prompt so the AI actually breaks down how the new algorithm works step-by-step instead of just talking about simple weighted sums.

### Global PDF Upload to Per-Option Upload (commit `497c670` → `f9c1519`)

Originally there was a single file uploader for all documents. This made it impossible to associate evidence with specific options. Refactored to:
- One  per option in the Streamlit UI
- Metadata tagging in ChromaDB: each chunk tagged with `option_name`
- Filtered retrieval: when scoring option X against criterion Y, only chunks from option X's documents are retrieved

### Structured Output to Plain Text for Synthesis (commit `cd1fe32`)

The synthesis LLM call used `with_structured_output(SynthesisOutput)` which forces the Groq API to parse the response as a function call. The LLM was generating perfect explanations, but Groq's API kept rejecting them with `tool_use_failed` because the LLM wrapped the output in `<function=SynthesisOutput>` format instead of the expected format.



---

## Mistakes and Corrections

### 1. Pydantic Validation Crashes on Startup (commit `28a4d7d`)

**Mistake:** Early Pydantic model definitions had incorrect field types and validators that caused import-time crashes before the app could even start.

**Fix:** Corrected field types, added proper `validator` decorators, and used `StrictBaseModel` with `extra = "forbid"` to catch schema mismatches early.

### 2. Fuzzy Normalization Logic Error (commit `1ad6f61`)

**Mistake:** The normalization step for benefit and cost criteria was implemented with inverted logic — benefit criteria were being normalized as cost and vice versa.

**Fix:** Corrected the normalization formulas: benefit criteria divide by the maximum upper bound, cost criteria invert using the minimum lower bound.

### 3. sentence-transformers Not Installed (commit `cd1fe32`)

**Mistake:** `sentence-transformers` was listed in `requirements.txt` but was never actually installed in the virtual environment. The RAG pipeline silently failed every time — the exception was caught in `views.py` and `rag_context` was set to an empty string. The LLM received no document context whatsoever, so all scores were based purely on the LLM's general knowledge, completely ignoring uploaded PDFs.

**How I found it:** It was the most difficult which I had to solve.The llm model kept in on giving same scores for different options which meant Rag was not working properly.I first tried with changing system prompts but it didn't work. Then decide to print out the llm context which is passed to llm scoring which turned out to be empty even when the pdfs were uploaded. Then found out sentence_transformers python package is not installed.`

**Fix:** `pip install sentence-transformers` (which pulls PyTorch and related dependencies). After installation, the RAG pipeline worked immediately — PDFs were chunked, embedded, stored in ChromaDB, and retrieved per (option, criterion).



### 4. Groq `tool_use_failed` on Synthesis (commit `cd1fe32`)

**Mistake:**  The LLM generated a perfect markdown explanation, but Groq's API rejected it because the function-call format didn't match what Groq expected.

**Fix:** Replaced `llm.with_structured_output(SynthesisOutput)` with a plain `llm.invoke()` call. Since the synthesis model only needs a single string field, structured output was unnecessary overhead that introduced a point of failure.

### 6. "Run Another Decision" Not Clearing Inputs (commit `cd1fe32`)

**Mistake:** Clicking "Run another decision" cleared the internal session state (scores, results) but left all Streamlit widget values intact — option names, criteria, uploaded PDFs, and the goal text all persisted from the previous run.

**Fix:** The reset now explicitly pops all widget-managed keys (`option_name_{i}`, `option_pdf_{i}`, `criterion_name_{i}`, `criterion_weight_{i}`, `criterion_desc_{i}`, `overall_goal`) from `st.session_state` and resets counts to defaults, so Streamlit recreates every widget fresh.

---

## What Changed During Development and Why

| Date | What Changed | Why |
|------|-------------|-----|
| 23/02 | Chose WSM as core algorithm | Simplest explainable math for MCDM |
| 23/02 | Decided on hybrid LLM architecture | Full-LLM approach was unreliable and unexplainable |
| 24/02 | Built CLI + Django + WSM implementation | First working prototype |
| 24/02 | Replaced WSM with Fuzzy TOPSIS | WSM couldn't handle extreme weaknesses in critical criteria |
| 25/02 | Replaced CLI with Streamlit | CLI was unusable for editing scores and uploading files |
| 26/02 | Added RAG pipeline with ChromaDB | Needed evidence-grounded scoring from uploaded documents |
| 27/02 | Switched to per-option PDF uploads | Global upload couldn't associate evidence with specific options |
| 27/02 | Switched embeddings to local HuggingFace model | ChromaDB's default ONNX embeddings failed on Windows |
| 01/03 | Added tiered score boosting and comparative post-processing | Needed stronger guarantee that evidence = higher scores |
| 02/03 | Installed sentence-transformers (was missing) | RAG had been silently failing — PDFs were never embedded |
| 02/03 | Fixed decision reset to clear all widget state | Previous inputs persisted after clicking "Run another decision" |

---

