# Candidate Assessment: Grant Booysen

Repository: [trial-grant-booysen](https://github.com/ubundi/trial-grant-booysen)

## Rubric Evaluation

| Category | Score (0–5) | Notes |
| --- | --- | --- |
| Context Handling | 3 | Builds a FAISS vector store over user documents, but correctness of references can't be fully verified without execution |
| Agentic Thinking | 3 | Multiple response modes are implemented, yet a wiring bug prevents the selected mode from reaching the agent and one mode is missing from the UI |
| Use of Personal Data | 4 | Data set includes CV, correspondence, and personal resources providing authentic context |
| Build Quality | 2 | Modular code, but contains misrouted mode parameter, redundant imports, partial API-key logging, and documentation errors |
| Voice & Reflection | 4 | System prompt enforces first-person tone and reflections of candidate's experiences |
| Bonus Effort | 3 | Extra touches like dynamic file uploads, quick-question buttons, and resource links add polish |
| AI Build Artifacts | 3 | Prompt history and conversations demonstrate AI-assisted development, though coverage is brief |
| RAG Usage | 4 | Effective use of SentenceTransformer embeddings and FAISS for retrieval |
| Submission Completeness | 3 | Repo and hosted app provided, but no video walkthrough included and some docs have formatting issues |

**Total:** 29 / 45

## Critical Feedback and Recommendations

1. **Mode selection is disconnected from the agent.** The cached initializer expects a mode but receives a document-change counter, so the agent always runs in the default style【F:src/ui.py†L38-L66】【F:src/ui.py†L75-L82】
   - Pass both `mode` and the `docs_changed` key to `init_agent_with_uploads` and forward `mode` into `create_agent`.

2. **“Projects” response mode is defined but cannot be chosen.** The radio button omits the option, despite instructions existing elsewhere【F:src/design.py†L76-L80】
   - Add `"Projects"` to the mode list so users can explore project‑focused replies.

3. **Redundant imports and risky API‑key logging.** `rag_agent.py` imports `os`, `load_dotenv`, and `Path` twice, and prints part of the Groq key to stdout【F:src/rag_agent.py†L2-L28】
   - Consolidate imports and avoid logging secrets.

4. **README setup instructions have a broken code block.** The code fence after “Install requirements” is never closed, causing formatting to bleed into subsequent sections【F:README.md†L8-L16】
   - Add a terminating triple backtick after the `pip install` line to fix the rendering.

---

This review highlights areas where the candidate demonstrates solid initiative—such as RAG integration and thoughtful personal data—while also identifying concrete issues that need resolution for production readiness.
