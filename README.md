# SHL Assessment Recommendation System

An end-to-end two-phase Retrieval-Augmented Generation (RAG) recommendation system that suggests relevant SHL assessments based on a natural language query or job description. The system leverages:

* **SBERT + FAISS** for semantic retrieval
* **Google Gemini 2.0 Flash** for requirement extraction and reranking
* **FastAPI** for serving the recommendation API
* **Static Web UI** (HTML/CSS/JavaScript) for interacting with the service

https://github.com/user-attachments/assets/a457a413-c7f3-4fd8-b09e-ca3b0710f6f6

---

## 🔍 Features

1. **Phase 1 – Requirement Extraction**

   * Uses Google Gemini with few-shot prompts to parse the user’s query or job description into structured fields:

     * `test_types` (e.g. Personality & Behavior, Knowledge & Skills)
     * `job_level` (Entry-Level, Manager, etc.)
     * `max_duration` (in minutes)
     * `remote_required` (boolean)

2. **Phase 2a – Semantic Retrieval**

   * Pre-filters the catalog by extracted constraints (duration, remote, test types)
   * Builds a temporary FAISS index on the filtered subset
   * Retrieves top-300 SBERT embeddings

3. **Phase 2b – LLM Reranking**

   * Constructs a rerank prompt combining the query, few-shot examples, and candidate metadata
   * Calls Gemini’s `generate_content` to rank candidates
   * Extracts the top-10 assessment URLs

4. **Phase 2c – Post-filters & Fallbacks**

   * Re-applies duration/remote/adaptive filters
   * Ensures at least 10 recommendations via candidate fallback

5. **Static Web UI**

   * Hosted separately under `web-ui/`
   * Simple form to enter queries and display results in a table

6. **Evaluation**

   * `evaluate.py` computes Mean Recall\@3/6 and MAP\@3/6 on a JSON test set

7. **Deployment**

   * API hosted on Render: `https://shl-assessment-recommendation-system-z06m.onrender.com`
   * Web UI hosted on Render: `https://shl-assessment-website.onrender.com`

---

## 📂 Directory Structure

```
SHL_Assessment_Recommendation_System/
├── web-ui/                        # Static front-end files
│   ├── index.html                 # Main UI layout
│   ├── styles.css                 # Stylesheet
│   └── script.js                  # AJAX calls to the API
├── app.py                         # FastAPI application with two-phase RAG pipeline
├── build_embeddings.py            # Script to generate SBERT embeddings
├── build_faiss.py                 # Script to build FAISS index from embeddings
├── evaluate.py                    # Evaluation script for Recall@K and MAP@K
├── gemini_test.py                 # Quick test of Gemini API key
├── requirements.txt               # Project dependencies
├── shl_assessments_final.csv      # Raw assessment metadata CSV
├── shl_embeddings.npy             # Precomputed embeddings
├── shl_index.faiss                # Serialized FAISS index
├── shl_meta.pkl                   # Serialized pandas DataFrame of metadata
├── test_set.json                  # Test queries with ground-truth URLs
└── test_retrieval.py              # Unit tests for retrieval pipeline
```

---

## ⚙️ Setup & Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/avgvcoding/SHL_Assessment_Recommendation_System.git
   cd SHL_Assessment_Recommendation_System
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv recsys-env
   source recsys-env/bin/activate    # Linux/macOS
   recsys-env\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**

   ```bash
   export GEMINI_API_KEY="your_key_here"    # Linux/macOS
   set GEMINI_API_KEY=your_key_here          # Windows
   ```

5. **Build embeddings & index (if needed)**

   ```bash
   python build_embeddings.py
   python build_faiss.py
   ```

6. **Run the API**

   ```bash
   uvicorn app:app --reload
   ```

7. **Open the Web UI**

   * Navigate to `web-ui/index.html` in your browser or deploy it on any static host.

---

## 🚀 Usage

* **Health check:**

  ```bash
  curl GET https://shl-assessment-recommendation-system-z06m.onrender.com/health
  ```

* **Recommendation endpoint:**

  ```bash
  curl -X POST https://shl-assessment-recommendation-system-z06m.onrender.com/recommend \
    -H "Content-Type: application/json" \
    -d '{"query": "Your job description here"}'
  ```

* **Web UI:**
  Open the hosted URL, enter your query, and view recommendations in the table.

---

## 📈 Evaluation

* The `evaluate.py` script calculates Mean Recall\@K and Mean Average Precision\@K against `test_set.json`.
* Modify `K` or add new test cases in `test_set.json` to measure performance on additional queries.

---


**Author:** Aviral | **Repository:** [https://github.com/avgvcoding/SHL\_Assessment\_Recommendation\_System](https://github.com/avgvcoding/SHL_Assessment_Recommendation_System)

Feel free to contribute or raise issues—happy hiring!
