# Ahoum: Conversational Personality and Behaviour Analysis Engine

## Overview

Ahoum is an AI-powered behavioural analysis system that identifies personality traits, emotional characteristics, cognitive tendencies, social behaviours, and safety-related signals from natural conversations.

Rather than relying on a single large language model call, the system uses a retrieval-driven architecture that combines semantic search, lexical search, evidence extraction, rubric-based scoring, and confidence estimation to produce transparent and explainable personality assessments.

The platform is designed to scale from a small set of behavioural facets to thousands of interpretable traits while maintaining retrieval quality and explainability.

---

## Problem Statement

Human conversations contain rich behavioural signals, but extracting them reliably is challenging.

Traditional approaches often suffer from:

* Poor explainability
* Hallucinated assessments
* Limited scalability
* Lack of evidence grounding
* Inconsistent scoring

Ahoum addresses these issues by grounding every assessment in retrieved behavioural facets and supporting evidence extracted directly from conversation text.

---

## Key Features

### Behavioural Facet Analysis

The system analyzes multiple behavioural dimensions including:

* Personality Traits
* Emotional Tendencies
* Cognitive Characteristics
* Social Behaviours
* Safety Signals

Examples:

* Risk Taking
* Compassion
* Honesty
* Curiosity
* Courage
* Assertiveness
* Empathy
* Leadership
* Integrity
* Gullibility

---

### Retrieval-Augmented Assessment

Instead of asking an LLM to reason over hundreds of traits simultaneously, Ahoum first retrieves the most relevant behavioural facets.

Benefits:

* Higher precision
* Better scalability
* Faster inference
* Improved explainability

---

### Evidence-Based Scoring

Every predicted trait is accompanied by:

* Supporting evidence span
* Behavioural rationale
* Confidence estimate

This allows reviewers to understand exactly why a score was assigned.

---

### Confidence Calibration

Predictions include confidence scores generated from:

* Retrieval quality
* Evidence strength
* Scoring consistency

This helps distinguish strong predictions from uncertain ones.

---

## System Architecture

Conversation
↓
Category Router
↓
Dense Retriever (FAISS + Sentence Transformers)
↓
BM25 Retriever
↓
Hybrid Fusion
↓
Top Relevant Facets
↓
Evidence Extractor
↓
Rubric-Based Scorer
↓
Confidence Engine

### Category Router

Identifies the most relevant behavioural categories before retrieval.

Examples:

* Personality
* Emotion
* Cognitive
* Social
* Safety

This reduces search space and improves retrieval quality.

---

### Dense Retrieval

Uses semantic embeddings to identify behaviourally similar facets even when exact keywords are absent.

Model:

* all-MiniLM-L6-v2

Vector Store:

* FAISS

---

### BM25 Retrieval

Captures exact lexical matches and domain-specific keywords that semantic retrieval may miss.

---

### Hybrid Fusion

Combines semantic and lexical retrieval signals to improve recall and ranking quality.

Benefits:

* Better coverage
* More robust retrieval
* Reduced missed facets

---

### Evidence Extraction

Locates supporting spans within conversation text that justify behavioural predictions.

Example:

Conversation:

"I quit my stable job and invested my savings into a startup."

Evidence:

"quit my stable job"
"invested my savings"

Facet:

Risk Taking

---

### Rubric-Based Scoring

Each behavioural facet contains a structured scoring rubric ranging from 1–5.

This enables:

* Consistent evaluation
* Explainable reasoning
* Human-auditable outputs

---

### Confidence Engine

Produces calibrated confidence estimates for each behavioural prediction.

Confidence is derived from:

* Retrieval relevance
* Evidence quality
* Scoring consistency

---

## Dataset Construction

A custom grounding dataset was created to evaluate retrieval and scoring quality.

The dataset contains:

* Conversations
* Ground-truth facets
* Expected scores
* Evidence spans
* Confidence labels

This enables systematic evaluation of:

* Retrieval accuracy
* Scoring quality
* Confidence calibration

---

## Evaluation

The project includes dedicated evaluation pipelines for:

### Retrieval Evaluation

Measures:

* Recall@K
* Hit Rate
* Ranking Quality

### Scoring Evaluation

Measures:

* Score Agreement
* Rubric Alignment
* Prediction Accuracy

### Confidence Evaluation

Measures:

* Calibration Quality
* Expected Calibration Error (ECE)
* Confidence Reliability

---

## Technology Stack

### Backend

* Python
* FastAPI

### Retrieval

* FAISS
* Sentence Transformers
* BM25

### LLM Layer

* Ollama
* Qwen 3 8B

### Frontend

* Streamlit

### Evaluation

* Pandas
* NumPy
* Scikit-Learn

---

## Repository Structure

src/
├── backend/
├── frontend/
├── retrieval/
├── routing/
├── evidence/
├── scoring/
├── confidence/
├── models/
└── utils/

evaluation/
├── run_retrieval_eval.py
├── run_scoring_eval.py
└── run_confidence_eval.py

grounding/
├── conversations_table.csv
└── manifest.csv

---

## Running the Project

### Install Dependencies

pip install -r requirements.txt

### Start Backend

python -m uvicorn src.backend.server:app --reload

### Start Frontend

streamlit run src/frontend/app.py

### Run Retrieval Evaluation

python evaluation/run_retrieval_eval.py

### Run Scoring Evaluation

python evaluation/run_scoring_eval.py

### Run Confidence Evaluation

python evaluation/run_confidence_eval.py

---

## Impact

Ahoum demonstrates how retrieval-augmented behavioural analysis can provide:

* Explainable personality assessment
* Evidence-grounded reasoning
* Scalable behavioural inference
* Transparent confidence estimation

The architecture is suitable for applications in:

* Conversational AI
* Behavioural Analytics
* Coaching Systems
* Interview Intelligence
* Customer Interaction Analysis
* Human-AI Alignment Research

---

## Future Work

* Cross-conversation memory
* Temporal personality tracking
* Multi-modal behaviour analysis
* Improved calibration techniques
* Domain-specific behavioural taxonomies
* Human-in-the-loop evaluation workflows

---

## Authors

Aditya Sharma

IIIT Nagpur
B.Tech Computer Science and Design
