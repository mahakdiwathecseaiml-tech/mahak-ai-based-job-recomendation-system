# PROJECT REPORT
## AI-BASED JOB RECOMMENDATION SYSTEM USING MACHINE LEARNING AND NATURAL LANGUAGE PROCESSING (NLP)

### 1. Introduction
The system addresses the problem of manually searching large numbers of job opportunities. It analyzes a user's resume/profile and compares it with available job descriptions to produce ranked recommendations.

### 2. Problem Statement
Job seekers may spend considerable time searching and applying to opportunities that do not match their skills, qualifications, experience or interests. The proposed system automates profile-job comparison using NLP and similarity-based matching.

### 3. Objectives
- Prepare resume/profile and job-description datasets.
- Preprocess text using NLP techniques.
- Identify skills, qualifications, experience-related terms and keywords.
- Represent text using TF-IDF.
- Calculate profile-job similarity using cosine similarity.
- Rank jobs by relevance.
- Provide relevant recommendations.
- Evaluate the system with suitable metrics.

### 4. Proposed Methodology
**User Resume/Profile → Data Collection → Data Preprocessing → Skill & Keyword Extraction using NLP → TF-IDF Feature Extraction → Similarity Calculation → Job Ranking → Recommended Jobs → Testing & Evaluation → Verification & Results**

### 5. Implementation
The web application is built with Streamlit. PDF text is extracted with pdfplumber. Pandas/openpyxl handle the Excel job dataset. Scikit-learn provides TF-IDF vectorization and cosine similarity.

### 6. Recommendation Formula
For each job:
- `text_score = cosine_similarity(profile_vector, job_vector)`
- `skill_score = matched_skills / total_job_skills`
- `match_score = 0.70 × text_score + 0.30 × skill_score`

The displayed score is normalized to a 0–100 scale.

### 7. Testing and Evaluation
The repository includes a small manually labeled verification set and `evaluate.py` to calculate Precision, Recall and F1-score. The included set is for academic verification and is not a statistically representative benchmark.

### 8. Hardware and Software Requirements
Computer/laptop, 4 GB+ RAM, 10 GB+ storage, Python, Pandas, NumPy, Scikit-learn, pdfplumber, Excel dataset and Streamlit.

### 9. Applications
Student placement systems, internships, job portals, recruitment support, college placement departments, corporate recruitment and career guidance.

### 10. Limitations
- The baseline is text/content based.
- Skill extraction uses a defined vocabulary.
- Scanned PDFs require OCR.
- Demo authentication is not production-grade.
- Saved jobs are session based.

### 11. Future Scope
Real-time job portals, advanced semantic models/BERT, multilingual resumes, location and salary preferences, real-time alerts, reciprocal candidate-employer matching and improved fairness mechanisms.

### 12. Conclusion
The implementation provides a practical application of AI, machine learning, NLP and recommender-system concepts. It converts resume/job text into comparable representations, calculates similarity, incorporates explicit skill overlap and presents ranked recommendations through a professional web interface.