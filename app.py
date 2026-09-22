import os
import re
import hashlib
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JOBS_PATH_1 = os.path.join(BASE_DIR, "dataset", "jobs.xlsx")
JOBS_PATH_2 = os.path.join(BASE_DIR, "jobs.xlsx")

USERS_FILE = os.path.join(BASE_DIR, "users.csv")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.16), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(59,130,246,0.12), transparent 25%),
            #080d1c;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101936 0%, #0b1125 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .brand {
        font-size: 26px;
        font-weight: 800;
        padding: 10px 0 4px 0;
    }

    .brand span {
        color: #8b5cf6;
    }

    .tagline {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 25px;
    }

    .hero {
        padding: 38px;
        border-radius: 28px;
        background:
            linear-gradient(
                135deg,
                rgba(30,41,90,0.96),
                rgba(45,32,92,0.90)
            );
        border: 1px solid rgba(139,92,246,0.25);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        margin-bottom: 24px;
    }

    .hero-small {
        color: #a5b4fc;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .hero h1 {
        font-size: 48px;
        line-height: 1.05;
        margin: 12px 0;
    }

    .gradient-text {
        background: linear-gradient(
            90deg,
            #60a5fa,
            #8b5cf6,
            #c084fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .muted {
        color: #94a3b8;
        font-size: 15px;
    }

    .glass {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.16);
    }

    .glass h3 {
        margin-top: 0;
        margin-bottom: 8px;
    }

    .stat-card {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
    }

    .stat-title {
        color: #94a3b8;
        font-size: 13px;
    }

    .stat-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .job-card {
        background: rgba(15,23,42,0.86);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 20px;
        padding: 22px;
        margin: 14px 0;
    }

    .job-title {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
    }

    .company {
        color: #a5b4fc;
        font-weight: 600;
        margin-top: 4px;
    }

    .job-meta {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 10px;
    }

    .match {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(34,197,94,0.14);
        color: #86efac;
        font-weight: 800;
        font-size: 14px;
    }

    .skill {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 4px 4px 0;
        border-radius: 999px;
        background: rgba(99,102,241,0.16);
        color: #c7d2fe;
        border: 1px solid rgba(99,102,241,0.2);
        font-size: 12px;
    }

    .missing {
        background: rgba(239,68,68,0.13);
        color: #fca5a5;
        border-color: rgba(239,68,68,0.2);
    }

    .section {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .sub {
        color: #94a3b8;
        margin-bottom: 25px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = True

if "username" not in st.session_state:
    st.session_state.username = "demo"

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "saved_jobs" not in st.session_state:
    st.session_state.saved_jobs = []


# =========================================================
# DATASET
# =========================================================

def create_sample_dataset():

    os.makedirs(os.path.dirname(JOBS_PATH_1), exist_ok=True)

    data = [
        {
            "Job Title": "Python Developer",
            "Company": "Tech Solutions",
            "Location": "Nagpur",
            "Skills": "Python, Pandas, NumPy, Scikit-learn, SQL, Flask, Git",
            "Education": "BTech CSE",
            "Experience": "0-2 years",
            "Description": "Develop Python applications and APIs."
        },
        {
            "Job Title": "Machine Learning Engineer",
            "Company": "InnovateAI",
            "Location": "Remote",
            "Skills": "Python, Machine Learning, TensorFlow, Pandas, SQL",
            "Education": "BTech CSE AIML",
            "Experience": "2-5 years",
            "Description": "Build and evaluate machine learning pipelines."
        },
        {
            "Job Title": "Data Analyst",
            "Company": "Analytics Hub",
            "Location": "Pune",
            "Skills": "SQL, Excel, Pandas, Power BI, Python, Statistics",
            "Education": "BTech / BSc",
            "Experience": "0-3 years",
            "Description": "Analyze datasets and build business dashboards."
        },
        {
            "Job Title": "NLP Engineer",
            "Company": "Language AI",
            "Location": "Bengaluru",
            "Skills": "Python, NLP, Transformers, BERT, Machine Learning",
            "Education": "BTech CSE AIML",
            "Experience": "1-4 years",
            "Description": "Develop natural language processing solutions."
        },
        {
            "Job Title": "AI Intern",
            "Company": "FutureMind Labs",
            "Location": "Nagpur",
            "Skills": "Python, Machine Learning, NLP, Pandas, NumPy",
            "Education": "BTech CSE AIML",
            "Experience": "0-1 years",
            "Description": "Assist with AI experiments and data preparation."
        },
        {
            "Job Title": "Software Engineer",
            "Company": "CloudNova",
            "Location": "Hyderabad",
            "Skills": "Java, Python, SQL, Git, REST API, Docker",
            "Education": "BTech CSE IT",
            "Experience": "1-4 years",
            "Description": "Build scalable software services."
        },
        {
            "Job Title": "Data Scientist",
            "Company": "InsightWorks",
            "Location": "Mumbai",
            "Skills": "Python, SQL, Pandas, Machine Learning, Statistics",
            "Education": "BTech / MSc",
            "Experience": "1-4 years",
            "Description": "Create predictive models and analytical solutions."
        },
        {
            "Job Title": "Computer Vision Engineer",
            "Company": "VisionForge",
            "Location": "Pune",
            "Skills": "Python, OpenCV, Deep Learning, PyTorch",
            "Education": "BTech / MTech",
            "Experience": "2-5 years",
            "Description": "Develop computer vision models and pipelines."
        }
    ]

    df = pd.DataFrame(data)
    df.to_excel(JOBS_PATH_1, index=False)


@st.cache_data
def load_jobs():

    path = None

    if os.path.exists(JOBS_PATH_1):
        path = JOBS_PATH_1
    elif os.path.exists(JOBS_PATH_2):
        path = JOBS_PATH_2

    if path is None:
        create_sample_dataset()
        path = JOBS_PATH_1

    return pd.read_excel(path)


jobs = load_jobs()


# =========================================================
# TEXT PROCESSING
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#.\-/ ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


SKILLS = [
    "python",
    "java",
    "sql",
    "excel",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "nlp",
    "bert",
    "transformers",
    "opencv",
    "statistics",
    "power bi",
    "flask",
    "django",
    "git",
    "docker",
    "rest api",
    "html",
    "css",
    "javascript",
    "react",
    "data analysis",
]


def extract_skills(text):

    text = clean_text(text)

    found = []

    for skill in SKILLS:

        if skill.lower() in text:
            found.append(skill)

    return sorted(set(found))


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):

    try:

        import pdfplumber

        with pdfplumber.open(uploaded_file) as pdf:

            pages = []

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    pages.append(page_text)

            return "\n".join(pages)

    except Exception as e:

        raise Exception(
            f"Could not extract PDF text: {e}"
        )


# =========================================================
# RECOMMENDATION MODEL
# =========================================================

def prepare_job_text(row):

    parts = [
        row.get("Job Title", ""),
        row.get("Company", ""),
        row.get("Location", ""),
        row.get("Skills", ""),
        row.get("Education", ""),
        row.get("Experience", ""),
        row.get("Description", "")
    ]

    return " ".join(
        clean_text(x)
        for x in parts
        if pd.notna(x)
    )


def recommend(resume_text, job_df):

    resume_clean = clean_text(resume_text)

    if not resume_clean.strip():
        return pd.DataFrame()

    job_texts = [
        prepare_job_text(row)
        for _, row in job_df.iterrows()
    ]

    documents = [resume_clean] + job_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=7000
    )

    matrix = vectorizer.fit_transform(documents)

    similarities = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()

    resume_skills = set(
        extract_skills(resume_text)
    )

    output = job_df.copy()

    output["Text Similarity"] = similarities

    match_scores = []
    matched_skills = []
    missing_skills = []

    for _, row in output.iterrows():

        job_skills = set(
            extract_skills(
                str(row.get("Skills", ""))
            )
        )

        matched = sorted(
            resume_skills.intersection(job_skills)
        )

        missing = sorted(
            job_skills.difference(resume_skills)
        )

        if len(job_skills) > 0:
            skill_score = (
                len(matched) / len(job_skills)
            )
        else:
            skill_score = 0

        final_score = (
            0.70 * float(
                similarities[len(match_scores)]
            )
            +
            0.30 * skill_score
        )

        match_scores.append(
            round(final_score * 100, 2)
        )

        matched_skills.append(
            ", ".join(matched)
        )

        missing_skills.append(
            ", ".join(missing)
        )

    output["Skill Match"] = [
        round(
            len(set(extract_skills(resume_text)).intersection(
                set(extract_skills(str(x)))
            ))
            /
            max(len(extract_skills(str(x))), 1)
            * 100,
            2
        )
        for x in output["Skills"]
    ]

    output["Match Score"] = match_scores
    output["Matched Skills"] = matched_skills
    output["Missing Skills"] = missing_skills

    output = output.sort_values(
        "Match Score",
        ascending=False
    ).reset_index(drop=True)

    return output


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            CareerMatch <span>AI</span>
        </div>
        <div class="tagline">
            Your Skills + Our AI = Your Future
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    if st.button(
        "🏠 Home",
        use_container_width=True
    ):
        st.session_state.page = "Dashboard"
        st.rerun()

    if st.button(
        "🔎 Find Jobs",
        use_container_width=True
    ):
        st.session_state.page = "Recommendations"
        st.rerun()

    if st.button(
        "📄 Resume Studio",
        use_container_width=True
    ):
        st.session_state.page = "Resume Studio"
        st.rerun()

    if st.button(
        "💾 Saved Jobs",
        use_container_width=True
    ):
        st.session_state.page = "Saved Jobs"
        st.rerun()

    if st.button(
        "👤 My Profile",
        use_container_width=True
    ):
        st.session_state.page = "Profile"
        st.rerun()

    if st.button(
        "📊 Analytics",
        use_container_width=True
    ):
        st.session_state.page = "Analytics"
        st.rerun()

    if st.button(
        "ℹ️ About",
        use_container_width=True
    ):
        st.session_state.page = "About"
        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        <div class="muted">
        <b>AI Matching</b><br>
        70% TF-IDF text similarity<br>
        30% skill overlap
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

top1, top2 = st.columns([5, 1])

with top1:

    st.markdown(
        f"""
        <div class="muted">
        Welcome back, <b>{st.session_state.username}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

with top2:

    if st.button(
        "Resume →",
        use_container_width=True
    ):
        st.session_state.page = "Resume Studio"
        st.rerun()


page = st.session_state.page


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-small">
        AI + MACHINE LEARNING + NLP
        </div>

        <h1>
        AI-Powered<br>
        <span class="gradient-text">
        Job Recommendation System
        </span>
        </h1>

        <p class="muted">
        Upload your resume or enter your profile and let our AI
        find job opportunities that match your skills,
        education, experience and interests.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="stat-card">
            <div class="stat-title">Smart Matching</div>
            <div class="stat-value">NLP</div>
            <div class="muted">Text analysis</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="stat-card">
            <div class="stat-title">Recommendation</div>
            <div class="stat-value">AI</div>
            <div class="muted">Personalized jobs</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="stat-card">
            <div class="stat-title">Matching Model</div>
            <div class="stat-value">TF-IDF</div>
            <div class="muted">Feature extraction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="stat-card">
            <div class="stat-title">Similarity</div>
            <div class="stat-value">Cosine</div>
            <div class="muted">Job ranking</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown(
            """
            <div class="glass">

            <h3>🚀 Quick Start</h3>

            <p class="muted">
            Start with your resume to generate a personalized
            recommendation list.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Resume Studio →",
            type="primary",
            use_container_width=True,
            key="dashboard_resume"
        ):
            st.session_state.page = "Resume Studio"
            st.rerun()

    with right:

        st.markdown(
            """
            <div class="glass">

            <h3>🔬 Model Transparency</h3>

            <p class="muted">
            Recommendations use 70% text similarity and
            30% skill overlap.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# RESUME STUDIO
# =========================================================

elif page == "Resume Studio":

    st.markdown(
        '<div class="section">Resume Studio</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub">
        Upload your resume or paste profile text, then run the AI analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.2, 0.8],
        gap="large"
    )

    with left:

        st.markdown(
            """
            <div class="glass">
            <h3>📄 Upload Resume</h3>
            """,
            unsafe_allow_html=True
        )

        pdf = st.file_uploader(
            "Upload resume PDF",
            type=["pdf"],
            key="resume_pdf"
        )

        pasted = st.text_area(
            "Or paste resume / profile text",
            height=220,
            placeholder=(
                "Example: B.Tech CSE AIML student with "
                "Python, SQL, Pandas, NumPy, "
                "Machine Learning and NLP skills."
            )
        )

        text_value = pasted or ""

        if pdf:

            try:

                extracted = extract_pdf_text(pdf)

                if extracted.strip(
