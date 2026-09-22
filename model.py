from pathlib import Path
import re
import hashlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE=Path(__file__).parent
USERS=BASE/'users.csv'
JOBS=BASE/'dataset'/'jobs.xlsx'
SKILLS=['python','java','c++','javascript','typescript','sql','mysql','postgresql','mongodb','html','css','react','angular','node.js','django','flask','fastapi','pandas','numpy','scikit-learn','tensorflow','pytorch','machine learning','deep learning','nlp','natural language processing','transformers','bert','computer vision','opencv','data analysis','data visualization','power bi','tableau','excel','git','github','docker','aws','azure','spark','statistics','matplotlib','seaborn','rest api','linux']
ALIASES={'natural language processing':'nlp','sklearn':'scikit-learn','scikit learn':'scikit-learn','powerbi':'power bi','restful api':'rest api'}

def sha(x): return hashlib.sha256(x.encode()).hexdigest()
def clean(x):
    t=str(x or '').lower()
    for a,b in ALIASES.items(): t=t.replace(a,b)
    t=re.sub(r'[^a-z0-9+#.\-/ ]+',' ',t)
    return re.sub(r'\s+',' ',t).strip()
def extract_skills(x):
    t=clean(x)
    found=set()
    for s in SKILLS:
        if re.search(r'(?<![a-z0-9])'+re.escape(s)+r'(?![a-z0-9])',t): found.add(ALIASES.get(s,s))
    return sorted(found)

def seed():
    if not USERS.exists(): pd.DataFrame([{'username':'demo','password_hash':sha('demo123'),'name':'Demo User'}]).to_csv(USERS,index=False)
    if not JOBS.exists():
        rows=[
        ['Python Developer','Tech Solutions','Nagpur','Python, Pandas, NumPy, Scikit-learn, SQL, Flask, Git','BTech CSE','0-2 years','Develop Python applications and APIs.'],
        ['Machine Learning Engineer','InnovateAI','Remote','Python, Machine Learning, TensorFlow, Pandas, SQL','BTech CSE AIML','2-5 years','Build and evaluate machine learning pipelines.'],
        ['Data Analyst','Analytics Hub','Pune','SQL, Excel, Pandas, Power BI, Python, Statistics','BTech / BSc','0-3 years','Analyze datasets and build business dashboards.'],
        ['NLP Engineer','Language AI','Bengaluru','Python, NLP, Transformers, BERT, Machine Learning','BTech CSE AIML','1-4 years','Develop natural language processing solutions.'],
        ['AI Intern','FutureMind Labs','Nagpur','Python, Machine Learning, NLP, Pandas, NumPy','BTech CSE AIML','0-1 years','Assist with AI experiments and data preparation.'],
        ['Software Engineer','CloudNova','Hyderabad','Java, Python, SQL, Git, REST API, Docker','BTech CSE IT','1-4 years','Build scalable software services.'],
        ['Data Scientist','InsightWorks','Mumbai','Python, SQL, Pandas, Machine Learning, Statistics','BTech / MSc','1-4 years','Create predictive models and analytical solutions.'],
        ['Computer Vision Engineer','VisionForge','Pune','Python, OpenCV, Deep Learning, PyTorch','BTech/MTech','2-5 years','Develop computer vision models and pipelines.'],
        ['AI/ML Intern','NeuralWorks','Remote','Python, NumPy, Pandas, Machine Learning, Git','BTech CSE AIML','0-1 years','Support model experiments, preprocessing and evaluation.'],
        ['Backend Developer','StackForge','Bengaluru','Python, FastAPI, SQL, Docker, REST API, Git','BTech CSE IT','1-3 years','Develop backend services and APIs.']]
        pd.DataFrame(rows,columns=['job_title','company','location','skills','education','experience','description']).to_excel(JOBS,index=False)

def load_jobs(upload=None):
    df=pd.read_excel(upload if upload is not None else JOBS)
    required=['job_title','company','location','skills','education','experience','description']
    for c in required:
        if c not in df.columns: df[c]=''
        df[c]=df[c].fillna('').astype(str)
    df['combined']=df.apply(lambda r:clean(' '.join(r[c] for c in required)),axis=1)
    return df

def extract_pdf(file):
    with pdfplumber.open(file) as pdf: return '\n'.join((p.extract_text() or '') for p in pdf.pages)

def recommend(resume,jobs,top_n=10):
    texts=[clean(resume)]+jobs['combined'].tolist()
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=7000)
    matrix=vectorizer.fit_transform(texts)
    sim=cosine_similarity(matrix[0:1],matrix[1:]).ravel()
    resume_skills=set(extract_skills(resume)); out=[]
    for i,row in jobs.iterrows():
        job_skills=set(extract_skills(row['skills']+' '+row['description']))
        matched=sorted(resume_skills & job_skills); missing=sorted(job_skills-resume_skills)
        skill_score=len(matched)/max(len(job_skills),1)
        score=(0.70*float(sim[i])+0.30*skill_score)*100
        out.append({**row.to_dict(),'match_score':round(score,1),'text_score':round(float(sim[i])*100,1),'skill_score':round(skill_score*100,1),'matched':matched,'missing':missing})
    return pd.DataFrame(out).sort_values(['match_score','skill_score'],ascending=False).head(top_n).reset_index(drop=True)



def extract_pdf_text(file):
    import pdfplumber
    with pdfplumber.open(file) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)
