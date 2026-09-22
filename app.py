import hashlib, re
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from model import SKILLS, sha, clean, extract_skills, load_jobs, recommend, extract_pdf_text

st.set_page_config(page_title='CareerMatch AI', page_icon='✦', layout='wide', initial_sidebar_state='expanded')
BASE=Path(__file__).parent
USERS=BASE/'users.csv'; JOBS=BASE/'dataset'/'jobs.xlsx'


def seed_data():
    if not USERS.exists():
        pd.DataFrame([{'username':'demo','password_hash':sha('demo123'),'name':'Demo User'}]).to_csv(USERS,index=False)
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

def badge(score):
    if score>=80: return 'Excellent match'
    if score>=65: return 'Strong match'
    if score>=50: return 'Potential match'
    return 'Explore match'

def chips(items,cls='skill'):
    return ''.join(f'<span class="chip {cls}">{i}</span>' for i in items)

seed_data()

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#07111f;--navy2:#0c1730;--violet:#7c3aed;--violet2:#a855f7;--ink:#e9eefb;--muted:#9aa8c2;--line:#23304a;--card:#0e1a2e;--card2:#111f36;}
html,body,[class*="css"]{font-family:Inter,sans-serif} .stApp{background:radial-gradient(circle at 80% 0%,#251252 0%,#0a1425 38%,#07111f 100%);color:var(--ink)}
.block-container{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}.stMarkdown,.stText,.stCaption{color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#050b15,#0d1530 70%,#17103a);border-right:1px solid #202c46}
[data-testid="stSidebar"] *{color:#eaf0ff!important}.stRadio>div{gap:8px}.stRadio label{padding:9px 10px;border-radius:12px}.stRadio label:hover{background:#ffffff0d}
.brand{padding:10px 8px 24px}.brand h2{margin:0;font-size:22px;letter-spacing:-.5px}.brand p{margin:4px 0;color:#9aa8c2;font-size:11px!important}.brandmark{display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;border-radius:11px;background:linear-gradient(135deg,#7c3aed,#a855f7);box-shadow:0 8px 25px #7c3aed55;margin-right:8px}
.hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#101d35dd,#24124fe8 62%,#4c1d95e8);border:1px solid #6d5ad755;border-radius:28px;padding:36px 38px;box-shadow:0 24px 70px #00000035;margin-bottom:22px}.hero:after{content:'';position:absolute;width:320px;height:320px;border-radius:50%;background:#a855f722;right:-90px;top:-130px;filter:blur(8px)}
.eyebrow{font-size:11px;letter-spacing:1.5px;font-weight:800;color:#c4b5fd;text-transform:uppercase}.hero h1{font-size:42px;line-height:1.08;margin:8px 0 12px;font-weight:800;letter-spacing:-1.4px}.hero p{max-width:720px;color:#c5d0e7;font-size:15px;line-height:1.7;margin:0}.glass{background:linear-gradient(145deg,#12213adf,#0b1729e8);border:1px solid #2b3a56;border-radius:22px;padding:22px;box-shadow:0 16px 50px #00000022}.section{font-size:24px;font-weight:800;letter-spacing:-.6px;margin:8px 0 3px}.sub{color:#94a3bd;font-size:13px;margin-bottom:18px}.metric{background:linear-gradient(145deg,#102039,#0c1728);border:1px solid #263650;border-radius:18px;padding:18px}.metric .label{font-size:11px;color:#8797b5;text-transform:uppercase;letter-spacing:.8px}.metric .value{font-size:28px;font-weight:800;margin-top:4px}.metric .hint{font-size:11px;color:#7788a7;margin-top:3px}
.job{background:linear-gradient(145deg,#0e1b2f,#0b1628);border:1px solid #273650;border-radius:20px;padding:20px;margin:12px 0;box-shadow:0 12px 35px #0000001f}.job:hover{border-color:#6d5ad7;box-shadow:0 15px 45px #7c3aed18}.jobtitle{font-size:18px;font-weight:800}.company{color:#b794f6;font-size:13px;font-weight:700;margin-top:3px}.muted{color:#8494b0;font-size:12px}.scorebox{text-align:right}.score{font-size:32px;font-weight:900;background:linear-gradient(90deg,#c4b5fd,#a78bfa,#f0abfc);-webkit-background-clip:text;color:transparent}.matchtag{display:inline-block;font-size:10px;color:#c4b5fd;background:#7c3aed18;border:1px solid #7c3aed44;padding:5px 8px;border-radius:999px;margin-top:3px}.chip{display:inline-block;padding:6px 9px;border-radius:999px;margin:3px 4px 0 0;font-size:10px;font-weight:700}.skill{background:#171f36;color:#c4b5fd;border:1px solid #303c5b}.ok{background:#0b2a24;color:#5eead4;border:1px solid #164e46}.miss{background:#2a1a15;color:#fdba74;border:1px solid #63351f}
.dropzone{border:1.5px dashed #5b6c8d;border-radius:18px;padding:36px;text-align:center;background:#0a1527}.mini{font-size:12px;color:#90a0bb}.feature{padding:15px;border:1px solid #283650;border-radius:16px;background:#0b1729}.feature b{display:block;margin-bottom:4px}.footer{text-align:center;color:#687995;font-size:11px;padding:28px}
.stButton>button{border-radius:12px;border:1px solid #394765;background:#121f35;color:#eef2ff;font-weight:700}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#6d28d9,#8b5cf6);border:0;box-shadow:0 10px 28px #7c3aed35}.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div,.stSlider{background:#0c1728;color:#eef2ff;border-color:#2b3a56}.stFileUploader section{background:#0b1729;border:1px dashed #455675}.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{background:#0e1b2f;border-radius:10px;color:#9aa8c2}.stTabs [aria-selected="true"]{color:#c4b5fd;background:#171f36}.dataframe{border:1px solid #283650}
</style>''',unsafe_allow_html=True)

if 'logged_in' not in st.session_state: st.session_state.logged_in=False
if 'saved' not in st.session_state: st.session_state.saved=[]
if 'results' not in st.session_state: st.session_state.results=None

if not st.session_state.logged_in:
    st.markdown('<div style="height:7vh"></div>',unsafe_allow_html=True)
    left,right=st.columns([1.25,.75],gap='large')
    with left:
        st.markdown('''<div class="hero"><div class="eyebrow">AI-powered career discovery</div><h1>Turn your resume into your next opportunity.</h1><p>Upload your resume, extract your skills and compare your profile with job descriptions using NLP, TF-IDF and cosine similarity.</p><div style="margin-top:22px"><span class="chip skill">NLP</span><span class="chip skill">TF-IDF</span><span class="chip skill">Cosine Similarity</span><span class="chip skill">Skill Matching</span></div></div>''',unsafe_allow_html=True)
        a,b,c=st.columns(3)
        for col,icon,title,desc in [(a,'✦','Smart matching','NLP + ML'),(b,'◎','Skill insights','Matched + missing skills'),(c,'↗','Ranked results','Transparent match scores')]:
            col.markdown(f'<div class="feature"><div style="font-size:22px">{icon}</div><b>{title}</b><span class="mini">{desc}</span></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass">',unsafe_allow_html=True); st.markdown('### Welcome back')
        tab1,tab2=st.tabs(['Sign in','Create account'])
        with tab1:
            u=st.text_input('Username',key='login_u'); p=st.text_input('Password',type='password',key='login_p')
            if st.button('Sign in',type='primary',use_container_width=True):
                users=pd.read_csv(USERS); ok=((users.username.str.lower()==u.strip().lower())&(users.password_hash==sha(p))).any()
                if ok: st.session_state.logged_in=True; st.session_state.user=u.strip(); st.rerun()
                else: st.error('Invalid username or password.')
            st.caption('Demo account: demo / demo123')
        with tab2:
            name=st.text_input('Full name',key='reg_name'); nu=st.text_input('Username',key='reg_u'); np=st.text_input('Password',type='password',key='reg_p')
            if st.button('Create account',use_container_width=True):
                users=pd.read_csv(USERS)
                if len(np)<6: st.warning('Use at least 6 characters.')
                elif not nu.strip(): st.warning('Enter a username.')
                elif nu.lower() in users.username.str.lower().tolist(): st.error('Username already exists.')
                else:
                    new={'username':nu.strip(),'password_hash':sha(np),'name':name.strip() or nu.strip()}; pd.concat([users,pd.DataFrame([new])],ignore_index=True).to_csv(USERS,index=False); st.success('Account created. Sign in to continue.')
        st.markdown('</div>',unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand"><h2><span class="brandmark">✦</span>CareerMatch AI</h2><p>Your skills + our AI = smarter discovery</p></div>',unsafe_allow_html=True)
    page=st.radio('Workspace',['Overview','Resume Studio','Job Matches','Saved Jobs','My Profile','Analytics','Settings'])
    st.divider(); st.caption('SIGNED IN AS'); st.write(st.session_state.get('user','User'))
    if st.button('Log out',use_container_width=True): st.session_state.logged_in=False; st.rerun()

upload=st.session_state.get('dataset_upload')
if page=='Settings':
    up=st.file_uploader('Replace job dataset for this session',type=['xlsx','xls'],help='Required columns: job_title, company, location, skills, education, experience, description')
    if up: st.session_state.dataset_upload=up; upload=up
jobs=load_jobs(upload)

if page=='Overview':
    top=st.session_state.results
    st.markdown('<div class="hero"><div class="eyebrow">Career intelligence workspace</div><h1>Build a better job search, one match at a time.</h1><p>Analyze your profile, understand your strongest skills and explore ranked opportunities with an explainable content-based recommendation pipeline.</p></div>',unsafe_allow_html=True)
    m=st.columns(4)
    vals=[(len(jobs),'Jobs indexed','Current dataset'),(len(SKILLS),'Skills detected','Vocabulary'),(f"{top.iloc[0].match_score:.0f}%" if top is not None and len(top) else '—','Top AI match','Latest analysis'),(len(st.session_state.saved),'Saved jobs','This session')]
    for col,(v,l,hint) in zip(m,vals): col.markdown(f'<div class="metric"><div class="label">{l}</div><div class="value">{v}</div><div class="hint">{hint}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Your AI workflow</div><div class="sub">A transparent pipeline aligned with the project methodology.</div>',unsafe_allow_html=True)
    a,b=st.columns([1.15,.85],gap='large')
    with a:
        st.markdown('<div class="glass"><h3>01 · Resume → Profile</h3><p class="muted">Upload a PDF or paste profile text. The system extracts text and identifies skills, education, experience and job-related keywords.</p><h3>02 · Profile → Job vectors</h3><p class="muted">TF-IDF transforms the profile and job descriptions into numerical feature vectors.</p><h3>03 · Similarity → Ranking</h3><p class="muted">Cosine similarity is combined with explicit skill overlap to produce a transparent match score and ranked recommendations.</p></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="glass"><h3>Quick start</h3><p class="muted">Start with your resume to generate the recommendation list.</p></div>',unsafe_allow_html=True)
        if st.button('Open Resume Studio →',type='primary',use_container_width=True): st.session_state.next_page='Resume Studio'; st.rerun()
        st.markdown('<div style="height:8px"></div><div class="glass"><h3>Model transparency</h3><p class="muted">70% text similarity + 30% skill overlap. This is a content-based baseline, not a hiring decision.</p></div>',unsafe_allow_html=True)

elif page=='Resume Studio':
    st.markdown('<div class="section">Resume Studio</div><div class="sub">Upload your resume or paste profile text, then run the AI matching pipeline.</div>',unsafe_allow_html=True)
    left,right=st.columns([1.2,.8],gap='large')
    with left:
        st.markdown('<div class="glass">',unsafe_allow_html=True)
        pdf=st.file_uploader('Upload resume (PDF)',type=['pdf'])
        pasted=st.text_area('Or paste resume / profile text',height=220,placeholder='Example: B.Tech CSE AIML student with Python, SQL, Pandas, Machine Learning, NLP...')
        text=pasted
        if pdf:
            try: text=(extract_pdf_text(pdf)+'\n'+pasted).strip(); st.success('Resume text extracted successfully.')
            except Exception as e: st.error(f'Could not extract PDF text: {e}')
        if st.button('✦ Analyze & Find Jobs',type='primary',use_container_width=True):
            if text.strip(): st.session_state.resume=text; st.session_state.results=recommend(text,jobs); st.success('Analysis complete. Open Job Matches to explore the ranking.')
            else: st.warning('Upload a PDF or enter profile text first.')
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        sk=extract_skills(st.session_state.get('resume',''))
        st.markdown(f'<div class="glass"><h3>Skill intelligence</h3><p class="muted">Detected from the latest profile text.</p><div>{chips(sk) if sk else "<span class=mini>No skills detected yet.</span>"}</div></div>',unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div><div class="glass"><h3>What the model uses</h3><p class="muted">Text normalization · skill/keyword extraction · TF-IDF · cosine similarity · skill overlap · ranked output</p></div>',unsafe_allow_html=True)

elif page=='Job Matches':
    st.markdown('<div class="section">Job Matches</div><div class="sub">Ranked from the latest profile analysis. Use filters to inspect the result set.</div>',unsafe_allow_html=True)
    r=st.session_state.results
    if r is None: st.info('Run an analysis in Resume Studio first.')
    else:
        f1,f2,f3=st.columns([1,1,1]); loc=f1.selectbox('Location',['All']+sorted(r.location.unique().tolist())); minimum=f2.slider('Minimum match',0,100,0); limit=f3.slider('Show jobs',1,len(r),len(r))
        view=r[(r.location==loc) if loc!='All' else pd.Series(True,index=r.index)]; view=view[view.match_score>=minimum].head(limit)
        for idx,x in view.iterrows():
            saved=x.job_title in st.session_state.saved
            st.markdown(f'''<div class="job"><div style="display:flex;justify-content:space-between;gap:20px"><div><div class="jobtitle">{x.job_title}</div><div class="company">{x.company}</div><div class="muted">⌖ {x.location} &nbsp;·&nbsp; {x.education} &nbsp;·&nbsp; {x.experience}</div></div><div class="scorebox"><div class="score">{x.match_score:.0f}%</div><div class="matchtag">{badge(x.match_score)}</div></div></div><p class="muted" style="margin:14px 0">{x.description}</p><div>{chips(x.matched[:8],'ok')}</div><div>{chips(x.missing[:6],'miss')}</div></div>''',unsafe_allow_html=True)
            if st.button('★ Saved' if saved else '☆ Save job',key=f'save_{idx}'):
                if saved: st.session_state.saved.remove(x.job_title)
                else: st.session_state.saved.append(x.job_title)
                st.rerun()

elif page=='Saved Jobs':
    st.markdown('<div class="section">Saved Jobs</div><div class="sub">Jobs you bookmarked during this session.</div>',unsafe_allow_html=True)
    if not st.session_state.saved: st.info('No saved jobs yet.')
    for title in st.session_state.saved:
        row=jobs[jobs.job_title==title]
        if not row.empty:
            x=row.iloc[0]; st.markdown(f'<div class="job"><div class="jobtitle">{x.job_title}</div><div class="company">{x.company}</div><div class="muted">⌖ {x.location} · {x.experience}</div><p class="muted">{x.description}</p><div>{chips(extract_skills(x.skills),'skill')}</div></div>',unsafe_allow_html=True)

elif page=='My Profile':
    st.markdown('<div class="section">My Profile</div><div class="sub">A concise view of the information currently available to the recommendation engine.</div>',unsafe_allow_html=True)
    resume=st.session_state.get('resume',''); sk=extract_skills(resume); strength=min(100,35+len(sk)*6)
    a,b=st.columns([1.05,.95],gap='large')
    with a: st.markdown(f'<div class="glass"><div class="eyebrow">Career profile</div><h2>{st.session_state.get("user","User")}</h2><p class="muted">Profile coverage score: {strength}%</p><div style="height:8px;background:#1c2940;border-radius:99px"><div style="width:{strength}%;height:8px;border-radius:99px;background:linear-gradient(90deg,#7c3aed,#c084fc)"></div></div><h4 style="margin-top:22px">Detected skills</h4>{chips(sk) if sk else "<span class=mini>Analyze a resume to populate your profile.</span>"}</div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="glass"><h3>Profile improvement checklist</h3><p class="muted">Include education, relevant skills, projects, certifications and experience in your resume/profile so the text-matching pipeline has more evidence to compare.</p><ul class="muted"><li>Use consistent skill names.</li><li>Describe project work with technologies.</li><li>Include relevant experience and interests.</li><li>Keep the resume text machine-readable.</li></ul></div>',unsafe_allow_html=True)

elif page=='Analytics':
    st.markdown('<div class="section">Match Analytics</div><div class="sub">Transparent diagnostics for the latest recommendation run.</div>',unsafe_allow_html=True)
    r=st.session_state.results
    if r is None: st.info('Run an analysis first.')
    else:
        a,b,c,d=st.columns(4); a.metric('Top match',f'{r.match_score.max():.0f}%'); b.metric('Average',f'{r.match_score.mean():.0f}%'); c.metric('Jobs ranked',len(r)); d.metric('Matched skills',sum(len(x) for x in r.matched))
        st.markdown('<div class="glass">',unsafe_allow_html=True); st.bar_chart(r.set_index('job_title')[['match_score']]); st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div><div class="glass">',unsafe_allow_html=True); st.dataframe(r[['job_title','company','location','match_score','text_score','skill_score']],use_container_width=True,hide_index=True); st.markdown('</div>',unsafe_allow_html=True)

elif page=='Settings':
    st.markdown('<div class="section">Settings</div><div class="sub">Configure the dataset and review deployment notes.</div>',unsafe_allow_html=True)
    st.markdown('<div class="glass"><h3>Dataset schema</h3><p class="muted">Excel columns: <b>job_title, company, location, skills, education, experience, description</b>. The uploaded dataset is used for the current Streamlit session.</p></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div><div class="glass"><h3>Project methodology</h3><p class="muted">Data Collection → Data Preprocessing → Skill & Keyword Extraction using NLP → TF-IDF Feature Extraction → Similarity Calculation → Job Ranking → Testing & Evaluation → Verification & Results.</p></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div><div class="glass"><h3>Deployment</h3><p class="muted">Run locally with <b>pip install -r requirements.txt</b> and <b>streamlit run app.py</b>. For Streamlit Community Cloud, push the project to GitHub and deploy the repository entry point.</p></div>',unsafe_allow_html=True)

st.markdown('<div class="footer">AI-Based Job Recommendation System · Machine Learning + NLP · Content-based recommendation baseline</div>',unsafe_allow_html=True)
