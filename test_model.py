import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from model import clean, extract_skills, load_jobs, recommend

def test_clean_lowercases_and_normalizes():
    assert clean('Python!!!  SQL') == 'python sql'

def test_skill_extraction():
    skills=extract_skills('Python, SQL, Machine Learning and NLP')
    assert 'python' in skills and 'sql' in skills and 'machine learning' in skills and 'nlp' in skills

def test_recommendation_output():
    jobs=load_jobs(); r=recommend('Python SQL Pandas Machine Learning NLP',jobs,top_n=5)
    assert len(r)==5
    assert r['match_score'].between(0,100).all()
    assert 'matched' in r.columns and 'missing' in r.columns