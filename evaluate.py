from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from model import load_jobs, recommend

BASE=Path(__file__).parent
resume=(BASE/'sample_resume.txt').read_text()
jobs=load_jobs()
results=recommend(resume,jobs,top_n=len(jobs))
truth=pd.read_csv(BASE/'evaluation_ground_truth.csv')
pred=results[['job_title','match_score']].copy()
# Verification convention: top 60% of ranked jobs are treated as recommended.
cut=max(1,round(len(pred)*0.60)); pred['predicted']=(pred.index < cut).astype(int)
merged=truth.merge(pred,on='job_title',how='left')
y_true=merged['relevant'].fillna(0).astype(int)
y_pred=merged['predicted'].fillna(0).astype(int)
print('Verification set:',len(merged),'jobs')
print('Precision:',round(precision_score(y_true,y_pred,zero_division=0),3))
print('Recall:',round(recall_score(y_true,y_pred,zero_division=0),3))
print('F1:',round(f1_score(y_true,y_pred,zero_division=0),3))
print()
print('Top ranked jobs:')
print(results[['job_title','match_score','text_score','skill_score']].to_string(index=False))