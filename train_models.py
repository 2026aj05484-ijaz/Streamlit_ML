import os, urllib.request, zipfile, io, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from pathlib import Path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

os.makedirs('model',exist_ok=True)
url='https://archive.ics.uci.edu/static/public/2/adult.zip'
data=urllib.request.urlopen(url,timeout=60).read()
z=zipfile.ZipFile(io.BytesIO(data))
raw=z.read('adult.data').decode('utf-8')
cols=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']

(BASE_DIR / "adult.data").write_bytes(raw.encode())

df=pd.read_csv(io.StringIO(raw),names=cols,skipinitialspace=True,na_values='?')
df=df.dropna().reset_index(drop=True)
df['income']=(df['income'].str.strip().str.rstrip('.')=='>50K').astype(int)
# preserve a manageable test set; split stratified
X=df.drop(columns='income'); y=df['income']
cat=X.select_dtypes(include='object').columns.tolist(); num=X.select_dtypes(exclude='object').columns.tolist()
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore', sparse_output=False ))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={'Logistic Regression':LogisticRegression(max_iter=1000), 'Decision Tree':DecisionTreeClassifier(max_depth=12,random_state=42), 'KNN':KNeighborsClassifier(n_neighbors=7), 'Naive Bayes':GaussianNB(), 'Support Vector Machine':SVC(probability=True,random_state=42)}
rows=[]
for name,est in models.items():
    pipe=Pipeline([('preprocess',pre),('model',est)])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    rows.append({'ML Model':name,'Accuracy':accuracy_score(yte,pred),'AUC':roc_auc_score(yte,prob),'Precision':precision_score(yte,pred),'Recall':recall_score(yte,pred),'F1':f1_score(yte,pred),'MCC':matthews_corrcoef(yte,pred)})
    joblib.dump(pipe,f'model/{name.lower().replace(" ","_")}.joblib')
test=Xte.copy(); test['income']=yte.values; test.to_csv('test_data.csv',index=False)
pd.DataFrame(rows).to_csv('metrics.csv',index=False)
# metadata for app
joblib.dump({'features':X.columns.tolist(),'target':'income','categorical':cat,'numeric':num},'model/metadata.joblib')
print(pd.DataFrame(rows).to_string(index=False))
