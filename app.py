import streamlit as st, pandas as pd, joblib, os
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix,classification_report
st.set_page_config(page_title='Adult Income Classifier',page_icon='📊',layout='wide')
st.title('📊 Adult Income Classification Dashboard')
st.caption('M.Tech AIML Machine Learning Assignment 2 | Student: IJAZ E | ID: 2026AJ05484')
meta=joblib.load('model/metadata.joblib')
files={'Logistic Regression':'model/logistic_regression.joblib','Decision Tree':'model/decision_tree.joblib','KNN':'model/knn.joblib','Naive Bayes':'model/naive_bayes.joblib','Support Vector Machine':'model/support_vector_machine.joblib'}
choice=st.selectbox('Choose a trained classifier',list(files))
up=st.file_uploader('Upload test CSV containing input features and income target',type='csv')
if up is None:
    st.info('Upload test_data.csv from the repository to view evaluation results.')
else:
    df=pd.read_csv(up); target=meta['target']
    if target not in df.columns: st.error('The CSV must contain the income target column.'); st.stop()
    X=df[meta['features']]; y=df[target].astype(int); pipe=joblib.load(files[choice]); pred=pipe.predict(X); prob=pipe.predict_proba(X)[:,1]
    metrics={'Accuracy':accuracy_score(y,pred),'AUC':roc_auc_score(y,prob),'Precision':precision_score(y,pred,zero_division=0),'Recall':recall_score(y,pred,zero_division=0),'F1':f1_score(y,pred,zero_division=0),'MCC':matthews_corrcoef(y,pred)}
    st.subheader(f'{choice} results'); st.dataframe(pd.DataFrame([metrics]).style.format('{:.4f}'),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.write('Confusion matrix'); st.dataframe(pd.DataFrame(confusion_matrix(y,pred),index=['Actual 0','Actual 1'],columns=['Predicted 0','Predicted 1']))
    with c2: st.write('Classification report'); st.code(classification_report(y,pred,target_names=['<=50K','>50K'],zero_division=0))
