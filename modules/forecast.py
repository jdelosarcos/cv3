import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge,PoissonRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import plotly.express as px
FEATURES=['lag_1','lag_2','lag_3','rolling_mean_3','month','year','month_sin','month_cos']
def panel_data(d):
    d=d.dropna(subset=['month_start'])
    counts=d.groupby(['month_start','location_label']).size().rename('target').reset_index()
    if counts.empty:return pd.DataFrame(),pd.DataFrame()
    months=pd.date_range(counts.month_start.min(),counts.month_start.max(),freq='MS')
    panel=counts.pivot(index='month_start',columns='location_label',values='target').reindex(months).fillna(0)
    rows=[]
    for area in panel.columns:
        s=panel[area]
        for i in range(3,len(s)):
            dt=s.index[i]
            rows.append({'area':area,'month_start':dt,'target':float(s.iloc[i]),'lag_1':float(s.iloc[i-1]),'lag_2':float(s.iloc[i-2]),'lag_3':float(s.iloc[i-3]),'rolling_mean_3':float(s.iloc[i-3:i].mean()),'month':dt.month,'year':dt.year,'month_sin':np.sin(2*np.pi*dt.month/12),'month_cos':np.cos(2*np.pi*dt.month/12)})
    return panel,pd.DataFrame(rows)
def models():
    return {'Seasonal naive (lag 1)':None,'Historical mean':DummyRegressor(strategy='mean'),'Ridge':Ridge(alpha=10),'Poisson GLM':PoissonRegressor(alpha=1,max_iter=500),'Random Forest':RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=-1),'Extra Trees':ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=-1),'Gradient Boosting':HistGradientBoostingRegressor(max_iter=100,l2_regularization=1,random_state=42)}
def predict(name,model,tr,te):
    if model is None:return te.lag_1.clip(lower=0).to_numpy()
    m=clone(model);m.fit(tr[FEATURES],tr.target)
    return np.maximum(0,m.predict(te[FEATURES]))
def score(y,p):
    mae=mean_absolute_error(y,p);rmse=np.sqrt(mean_squared_error(y,p));r2=r2_score(y,p) if len(y)>1 and np.var(y)>0 else np.nan
    return {'MAE':mae,'RMSE':rmse,'R2':r2}
@st.cache_data(show_spinner='Comparing models and performing forward-chaining validation…')
def evaluate(model_df):
    months=sorted(model_df.month_start.unique())
    if len(months)<8:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), 'At least 8 supervised months (11 total months) are required for this evaluation.'
    cut=max(4,int(len(months)*.8));cut=min(cut,len(months)-1)
    tr=model_df[model_df.month_start.isin(months[:cut])];te=model_df[model_df.month_start.isin(months[cut:])]
    if len(tr)<8 or te.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),'Insufficient training/test rows.'
    hold=[];preds=[];cv=[]
    for name,model in models().items():
        try:
            p=predict(name,model,tr,te);hold.append({'algorithm':name,**score(te.target,p)})
            preds.extend({'algorithm':name,'area':a,'month_start':dt,'actual':y,'prediction':v} for a,dt,y,v in zip(te.area,te.month_start,te.target,p))
            # Expanding-window month-grouped folds; never split the same calendar month across train and validation.
            initial=max(4,len(months)//2)
            remaining=months[:cut]
            validation_months=remaining[initial:]
            folds=np.array_split(np.array(validation_months,dtype='datetime64[ns]'),min(5,len(validation_months))) if validation_months else []
            for fold_id,fold in enumerate(folds,1):
                if not len(fold):continue
                first=pd.Timestamp(fold[0]);last=pd.Timestamp(fold[-1])
                train_fold=model_df[model_df.month_start<first]
                test_fold=model_df[(model_df.month_start>=first)&(model_df.month_start<=last)]
                if train_fold.empty or test_fold.empty:continue
                pf=predict(name,model,train_fold,test_fold)
                cv.append({'algorithm':name,'fold':fold_id,'train_end':train_fold.month_start.max(),'validation_start':first,'validation_end':last,**score(test_fold.target,pf)})
        except Exception as exc:
            hold.append({'algorithm':name,'error':str(exc)[:160]})
    return pd.DataFrame(hold),pd.DataFrame(preds),pd.DataFrame(cv),''
def next_month_forecast(panel,df,name):
    model=models()[name]
    dt=panel.index.max()+pd.offsets.MonthBegin(1)
    rows=[]
    for area in panel.columns:
        s=panel[area]
        rows.append({'area':area,'month_start':dt,'lag_1':s.iloc[-1],'lag_2':s.iloc[-2],'lag_3':s.iloc[-3],'rolling_mean_3':s.iloc[-3:].mean(),'month':dt.month,'year':dt.year,'month_sin':np.sin(2*np.pi*dt.month/12),'month_cos':np.cos(2*np.pi*dt.month/12)})
    future=pd.DataFrame(rows)
    if model is None:future['predicted_count']=future.lag_1
    else:
        model.fit(df[FEATURES],df.target)
        future['predicted_count']=np.maximum(0,model.predict(future[FEATURES]))
    future['relative_tier']=pd.cut(future.predicted_count.rank(pct=True,method='average'),bins=[0,.5,.8,1],labels=['Lower','Moderate','Higher'],include_lowest=True)
    return future.sort_values('predicted_count',ascending=False)
def show(d):
    panel,df=panel_data(d)
    if df.empty:st.warning('At least four months of records are needed to build lag features.');return
    hold,pred,cv,error=evaluate(df)
    if error:st.warning(error);return
    valid=hold.dropna(subset=['MAE']).sort_values('MAE')
    st.subheader('Chronological 80/20 holdout comparison')
    st.dataframe(hold,use_container_width=True,hide_index=True)
    if valid.empty:st.warning('No model completed successfully.');return
    st.plotly_chart(px.bar(valid,x='algorithm',y='MAE',title='Holdout mean absolute error (lower is better)'),use_container_width=True)
    baseline=valid[valid.algorithm=='Seasonal naive (lag 1)']
    base_mae=float(baseline.MAE.iloc[0]) if not baseline.empty else np.nan
    hold['skill_vs_naive_pct']=np.where(base_mae>0,100*(1-hold.MAE/base_mae),np.nan)
    hold['rating']=np.where((hold.skill_vs_naive_pct>=10)&(hold.R2>=0),'PASS','DOES NOT PASS')
    hold.loc[hold.MAE.isna(),'rating']='NOT EVALUATED'
    st.subheader('Transparent passing criteria')
    st.caption('PASS requires ≥10% lower holdout MAE than last-month naive baseline AND holdout R² ≥ 0. This is a configurable operational rubric, not an official certification. No model is guaranteed to pass.')
    st.dataframe(hold[['algorithm','MAE','RMSE','R2','skill_vs_naive_pct','rating']],hide_index=True,use_container_width=True)
    st.subheader('Forward-chaining cross-validation (training period only)')
    if cv.empty:st.warning('Insufficient pre-holdout months for cross-validation.')
    else:
        st.dataframe(cv.groupby('algorithm',as_index=False)[['MAE','RMSE','R2']].mean().sort_values('MAE'),hide_index=True,use_container_width=True)
        st.plotly_chart(px.box(cv,x='algorithm',y='MAE',points='all',title='CV error variability across folds'),use_container_width=True)
        with st.expander('Individual chronological CV folds'):st.dataframe(cv,hide_index=True,use_container_width=True)
    name=st.selectbox('Algorithm for actual vs predicted and next-month estimate',valid.algorithm.tolist())
    subset=pred[pred.algorithm==name].groupby('month_start')[['actual','prediction']].sum().reset_index().melt(id_vars='month_start',var_name='series',value_name='count')
    st.plotly_chart(px.line(subset,x='month_start',y='count',color='series',markers=True,title='Holdout: observed vs predicted monthly counts'),use_container_width=True)
    if len(panel)>=3:
        future=next_month_forecast(panel,df,name)
        st.subheader(f'Next-month area forecast: {future.month_start.iloc[0]:%B %Y}')
        st.dataframe(future[['area','predicted_count','relative_tier']].head(30),hide_index=True,use_container_width=True)
        st.download_button('Download forecast CSV',future.to_csv(index=False).encode(),'crime_forecast.csv','text/csv')
    st.info('Predictions estimate future *reported counts*, not individual criminality. Relative tiers are within-dataset ranks; low counts and changes in reporting practice can make forecasts unstable.')
