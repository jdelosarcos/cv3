import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
def show(d):
    st.subheader('Recurring offense patterns')
    pairs=d.groupby(['location_label','offense_type']).size().reset_index(name='reports').sort_values('reports',ascending=False)
    st.dataframe(pairs.head(50),use_container_width=True,hide_index=True)
    st.caption('Frequency mining identifies recurring area–offense combinations; it does not establish association-rule causality.')
    st.subheader('K-means clustering of area reporting profiles')
    x=pd.crosstab(d.location_label,d.offense_type)
    x=x.loc[x.sum(axis=1)>0]
    if len(x)<3 or x.shape[1]<2:
        st.warning('Need at least three areas and two offense types to assess clusters.');return
    # Normalize to proportions to distinguish offense mix from absolute reporting volume.
    proportions=x.div(x.sum(axis=1),axis=0)
    scaled=StandardScaler().fit_transform(proportions)
    candidates=[]
    for k in range(2,min(7,len(x)-1)+1):
        model=KMeans(n_clusters=k,random_state=42,n_init=10)
        labels=model.fit_predict(scaled)
        if len(set(labels))<2 or len(set(labels))>=len(x):continue
        candidates.append((k,silhouette_score(scaled,labels),davies_bouldin_score(scaled,labels)))
    if not candidates:
        st.warning('Insufficient distinct area profiles for reliable clustering.');return
    scores=pd.DataFrame(candidates,columns=['clusters','silhouette','davies_bouldin'])
    best=int(scores.sort_values('silhouette',ascending=False).iloc[0].clusters)
    labels=KMeans(n_clusters=best,random_state=42,n_init=10).fit_predict(scaled)
    st.dataframe(scores,hide_index=True,use_container_width=True)
    st.caption(f'Selected k={best} by highest silhouette score. Higher silhouette and lower Davies–Bouldin indicate better separation; these are unsupervised diagnostics, not prediction accuracy.')
    pts=PCA(n_components=2).fit_transform(scaled)
    vis=pd.DataFrame({'PC1':pts[:,0],'PC2':pts[:,1],'cluster':labels.astype(str),'area':x.index,'reports':x.sum(axis=1).values})
    st.plotly_chart(px.scatter(vis,x='PC1',y='PC2',color='cluster',hover_name='area',size='reports',title='Area offense-mix clusters (PCA projection)'),use_container_width=True)
    st.dataframe(vis[['area','cluster','reports']].sort_values(['cluster','reports'],ascending=[True,False]),hide_index=True,use_container_width=True)
    st.info('Interpretation: areas grouped together have similar *proportions* of reported offense types. Inspect cluster membership and data completeness before allocating resources.')
