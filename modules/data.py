import io,re
import numpy as np
import pandas as pd
import streamlit as st
ALIASES={'date_committed':'date','datecommitted':'date','time_committed':'time','timecommitted':'time','stages_of_felony':'stage','stage_of_felony':'stage','stageoffelony':'stage','offensetype':'offense_type','latitude':'latitude','lat':'latitude','longitude':'longitude','lng':'longitude','lon':'longitude','victims':'victim'}
def norm(x):
    x=re.sub(r'[^a-z0-9]+','_',str(x).strip().lower()).strip('_')
    return ALIASES.get(x,x)
def parse_hour(v):
    if pd.isna(v):return np.nan
    if hasattr(v,'hour'):return v.hour
    if isinstance(v,(float,int)) and 0<=v<1:return int(v*24)
    try:return pd.to_datetime(str(v),errors='coerce').hour
    except:return np.nan
@st.cache_data(show_spinner='Reading and standardizing workbooks…')
def load(files):
    frames=[]
    for name,raw in files:
        book=pd.ExcelFile(io.BytesIO(raw))
        for sheet in book.sheet_names:
            d=pd.read_excel(book,sheet_name=sheet)
            if d.empty:continue
            d.columns=[norm(c) for c in d.columns]
            d=d.loc[:,~d.columns.duplicated()].copy()
            d['source_file']=name;d['source_sheet']=sheet
            frames.append(d)
    if not frames:raise ValueError('No nonempty worksheets found')
    d=pd.concat(frames,ignore_index=True,sort=False)
    for col in ['barangay','date','time','stage','offense','offense_type','latitude','longitude']:
        if col not in d:d[col]=np.nan
    for col in ['barangay','stage','offense','offense_type']:
        d[col]=d[col].astype('string').str.strip().replace({'':pd.NA}).fillna('Unknown')
    d['date']=pd.to_datetime(d['date'],errors='coerce')
    d['hour']=d['time'].map(parse_hour)
    d['latitude']=pd.to_numeric(d['latitude'],errors='coerce')
    d['longitude']=pd.to_numeric(d['longitude'],errors='coerce')
    d['valid_coordinates']=d.latitude.between(-90,90)&d.longitude.between(-180,180)&~((d.latitude==0)&(d.longitude==0))
    d['location_label']=d.barangay.astype(str)
    d['possible_duplicate']=d.duplicated(subset=['barangay','date','time','offense','latitude','longitude'],keep=False)
    d['month_start']=d.date.dt.to_period('M').dt.to_timestamp()
    d['day_of_week']=d.date.dt.day_name()
    d['year']=d.date.dt.year
    return d
