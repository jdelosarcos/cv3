import streamlit as st
import pandas as pd
from modules.data import load
st.set_page_config(page_title='Crime Intelligence | Decision Support',page_icon='📊',layout='wide')
st.markdown('''<style>
.stApp{background:#f5f7fb;color:#14233b}.block-container{padding-top:1.5rem;max-width:1450px}
h1,h2,h3{color:#12315a}.stMetric{background:white;padding:16px;border:1px solid #dce5ef;border-radius:12px}
[data-testid="stSidebar"]{background:#112a49}[data-testid="stSidebar"] *{color:#f2f6ff}
</style>''',unsafe_allow_html=True)
st.title('Crime Occurrence Intelligence')
st.caption('Operational analytics • transparent model validation • evidence-based resource planning')
files=st.sidebar.file_uploader('Upload crime Excel datasets (one or more)',type=['xlsx','xls'],accept_multiple_files=True)
st.sidebar.caption('Data stays within this Streamlit session. Avoid uploading personally identifying victim/suspect information to public deployments.')
if not files:
    st.info('Upload the source Excel files used by the original Colab notebook to activate the dashboard. The notebook itself contains code, not the four source workbooks.')
    st.stop()
try:d=load(tuple((f.name,f.getvalue()) for f in files))
except Exception as e:st.error(f'Unable to read input: {e}');st.stop()
areas=sorted(d.location_label.dropna().unique())
selected=st.sidebar.multiselect('Filter barangays',areas,default=[])
if selected:d=d[d.location_label.isin(selected)]
offenses=sorted(d.offense_type.dropna().unique())
chosen=st.sidebar.multiselect('Filter offense types',offenses,default=[])
if chosen:d=d[d.offense_type.isin(chosen)]
valid=d.dropna(subset=['date'])
if not valid.empty:
    bounds=(valid.date.min().date(),valid.date.max().date())
    dates=st.sidebar.date_input('Incident date range',value=bounds,min_value=bounds[0],max_value=bounds[1])
    if isinstance(dates,(tuple,list)) and len(dates)==2:d=d[d.date.between(pd.Timestamp(dates[0]),pd.Timestamp(dates[1])+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))]
section=st.sidebar.radio('Workspace',['Executive overview','Trends & patterns','Geographic intelligence','Pattern mining & clustering','Prediction & validation','Data quality & export'])
if d.empty:st.warning('No records match the selected filters.');st.stop()
if section=='Executive overview':
    from modules.charts import trend
    a,b,c,e=st.columns(4)
    a.metric('Reported incidents',f'{len(d):,}')
    b.metric('Barangays',d.location_label.nunique())
    c.metric('Offense categories',d.offense_type.nunique())
    e.metric('Geocoded',f'{d.valid_coordinates.mean():.1%}')
    trend(d)
    st.subheader('Decision briefing')
    if not d.empty:
        top=d.location_label.value_counts().index[0]
        offense=d.offense_type.value_counts().index[0]
        st.info(f'The highest reported count is in {top}; the most frequently reported offense category is {offense}. Review the trend and geospatial tabs before planning patrol coverage. Counts are not adjusted for population or exposure.')
elif section=='Trends & patterns':
    from modules.charts import trend,patterns
    trend(d);patterns(d)
elif section=='Geographic intelligence':
    from modules.maps import show
    show(d)
elif section=='Pattern mining & clustering':
    from modules.mining import show
    show(d)
elif section=='Prediction & validation':
    from modules.forecast import show
    show(d)
else:
    st.subheader('Input completeness and audit checks')
    q=pd.DataFrame({'missing':d.isna().sum(),'missing_pct':(d.isna().mean()*100).round(2),'type':d.dtypes.astype(str)})
    st.dataframe(q,use_container_width=True)
    st.metric('Potential duplicate rows (not deleted)',int(d.possible_duplicate.sum()))
    st.metric('Invalid or missing coordinates',int((~d.valid_coordinates).sum()))
    st.metric('Missing/unparseable dates',int(d.date.isna().sum()))
    safe=d.drop(columns=[c for c in ['victim','suspect','victims','suspects'] if c in d],errors='ignore')
    st.download_button('Download cleaned records CSV (sensitive description fields omitted)',safe.to_csv(index=False).encode(),'crime_cleaned.csv','text/csv')
    st.caption('Inspect additional free-text columns for identifying details before sharing exports.')
st.divider()
st.caption('Decision support only • Reporting bias, partial months, missing geography, and small samples limit conclusions. Do not use these models to target individuals.')
