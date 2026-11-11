import pandas as pd
import plotly.express as px
import streamlit as st
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
def trend(d):
    m=d.dropna(subset=['month_start']).groupby('month_start').size().rename('incidents').reset_index()
    if m.empty:return
    m=m.set_index('month_start').asfreq('MS',fill_value=0).reset_index()
    m['rolling_3']=m.incidents.rolling(3,min_periods=1).mean()
    st.plotly_chart(px.line(m,x='month_start',y=['incidents','rolling_3'],markers=True,title='Monthly incidents and three-month moving average'),use_container_width=True)
    if len(m)>1:
        a,b=int(m.incidents.iloc[-2]),int(m.incidents.iloc[-1]);change=(b-a)/a*100 if a else None
        st.info(f'Latest month: {b:,} reports versus {a:,} previously. '+(f'Change: {change:+.1f}%. ' if change is not None else 'Percentage change unavailable because previous month is zero. ')+'Check whether the latest month is complete before drawing conclusions.')
    return m
def patterns(d):
    a,b=st.columns(2)
    with a:
        for col,title in [('offense_type','Offense type'),('location_label','Barangay')]:
            t=d[col].value_counts().head(12).reset_index();t.columns=[col,'reports']
            st.plotly_chart(px.bar(t,x='reports',y=col,orientation='h',title=title),use_container_width=True)
            if len(t):st.caption(f'Most reported {title.lower()}: {t.iloc[0,0]} ({int(t.iloc[0,1]):,} records). This describes reports, not population-adjusted risk.')
    with b:
        dow=d.day_of_week.value_counts().reindex(DAYS,fill_value=0).reset_index();dow.columns=['day','reports']
        st.plotly_chart(px.bar(dow,x='day',y='reports',title='Day-of-week pattern'),use_container_width=True)
        h=d.dropna(subset=['hour']).hour.astype(int).value_counts().reindex(range(24),fill_value=0).reset_index();h.columns=['hour','reports']
        st.plotly_chart(px.bar(h,x='hour',y='reports',title='Hour-of-day pattern'),use_container_width=True)
        if h.reports.sum():st.caption(f'Peak recorded hour: {int(h.loc[h.reports.idxmax(),"hour"]):02d}:00. Interpret cautiously when event times are missing.')
    heat=d.dropna(subset=['hour']).copy()
    if not heat.empty:
        heat['hour']=heat.hour.astype(int)
        p=pd.crosstab(heat.day_of_week,heat.hour).reindex(index=DAYS,columns=range(24),fill_value=0)
        st.plotly_chart(px.imshow(p,aspect='auto',title='Day × hour incident heatmap',labels={'color':'Reports'}),use_container_width=True)
    yearly=d.groupby('year').size().reset_index(name='reports').dropna()
    if len(yearly):st.plotly_chart(px.bar(yearly,x='year',y='reports',title='Annual trend'),use_container_width=True)
    top=d.location_label.value_counts().head(8).index
    area=d[d.location_label.isin(top)].dropna(subset=['month_start']).groupby(['month_start','location_label']).size().reset_index(name='reports')
    if not area.empty:st.plotly_chart(px.line(area,x='month_start',y='reports',color='location_label',title='Barangay monthly trend'),use_container_width=True)
