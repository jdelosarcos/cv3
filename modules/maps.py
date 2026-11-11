import pandas as pd
import plotly.express as px
import streamlit as st
def show(d):
    m=d[d.valid_coordinates].dropna(subset=['latitude','longitude']).copy()
    st.metric('Geocoded records',f'{len(m):,}',help='Only valid nonzero latitude/longitude pairs are plotted')
    if m.empty:
        st.warning('No valid coordinates in the uploaded files. Maps cannot be populated without genuine geocoded records. No coordinates are fabricated.')
        return
    # Scatter Map uses Plotly's maplibre-backed tiles; no Mapbox token needed.
    grouped=m.groupby(['latitude','longitude','location_label'],dropna=False).size().reset_index(name='reports')
    grouped['marker_size']=grouped.reports.clip(lower=1)
    fig=px.scatter_map(grouped,lat='latitude',lon='longitude',size='marker_size',color='reports',hover_name='location_label',hover_data={'reports':True,'marker_size':False},map_style='open-street-map',zoom=9,title='Geocoded reported incidents')
    fig.update_layout(height=560,margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(fig,use_container_width=True)
    fig2=px.density_map(m,lat='latitude',lon='longitude',radius=18,zoom=9,map_style='open-street-map',title='Reported incident density (descriptive)')
    fig2.update_layout(height=520,margin=dict(l=0,r=0,t=45,b=0))
    st.plotly_chart(fig2,use_container_width=True)
    st.caption('Coordinates are used exactly as supplied. Density indicates reporting concentration, not individual danger or causation. Basemap tiles require internet access.')
