# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 22:20:06 2023

@author: thoma
"""
import pandas as pd 
import numpy as np 
import fiona
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import math
from shapely.ops import cascaded_union
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
from dash import Dash, html, Input, Output, State, callback_context,dcc
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dtaidistance import dtw
from scipy.spatial import distance
import bisect
from dateutil.relativedelta import relativedelta

# =============================================================================
# Dataset Fatalities
# =============================================================================


df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
                 parse_dates=['date_start','date_end'],low_memory=False)
df_tot = pd.DataFrame(columns=df.country.unique(),index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
df_tot=df_tot.fillna(0)
for i in df.country.unique():
    df_sub=df[df.country==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j] == df_sub.date_end.iloc[j]:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            df_tot.loc[df_sub.date_start.iloc[j]:
            df_sub.date_end.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j]: \
                                                  df_sub.date_end.iloc[j],i]+ \
                                                  df_sub.best.iloc[j]/ \
                                                  (df_sub.date_end.iloc[j]- \
                                                  df_sub.date_start.iloc[j]).days 

df_tot_m=df_tot.resample('M').sum()
df_tot=df_tot.resample('W').sum()   
df_tot_d= df_tot_m.diff()



#### extract series
df_ind = df_tot_d.reset_index(drop=True)
seq=[]
for i in range(len(df_tot_m.columns)): 
    if len(df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[:-1])!=0:
        for j in range(len(df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[:-1])):
            if df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j+1]-df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j] > 60:    #min 5 years
                seq.append(df_tot_m.iloc[df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j]+1:df_ind.iloc[:,i][df_ind.iloc[:,i]==0].index[j+1],i])
    else : 
        seq.append(df_tot_m.iloc[:,i])

seq_n=[]
for i in seq:
    seq_n.append((i-i.mean())/i.std())

def int_exc(win=12):
    n_test=[]
    to=0
    exclude=[]
    interv=[0]
    for i in seq_n:
        n_test=np.concatenate([n_test,i])
        to=to+len(i)
        exclude=exclude+[*range(to-win,to)]
        interv.append(to)
    return exclude,interv,n_test

exclude,interv,n_test = int_exc()


# =============================================================================
# Visu 
# =============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = html.Div([
    html.Div([dcc.Store(id='memory')]),
    html.H1(children='Shape finder',style = {'textAlign': 'center','marginBottom':40,'marginTop':20}),
    html.Div([
        html.Div(children=[
            html.Div(id='slider_l',children=[html.Div(['1',dcc.Slider(0, 1, marks=None, value=0.5,id='s1',vertical=True)],style={'height': '30%'}),
                               html.Div(['2',dcc.Slider(0, 1, marks=None, value=0.5,id='s2',vertical=True)],style={'height': '30%'}),
                               html.Div(['3',dcc.Slider(0, 1, marks=None, value=0.5,id='s3',vertical=True)],style={'height': '30%'}),
                               html.Div(['4',dcc.Slider(0, 1, marks=None, value=0.5,id='s4',vertical=True)],style={'height': '30%'}),
                               html.Div(['5',dcc.Slider(0, 1, marks=None, value=0.5,id='s5',vertical=True)],style={'height': '30%'}),
                               html.Div(['6',dcc.Slider(0, 1, marks=None, value=0.5,id='s6',vertical=True)],style={'height': '30%'}),
                               html.Div(['7',dcc.Slider(0, 1, marks=None, value=0.5,id='s7',vertical=True)],style={'height': '30%'}),
                               html.Div(['8',dcc.Slider(0, 1, marks=None, value=0.5,id='s8',vertical=True)],style={'height': '30%'}),
                               html.Div(['9',dcc.Slider(0, 1, marks=None, value=0.5,id='s9',vertical=True)],style={'height': '30%'}),
                               html.Div(['10',dcc.Slider(0, 1, marks=None, value=0.5,id='s10',vertical=True)],style={'height': '30%'}),
                               html.Div(['11',dcc.Slider(0, 1, marks=None, value=0.5,id='s11',vertical=True)],style={'height': '30%'}),
                               html.Div(['12',dcc.Slider(0, 1, marks=None, value=0.5,id='s12',vertical=True)],style={'height': '30%'})],
                     style={'display':'flex','flex-direction':'row','height': 30, 'width':250}),
        ],style={'margin-left':100,'margin-right':100}),
        html.Div(children=[
        dcc.Graph(id='plot')],style={'margin-left':200})
        ], style={'display': 'flex', 'flex-direction': 'row','marginTop':20,'marginBottom':50}),
    html.Div([
        html.Div(['DTW window',dcc.Slider(0, 2, 1,value=0, id='submit')],style={'width': '10%','margin-inline':'80px'}),
        dcc.RadioItems(['DTW','Euclidean'],'Euclidean',id='sel',inline=True,style={'margin-inline':'80px'}),
        html.Div(['Month window', dcc.Slider(6,12,1,value=6,id='slider')],style={'margin-inline':'80px','width':500}),
        html.Div([
            html.Button("Download CSV", id="btn_csv"),
            dcc.Download(id="download-dataframe-csv"),
            ])
        ],
        style = {'display': 'flex','flex-direction': 'row','grid-auto-columns':'40%','width':'100%','marginTop':50,'marginBottom':20}),
    html.Div([
        html.Div(children=[
            dcc.Graph(id='plot2')],style={'width': '35%'}),
        html.Div(children=[
            dcc.Graph(id='plot3')],style={'width': '35%'}),
        html.Div(children=[
            dcc.Graph(id='plot4')],style={'width': '35%'})
        ],style={'display': 'flex', 'flex-direction': 'row','width': '99%'})
])


@app.callback(Output('plot', 'figure'),
              Output('plot2', 'figure'),
              Output('plot3', 'figure'),
              Output('plot4', 'figure'),
              Output('memory','data'),
              Input('submit', 'value'),
              Input('sel', 'value'),
              Input('slider', 'value'),
              Input('s1','value'),
              Input('s2','value'),
              Input('s3','value'),
              Input('s4','value'),
              Input('s5','value'),
              Input('s6','value'),
              Input('s7','value'),
              Input('s8','value'),
              Input('s9','value'),
              Input('s10','value'),
              Input('s11','value'),
              Input('s12','value')
              )

def update_elements(submit,sel,sli,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12):     
    x=[]
    y=[]
    s_l = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]
    for elmt in range(sli):
        x.append(elmt+1)
        y.append(s_l[elmt])
    df = pd.Series(data=y,index=x)
    df = df.sort_index()
    df.index = range(1,len(df)+1)
    fig = px.line(x=df.index, y=df,title='Shape Wanted')
    fig.update_layout(xaxis_title='Number of Month',
                       yaxis_title='Normalized Fatalities',title_x=0.5)
    if  sel=='DTW':   
        fig_2,fig_3,fig_4,memo = find_most_close(df,len(df),metric='dtw',loop=submit)
    elif  sel=='Euclidean':   
         fig_2,fig_3,fig_4,memo = find_most_close(df,len(df),metric='euclidean') 
    return fig,fig_2,fig_3,fig_4,memo.reset_index().to_json(orient="split")


def find_most_close(seq1,win,metric='euclidean',loop=0):
    if loop == 0:
        if seq1.var()!=0.0:
            seq1 = (seq1 - seq1.min())/(seq1.max() - seq1.min())
        seq1= np.array(seq1)
        tot=[]
        exclude,interv,n_test = int_exc(win)
        for i in range(len(n_test)):
            if i not in exclude:
                seq2 = n_test[i:i+win]
                seq2 = seq2 = (seq2 - seq2.min())/(seq2.max() - seq2.min())
                try:
                    if metric=='euclidean':
                        dist = distance.euclidean(seq1,seq2)
                    elif metric=='dtw':
                        dist = dtw.distance(seq1,seq2)
                    tot.append([i,dist])
                except:
                    1
        tot=pd.DataFrame(tot)
        tot = tot.sort_values([1])
        figlist=[]
        memo = pd.DataFrame(index=range(12))
        for i in tot.iloc[:3,0].tolist():
            col = seq[bisect.bisect_right(interv, i)-1].name
            index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
            index_obs_2 = index_obs + relativedelta(months=win)
            obs = df_tot_m.loc[:index_obs_2,col].iloc[-win-1:-1]
            memo=pd.concat([memo,pd.Series(obs.index).reset_index(drop=True)],axis=1)
            memo=pd.concat([memo,pd.Series(obs).reset_index(drop=True)],axis=1)                 
            fig_out = px.line(x=obs.index, y=obs,title=col,markers='o')
            fig_out.update_layout(xaxis_title='Date',
                           yaxis_title='Number of Fatalities',title_x=0.5)
            figlist.append(fig_out)
        figlist.append(memo)    
    else:
        if seq1.var()!=0.0:
            seq1 = (seq1 - seq1.min())/(seq1.max() - seq1.min())
        seq1= np.array(seq1)
        tot=[]
        for lop in range(int(-loop),int(loop)+1):
            exclude,interv,n_test = int_exc(win+lop)
            for i in range(len(n_test)):
                if i not in exclude:
                    seq2 = n_test[i:i+int(win+lop)]
                    seq2 = seq2 = (seq2 - seq2.min())/(seq2.max() - seq2.min())
                    try:
                        dist = dtw.distance(seq1,seq2)
                        tot.append([i,dist,win+lop])
                    except:
                        1
        tot=pd.DataFrame(tot)
        tot = tot.sort_values([1])
        figlist=[]
        df_fig=[]
        memo = pd.DataFrame(index=range(12))
        c_lo=0
        while len(figlist)<3:
            i = tot.iloc[c_lo,0]
            win = int(tot.iloc[c_lo,2])
            exclude,interv,n_test = int_exc(win)
            col = seq[bisect.bisect_right(interv, i)-1].name
            index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
            index_obs_2 = index_obs + relativedelta(months=win)
            obs = df_tot_m.loc[:index_obs_2,col].iloc[-win-1:-1]
            flag_ok=True
            if c_lo!=0:
                for ran in range(len(df_fig)):
                    if (obs.index[int(win/2)].year == df_fig[ran][0]) and (col == df_fig[ran][1]):
                        flag_ok=False
            if flag_ok==True:        
                fig_out = px.line(x=obs.index, y=obs,title=col,markers='o')
                fig_out.update_layout(xaxis_title='Date',
                               yaxis_title='Number of Fatalities',title_x=0.5)
                figlist.append(fig_out)
                df_fig.append([obs.index[int(win/2)].year,col])
                memo=pd.concat([memo,pd.Series(obs.index).reset_index(drop=True)],axis=1)
                memo=pd.concat([memo,pd.Series(obs).reset_index(drop=True)],axis=1)   
            c_lo=c_lo+1
        figlist.append(memo)   
    return figlist

@app.callback(
    Output('slider_l', 'children'),
    [Input('slider', 'value')],
    [State('slider_l', 'children')])

def update_slide(value,slid):
    sli_list=[]
    for i in range(value):
        sli_list.append(html.Div([str(i+1),dcc.Slider(0, 1, marks=None, value=0.5,id='s'+str(i+1),vertical=True)],style={'height': '30%'}))
    for i in range(len(sli_list),12):
        sli_list.append(html.Div([dcc.Slider(0, 0, marks=None,id='s'+str(i+1),vertical=True)],style={'display': 'none'}))
    sl = html.Div(id='slider_l',children=sli_list,
            style={'display':'flex','flex-direction':'row','height':30})
    return sl

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State('memory','data'),
    prevent_initial_call=True,
)
def func(n_clicks,data):
    df = pd.read_json(data,orient="split")
    print(df)
    return dcc.send_data_frame(df.to_csv, "Output.csv")

webbrowser.open('http://127.0.0.1:8050/',new=2)

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)