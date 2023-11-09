# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:03:09 2023

@author: thoma
"""

import base64
import io
import numpy as np
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dtaidistance import dtw,ed
import bisect

external_stylesheets=[dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'ShapeFinder'
app._favicon = ("icone_pace.ico")
server = app.server

pace_png = base64.b64encode(open('PaCE_final_icon.png', 'rb').read()).decode('ascii')
mail_png = base64.b64encode(open('Gmail_Logo_256px.png', 'rb').read()).decode('ascii')
git_png = base64.b64encode(open('github-mark.png', 'rb').read()).decode('ascii')
twitter_png = base64.b64encode(open('2021 Twitter logo - blue.png', 'rb').read()).decode('ascii')

app.layout = html.Div([
    html.Div([
    html.A([html.Img(src='data:image/png;base64,{}'.format(pace_png),style={
            'position': 'absolute',
            'right': '5px',
            'top': '10px',
            'display': 'block',
            'height': '25px',
            'width': '25px'
        })], href='https://paceconflictlab.wixsite.com/conflict-research-la'),
    html.A([html.Img(src='data:image/png;base64,{}'.format(mail_png),style={
            'position': 'absolute',
            'right': '45px',
            'top': '10px',
            'display': 'block',
            'height': '25px',
            'width': '25px'
        })], href='mailto:schincat@tcd.ie'),    
    html.A([html.Img(src='data:image/png;base64,{}'.format(git_png),style={
            'position': 'absolute',
            'right': '85px',
            'top': '10px',
            'display': 'block',
            'height': '25px',
            'width': '25px'
        })], href='https://github.com/conflictlab'),
    html.A([html.Img(src='data:image/png;base64,{}'.format(twitter_png),style={
            'position': 'absolute',
            'right': '125px',
            'top': '10px',
            'display': 'block',
            'height': '25px',
            'width': '25px'
        })], href='https://twitter.com/LabConflict')
    ]),
    html.H1(children='Time Series Shape Finder',style = {'textAlign': 'center','marginBottom':40,'marginTop':20}),
    html.Div([dcc.Markdown('''Shape Finder is a platform to identify and predict patterns in time series datasets. It lets you search for specific shapes within complex datasets. Through the use of adjustable sliders, you can define the shape you are interested in—be it a sharp peak, a gradual incline, or any other distinctive form. Once defined, Shape Finder sifts through your dataset to find the closest matching patterns.
	For added clarity and visualization, the application plots the three most similar shapes, allowing you to gauge the accuracy and relevance of the matches. Beyond mere identification, Shape Finder also predicts how these patterns might evolve over the next six periods.
	Finally, you can download a CSV file which includes all identified shapes and their associated similarity scores.''',
        style={'width': '100%','margin-left':'15px','margin-right':'15px'},
    )]),
    html.Hr(style={'width': '70%','margin':'auto'}),
    html.H5(children='STEP 1: SELECT THE DATASET',style = {'textAlign': 'center','marginBottom':40,'marginTop':20}),
    html.Div([dcc.Markdown('''
                           * Conflict-Fatalities : Monthly conflict fatalities from Ukraine, Sudan, and Ethiopia, extracted from the UCDP Dataset (source: https://ucdp.uu.se/downloads/). (selected by default)
                           * Inflation.csv: Monthly inflation at the country level (source: https://www.worldbank.org/en/research/brief/inflation-database).
                           * Temperatures.csv: Monthly mean temperatures at the country level, extracted from ERA-5 satellite data (source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
                           '''
        ,dangerously_allow_html=True,
        style={'width': '80%','margin':'auto','text-align': 'justify'})]),
    html.Div([html.Div([html.Button('Conflict-Fatalities', id='conf', n_clicks=0)],style={'margin-left':'425px','height': '60px'}),html.Div([html.Button('Inflation', id='inf', n_clicks=0)],style={'margin-left':'100px','height': '60px'}),html.Div([html.Button('Temperature', id='temp', n_clicks=0)],style={'margin-left':'100px','height': '60px'})],style={'display': 'flex','flex-direction':'row','marginBottom':30}),
    html.H5(children='OR : UPLOAD YOUR OWN DATA',style = {'textAlign': 'center','marginBottom':40,'marginTop':20}),
    html.Div([dcc.Markdown('''Please follow the format of the example datasets on [Github](https://github.com/ThomasSchinca/Shape_Finder_dataset). Once you uploaded it or select one, a vizualization of 
                           the first 10 rows and 5 columns will be displayed.''',dangerously_allow_html=True,
        style={'width': '80%','margin':'auto','text-align': 'justify'})]),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={'width': '50%','height': '60px','lineHeight': '60px','borderWidth': '1px','borderStyle': 'dashed',
            'borderRadius': '5px','textAlign': 'center','margin':'auto','marginBottom':20}),
    dcc.Store(id='store'),
    html.Div(id='output-data-upload'),
    html.Hr(style={'width': '70%','margin':'auto'}),
    html.H5(children='STEP 2: SELECT THE SHAPE AND PARAMETERS',style = {'textAlign': 'center','marginBottom':40,'marginTop':20}),
    html.Div([dcc.Markdown(''' 
                           * DTW / Euclidean : Selection of the distance metric 
                           * DTW Flexibility : Window flexibilty to look for patterns (-/+ flexibility). For example, when 
                           the window is 7 and flexibility is 1, the ShapeFinder is going to search in window 6,7 and 8. Only for DTW.
                           * Length of sequence : Window of the shape wanted. 
                           * Max distance : The maximal distance to allow the sequence found to be included in the matching ones. 
                           Only sequences with lower (or equal) distance are considered. 
                           ''',
    style={'width': '80%','margin':'auto','text-align': 'justify'})]),
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
    ], style={'display': 'flex', 'flex-direction': 'row','marginTop':20,'marginBottom':0}),
    html.Div([
    dcc.RadioItems(['DTW','Euclidean'],'Euclidean',id='sel',style={'margin-inline':'80px'}),
    html.Div(['DTW flexibility',dcc.Slider(0, 2, 1,value=0, id='submit')],style={'width': '10%','margin-inline':'80px'}),
    html.Div(['Length of sequence', dcc.Slider(6,12,1,value=6,id='slider')],style={'margin-inline':'80px','width':500}),
    html.Div(['Max distance',dcc.Input(id="dist_min", type="number", value=0.5,
        min=0, max=3,step=0.1)]),
    ],
    style = {'display': 'flex','flex-direction': 'row','grid-auto-columns':'30%','width':'100%','marginTop':10,'marginBottom':20,
             'lineHeight': '60px'}),
    html.Div([html.Button("Run Analysis", id="btn_start")],style={'width': '50%','height': '60px','lineHeight': '60px','borderWidth': '1px',
            'borderRadius': '5px','textAlign': 'center','margin':'auto','marginBottom':50}),
    html.Hr(style={'width': '70%','margin':'auto'}),
    html.H5(children='The closest matching patterns',style = {'textAlign': 'center','marginBottom':40,'marginTop':40}),
    html.Div([
        html.Div(children=[
            dcc.Graph(id='plot2')],style={'width': '35%'}),
        html.Div(children=[
            dcc.Graph(id='plot3')],style={'width': '35%'}),
        html.Div(children=[
            dcc.Graph(id='plot4')],style={'width': '35%'})
        ],style={'display': 'flex', 'flex-direction': 'row','width': '99%'}),
    html.Hr(style={'width': '70%','margin':'auto'}),
    html.H5(children='Forecast',style = {'textAlign': 'center','marginBottom':40,'marginTop':40}),
    html.Div([dcc.Markdown('''The forecasted values are generated by averaging the subsequent data points from the past patterns that closely match your selected shape.''',
    style={'width': '80%','margin':'auto','text-align': 'justify'})]),
    html.Div([
            dcc.Graph(id='plot5')]),
    html.Hr(style={'width': '70%','margin':'auto'}),
    html.H5(children='Output',style = {'textAlign': 'center','marginBottom':40,'marginTop':40}),
    dcc.Store(id='store2'),
    html.Div([
    html.Button("Download CSV", id="btn_down"),
    dcc.Download(id="download-dataframe-csv"),
    ],style={'width': '100%','height': '50px','lineHeight': '20px','textAlign': 'center',
             'margin':'auto','marginBottom':20}),
    html.Div(id='output-data-csv')
])
             
@app.callback(Output('store', 'data', allow_duplicate=True),
              Input('conf', 'n_clicks'),
              prevent_initial_call='True')

def update_output_1(n_clicks):
    df = pd.read_csv('https://github.com/ThomasSchinca/Shape_Finder_dataset/blob/main/Conflict-fatalities.csv?raw=True',parse_dates=True)
    return df.to_json(date_format='iso', orient='split')  

@app.callback(Output('store', 'data', allow_duplicate=True),
              Input('inf', 'n_clicks'),
              prevent_initial_call='True')

def update_output_2(n_clicks):
    df = pd.read_csv('https://github.com/ThomasSchinca/Shape_Finder_dataset/blob/main/Inflation.csv?raw=True',parse_dates=True)
    return df.to_json(date_format='iso', orient='split') 

@app.callback(Output('store', 'data', allow_duplicate=True),
              Input('temp', 'n_clicks'),
              prevent_initial_call='True')

def update_output_3(n_clicks):
    df = pd.read_csv('https://github.com/ThomasSchinca/Shape_Finder_dataset/blob/main/Temperatures.csv?raw=True',parse_dates=True)
    return df.to_json(date_format='iso', orient='split') 



@app.callback(Output('store', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified')
              )

def update_output(contents, list_of_names, list_of_dates):
    if contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('output-data-upload', 'children'),
    Input('store', 'data'))

def output_from_store(stored_data):
    df = pd.read_json(stored_data, orient='split')
    return html.Div([dash_table.DataTable(df.iloc[:,:5].to_dict('records'),[{'name': i, 'id': i} for i in df.columns[:5]]
                             ,page_size=10)],style={'width': '70%','margin':'auto'})

@app.callback(
    Output('output-data-csv', 'children'),
    Input('store2', 'data'),
    prevent_initial_call=True)

def output_from_store2(stored_data):
    df = pd.read_json(stored_data, orient='split')
    return html.Div([dash_table.DataTable(df.iloc[:,:5].to_dict('records'),[{'name': i, 'id': i} for i in df.columns[:5]]
                             )],style={'width': '70%','margin':'auto','marginBottom':20})

@app.callback(
    Output('plot', 'figure'),
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
    Input('s12','value'),
    Input('slider', 'value'))

def wanted_shape(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,sli):
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
    fig.update_layout(xaxis_title='Number of time points',
                       yaxis_title='Normalized Units',title_x=0.5)
    return fig

@app.callback(
    Output('plot2', 'figure'),
    Output('plot3', 'figure'),
    Output('plot4', 'figure'),
    Output('plot5', 'figure'),
    Output('store2','data'),
    State('s1','value'),
    State('s2','value'),
    State('s3','value'),
    State('s4','value'),
    State('s5','value'),
    State('s6','value'),
    State('s7','value'),
    State('s8','value'),
    State('s9','value'),
    State('s10','value'),
    State('s11','value'),
    State('s12','value'),
    State('slider', 'value'),
    State('submit', 'value'),
    State('sel', 'value'),
    State('store', 'data'),
    State('dist_min','value'),
    Input('btn_start','n_clicks'),
    prevent_initial_call=True)

def analysis(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,sli,submit,sel,stored_data,m_d,n_click):
    data = pd.read_json(stored_data, orient='split')
    data.index = data.iloc[:,0]
    data = data.iloc[:,1:]
    data.index = pd.to_datetime(data.index)
    
    x=[]
    y=[]
    s_l = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]
    for elmt in range(sli):
        x.append(elmt+1)
        y.append(s_l[elmt])
    df = pd.Series(data=y,index=x)
    df = df.sort_index()
    df.index = range(1,len(df)+1)
    
    if  sel=='DTW':   
        fig_2,fig_3,fig_4,fig_5,memo = find_most_close(df,len(df),m_d,data,metric='dtw',loop=submit)
    elif  sel=='Euclidean':   
        fig_2,fig_3,fig_4,fig_5,memo = find_most_close(df,len(df),m_d,data,metric='euclidean')  
    return fig_2,fig_3,fig_4,fig_5,memo.reset_index().to_json(orient="split")
    
def find_most_close(seq1,win,min_d,data,metric='euclidean',loop=0):
    seq=[]
    for i in range(len(data.columns)): 
        seq.append(data.iloc[:,i])
    seq_n=[]
    for i in seq:
        seq_n.append((i-i.mean())/i.std())
    exclude,interv,n_test = int_exc(seq_n,win)
    
    
    if loop == 0:
        if seq1.var()!=0.0:
            seq1 = (seq1 - seq1.min())/(seq1.max() - seq1.min())
        seq1= np.array(seq1)
        tot=[]
        tot_seq=[]
        for i in range(len(n_test)):
            if i not in exclude:
                seq2 = n_test[i:i+win]
                seq2 = (seq2 - seq2.min())/(seq2.max() - seq2.min())
                try:
                    if metric=='euclidean':
                        dist = ed.distance(seq1,seq2)
                    elif metric=='dtw':
                        dist = dtw.distance(seq1,seq2)
                    tot.append([i,dist])
                    if (i+6 not in exclude) & (i<len(n_test)-win-6):
                        seq2 = n_test[i:i+win]
                        seq3 = n_test[i+win:i+win+6]
                        seq3 = (seq3 - seq2.min())/(seq2.max() - seq2.min())
                        tot_seq.append(seq3.tolist())
                    else:
                        tot_seq.append([float('NaN')]*6)
                except:
                    1

        tot_seq=pd.DataFrame(tot_seq,columns=[2,3,4,5,6,7])        
        tot=pd.DataFrame(tot)
        tot = pd.concat([tot,tot_seq],axis=1)
        tot = tot.sort_values([1])
        figlist=[]
        c=0
        for i in tot.iloc[:3,0].tolist():
            col = seq[bisect.bisect_right(interv, i)-1].name
            index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
            obs = data.loc[index_obs:,col].iloc[:win]
            fig_out = px.line(x=obs.index, y=obs,title=col+" <br><sup> Distance = "+str(round(tot.iloc[c,1],2))+"</sup>",markers='o')
            fig_out.update_layout(xaxis_title='Date',
                           yaxis_title='Units',title_x=0.5)
            figlist.append(fig_out)
            c=c+1
        tot = tot[tot[1]<min_d]
        memo = pd.DataFrame(index=range(18))
        if len(tot)>0:
            mean_f = tot.iloc[:,2:].mean()
            std_f = (1.96*tot.iloc[:,2:].std())/np.sqrt(len(tot))
            std_f = std_f.fillna(0) # if only one obs
            x_c = range(len(seq1)+len(mean_f))
            y_c = pd.concat([pd.Series(seq1),mean_f])
            fig_out = px.line(x=x_c, y=y_c,title="Predicted dynamic")
            fig_out.update_layout(xaxis_title='Months',
                           yaxis_title='Normalized Units',title_x=0.5)
            fig_out.add_scatter(x =pd.Series(range(len(seq1)-1,len(seq1)+len(mean_f))), y = pd.Series([y_c.iloc[-7]]+(mean_f+std_f).tolist()),
                           mode = 'lines',showlegend=True,opacity=0.2,name='Confidence Interval 95%').update_traces(marker=dict(color='red'))
            fig_out.add_scatter(x =pd.Series(range(len(seq1)-1,len(seq1)+len(mean_f))), y =pd.Series([y_c.iloc[-7]]+(mean_f-std_f).tolist()),
                           mode = 'lines',showlegend=False,opacity=0.2,name='Confidence Interval 95%').update_traces(marker=dict(color='red'))
            fig_out.add_scatter(x =pd.Series(range(len(seq1),len(seq1)+len(mean_f))), y = mean_f,
                           mode = 'lines+markers',showlegend=False,name='Forecast').update_traces(marker=dict(color='red'))
            fig_out.add_scatter(x =pd.Series(range(len(seq1)-1,len(seq1)+1)), y = y_c.iloc[-7:-5],
                           mode = 'lines',showlegend=False,name='Forecast').update_traces(marker=dict(color='red'))
            figlist.append(fig_out)
            for i in tot.iloc[:,0].tolist() :
                if (i+6 not in exclude) & (i<len(n_test)-win-6):
                    col = seq[bisect.bisect_right(interv, i)-1].name
                    index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
                    
                    obs = data.loc[index_obs:,col].iloc[:win].tolist()
                    while len(obs)<win:
                        obs=obs+[float('NaN')]
                    obs= obs+ data.loc[index_obs:,col].iloc[win:win+6].tolist() 
                    memo=pd.concat([memo,pd.Series(obs,name=col+'-'+str(index_obs.year)+'/'+str(index_obs.month))],axis=1)  
            memo=memo.dropna(axis=0,how='all')
        else : 
            x_c = range(len(seq1))
            y_c = pd.Series(seq1)
            fig_out = px.line(x=x_c, y=y_c,title="Predicted dynamic",markers='o')
            fig_out.add_annotation(x=(len(seq1)-1)/2, y=0.65,text="No patterns found",showarrow=False)
            figlist.append(fig_out)  
        figlist.append(memo)
    else:
        if seq1.var()!=0.0:
            seq1 = (seq1 - seq1.min())/(seq1.max() - seq1.min())
        seq1= np.array(seq1)
        tot=[]
        tot_seq=[]
        for lop in range(int(-loop),int(loop)+1):
            exclude,interv,n_test = int_exc(seq_n,win+lop)
            for i in range(len(n_test)):
                if i not in exclude:
                    seq2 = n_test[i:i+int(win+lop)]
                    seq2 = seq2 = (seq2 - seq2.min())/(seq2.max() - seq2.min())
                    try:
                        dist = dtw.distance(seq1,seq2)
                        tot.append([i,dist,win+lop])
                        if (i+6 not in exclude) & (i<len(n_test)-win-6):
                            seq2 = n_test[i:i+int(win+lop)]
                            seq3 = n_test[i+int(win+lop):i+int(win+lop)+6]
                            seq3 = (seq3 - seq2.min())/(seq2.max() - seq2.min())
                            tot_seq.append(seq3.tolist())
                        else:
                            tot_seq.append([float('NaN')]*6)
                    except:
                        1
        tot_seq=pd.DataFrame(tot_seq,columns=[3,4,5,6,7,8])        
        tot=pd.DataFrame(tot)
        tot = pd.concat([tot,tot_seq],axis=1)
        tot = tot.sort_values([1])
        figlist=[]
        li=[]
        c_lo=0
        while len(figlist)<3:
            i = tot.iloc[c_lo,0]
            win_l = int(tot.iloc[c_lo,2])
            exclude,interv,n_test = int_exc(seq_n,win_l)
            col = seq[bisect.bisect_right(interv, i)-1].name
            index_obs = seq[bisect.bisect_right(interv, i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
            obs = data.loc[index_obs:,col].iloc[:win_l]
            flag_ok=True
            if c_lo!=0:
                for ran in range(len(li)):
                    if col+'-'+str(index_obs.year) in li:
                        flag_ok=False
            if flag_ok==True:        
                fig_out = px.line(x=obs.index, y=obs,title=col+" <br><sup> d = "+str(tot.iloc[c_lo,1])+"</sup>",markers='o')
                fig_out.update_layout(xaxis_title='Date',
                               yaxis_title='Units',title_x=0.5)
                figlist.append(fig_out)
                li.append(col+'-'+str(index_obs.year))
            c_lo=c_lo+1
        tot = tot[tot[1]<min_d]
        memo = pd.DataFrame(index=range(18))
        if len(tot)>0:
            mean_f = tot.iloc[:,3:].mean()
            std_f = (1.96*tot.iloc[:,3:].std())/np.sqrt(len(tot))
            std_f = std_f.fillna(0) # if only one obs
            x_c = range(len(seq1)+len(mean_f))
            y_c = pd.concat([pd.Series(seq1),mean_f])
            fig_out = px.line(x=x_c, y=y_c,title="Predicted dynamic")
            fig_out.update_layout(xaxis_title='Months',
                           yaxis_title='Normalized Units',title_x=0.5)
            fig_out.add_scatter(x =pd.Series(range(len(seq1)-1,len(seq1)+len(mean_f))), y = pd.Series([y_c.iloc[-7]]+(mean_f+std_f).tolist()),
                           mode = 'lines',showlegend=True,opacity=0.2,name='Confidence Interval 95%').update_traces(marker=dict(color='red'))
            fig_out.add_scatter(x =pd.Series(range(len(seq1)-1,len(seq1)+len(mean_f))), y =pd.Series([y_c.iloc[-7]]+(mean_f-std_f).tolist()),
                           mode = 'lines',showlegend=False,opacity=0.2,name='Confidence Interval 95%').update_traces(marker=dict(color='red'))
            fig_out.add_scatter(x =pd.Series(range(len(seq1),len(seq1)+len(mean_f))), y = mean_f,
                           mode = 'lines+markers',showlegend=False,name='Forecast').update_traces(marker=dict(color='red'))
            fig_out.add_scatter(x =pd.Series(range(len(seq1)-1,len(seq1)+1)), y = y_c.iloc[-7:-5],
                           mode = 'lines',showlegend=False,name='Forecast').update_traces(marker=dict(color='red'))
            figlist.append(fig_out)
            li=[]
            c_lo=0
            for k in range(len(tot)):
                i = tot.iloc[c_lo,0]
                win_l = int(tot.iloc[c_lo,2])
                exclude,interv,n_test = int_exc(seq_n,win_l)
                if (i+6 not in exclude) & (i<len(n_test)-win_l-6):
                    col = seq[bisect.bisect_right(interv, i)-1].name
                    index_obs = seq[bisect.bisect_right(interv,i)-1].index[i-interv[bisect.bisect_right(interv, i)-1]]
                    
                    if col+'-'+str(index_obs.year) not in li:
                        obs = data.loc[index_obs:,col].iloc[:win_l].tolist()
                        while len(obs)<win+loop:
                            obs=obs+[float('NaN')]
                        obs= obs+ data.loc[index_obs:,col].iloc[win_l:win_l+6].tolist()  
                        memo=pd.concat([memo,pd.Series(obs,name=col+'-'+str(index_obs.year)+'/'+str(index_obs.month))],axis=1)   
                        li.append(col+'-'+str(index_obs.year))
                c_lo=c_lo+1
            memo=memo.dropna(axis=0,how='all')
        else : 
            x_c = range(len(seq1))
            y_c = pd.Series(seq1)
            fig_out = px.line(x=x_c, y=y_c,title="Predicted dynamic",markers='o')
            fig_out.add_annotation(x=(len(seq1)-1)/2, y=0.65,text="No patterns found",showarrow=False)
            figlist.append(fig_out)  
        figlist.append(memo)
    return figlist


def int_exc(seq_n,win):
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
    Input("btn_down", "n_clicks"),
    State('store2','data'),
    prevent_initial_call=True,
)
def func(n_clicks,data):
    df = pd.read_json(data,orient="split")
    return dcc.send_data_frame(df.to_csv, "Output.csv")

if __name__ == '__main__':
    app.run_server(debug=True)



