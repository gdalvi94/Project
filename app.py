# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:17:24 2022

@author: Preshita
"""

import pandas as pd
import numpy as np
import pickle as pkl
import webbrowser
import dash_table
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px

from wordcloud import WordCloud
import matplotlib.pyplot as plt




data = pd.read_csv(r"final_reviews.csv")
scrape = pd.read_csv(r"clothes_final.csv")
scrape1 = pd.read_csv(r"jewlery_final.csv")
scrape2 = pd.read_csv(r"shoes_final.csv")

#scrape2['Positivity'] = np.where(scrape2['Rating'] > 3, 1, 0)
#scrape2.to_csv('shoes_final1.csv',index=False)
#
#x2011 = scrape2["Review"][scrape2["Rating"]==1]
#
#plt.subplots(figsize = (8,8))
#
#wordcloud1 = WordCloud (
#background_color = 'white',
#width = 512,
#height = 384
#).generate(' '.join(x2011))
#fig1=plt.imshow(wordcloud1)
#fig1=plt.axis('off')
#plt.savefig('assets/shoes_wordcloud.png')

def load_model():
    global df
    df = pd.read_csv(r"final_reviews.csv")
    
    global  pickle_model
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename,'rb') as file:
        pickle_model = pkl.load(file)
    
    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pkl.load(file)

def create_pie_clothes():
    result = []
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    for ind in scrape.index:
        rev = transformer.fit_transform(loaded_vec.fit_transform([scrape['Review'][ind]]))
        result.append(pickle_model.predict(rev))
    return result
labels = ["Positive","Negative"]
result = create_pie_clothes()
values=[result.count(1), result.count(0)]
graphc = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])  
rating = ['5', '4', '3', '2', '1']
count = [len(scrape[scrape.Rating==5]),len(scrape[scrape.Rating==4]),len(scrape[scrape.Rating==3]),len(scrape[scrape.Rating==2]),len(scrape[scrape.Rating==1])]
figc = go.Figure(data=[go.Bar(x = rating,y = count)])
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_pie_jewlery():
    result = []
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    for ind in scrape1.index:
        rev = transformer.fit_transform(loaded_vec.fit_transform([scrape1['Review'][ind]]))
        result.append(pickle_model.predict(rev))
    return result
labels = ["Positive","Negative"]
result1 = create_pie_jewlery()
values=[result1.count(1), result1.count(0)]
graphj = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])  
rating = ['5', '4', '3', '2', '1']
count = [len(scrape1[scrape1.Rating==5]),len(scrape1[scrape1.Rating==4]),len(scrape1[scrape1.Rating==3]),len(scrape1[scrape1.Rating==2]),len(scrape1[scrape1.Rating==1])]
figj = go.Figure(data=[go.Bar(x = rating,y = count)])

def create_pie_shoes():
    result = []
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    for ind in scrape2.index:
        rev = transformer.fit_transform(loaded_vec.fit_transform([scrape2['Review'][ind]]))
        result.append(pickle_model.predict(rev))
    return result
labels = ["Positive","Negative"]
result2 = create_pie_clothes()
values=[result2.count(1), result2.count(0)]
graphs = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])  
rating = ['5', '4', '3', '2', '1']
count = [len(scrape2[scrape2.Rating==5]),len(scrape2[scrape2.Rating==4]),len(scrape2[scrape2.Rating==3]),len(scrape2[scrape2.Rating==2]),len(scrape2[scrape2.Rating==1])]
figs = go.Figure(data=[go.Bar(x = rating,y = count)])
    
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    main_layout = html.Div([
    dbc.Container(
        [
            html.H1("Sentiment Analysis", style={"text-align": 'center'}),
           
        ],
        fluid=True,
        className="p-3 bg-light rounded-3"
    ),
    dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Clothes', value='tab-1-example-graph'),
        dcc.Tab(label='Jewelry', value='tab-2-example-graph'),
        dcc.Tab(label='Shoes', value='tab-3-example-graph'),

    ]),
    html.Div(id='tabs-content-example-graph')
    ])
    return main_layout

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))

def render_content(tab):
    if tab == 'tab-1-example-graph':
        html.H3('Clothes'),
        row = html.Div(
            [
                 dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Pie Chart', id='pieh1',style={'text-align':'center', 'margin-top':'30px'})])),
                        dbc.Col(html.Div([html.H1(children='Product', id='line1',style={'text-align':'center', 'margin-top':'30px'})])),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure = graphc, style={'width': '100%','height':700,},className = 'd-flex justify-content-center')),
                        dbc.Col(html.Img(src = app.get_asset_url ('clothesproduct.jpg'),style={'width': '600px', 'height': '600px','margin':'0 auto', 'margin-top':'20px'},className = 'd-flex justify-content-center')),
                    ]
                ), html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Bar Graph', id='cloud',style={'text-align':'center'})])),
                        dbc.Col(html.Div([html.H1(children='World Cloud', id='drop-down',style={'text-align':'center'})])),

                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure = figc, style={'width': '100%','height':700,},className = 'd-flex justify-content-center')),
                        dbc.Col(html.Img(src = app.get_asset_url ('clothes_wordcloud.png'),style={'width': '800px', 'height': '800px','display':'block','margin':'0 auto'})),
                    ]
                ), html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Check Review', id='pieh1',style={'margin-left':'20px','text-align':'center'})]))
                    ]
                ),
                html.Br(),
                html.Div(
                    [
                        html.Div(
                                dbc.Container([
                                        dbc.FormGroup([
                                                    dcc.Dropdown(id='dropdown', 
                                                     options=[{'label': i, 'value': i} for i in scrape.Review.unique()], 
                                                     placeholder='Enter your review...',
                                                     optionHeight=80,
                                                     style={'width': '100%', 'height': 50})    ,   
                                                ]),
                                        html.Div([html.Button(children='Find Review', id='button_review',n_clicks=0,
                                        style={'height':'40px','backgroundColor':'#4CAF50'})],className='text-center')
                                        ])
                                ),
                        
                        html.Div(html.Div([html.H2(children=None, id='result',style={'text-align':'center'})]))
                    ]
                ,style={'margin':'0 auto'}), html.Hr(),
                html.Div(children=[
                    dcc.Dropdown(
                        id='demo-dropdown',
                        options=[
                            {'label': '5', 'value': 5},
                            {'label': '4', 'value': 4},
                            {'label': '3', 'value': 3},
                            {'label': '2', 'value': 2},
                            {'label': '1', 'value': 1},
                        ],
                        value=5,
                        style={'margin-bottom':'30px'}
                    ),
                    dash_table.DataTable(id='dd-output-container',
                                         data=scrape.to_dict('records'),
                                         columns=[{'name': c,'id': c} for c in scrape.columns.values],
                                         
                                         style_table={
                        'height': 300,
                    },
                    style_data={
                        'width': '50px', 'minWidth': '50px', 'maxWidth': '50px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                         'border': '1px solid blue' },
                    style_header={ 'border': '1px solid pink' },
                    
                    )
                    
                ],style={'margin-left':'60px','margin-right':'60px'})
            ]
        )
        return row
    elif tab == 'tab-2-example-graph':
        html.H3('Jewellery'),
        row = html.Div(
            [
                 dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Pie Chart', id='pieh1',style={'text-align':'center', 'margin-top':'30px'})])),
                        dbc.Col(html.Div([html.H1(children='Product', id='line1',style={'text-align':'center', 'margin-top':'30px'})])),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure = graphj, style={'width': '100%','height':700,},className = 'd-flex justify-content-center')),
                        dbc.Col(html.Img(src = app.get_asset_url ('jewelleryproduct.jpg'),style={'width': '600px', 'height': '600px','margin':'0 auto', 'margin-top':'20px'},className = 'd-flex justify-content-center')),
                    ]
                ), html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Bar Graph', id='cloud',style={'text-align':'center'})])),
                        dbc.Col(html.Div([html.H1(children='Word Cloud', id='drop-down',style={'text-align':'center'})])),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure = figj, style={'width': '100%','height':700,},className = 'd-flex justify-content-center')),
                        dbc.Col(html.Img(src = app.get_asset_url ('jewlery_wordcloud.png'),style={'width': '800px', 'height': '800px','display':'block','margin':'0 auto'})),
                    ]
                ), html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Check Review', id='pieh1',style={'margin-left':'20px','text-align':'center'})]))
                    ]
                ),
                html.Br(),
                html.Div(
                    [
                        html.Div(
                                dbc.Container([
                                        dbc.FormGroup([
                                                    dcc.Dropdown(id='dropdown', 
                                                     options=[{'label': i, 'value': i} for i in scrape1.Review.unique()], 
                                                     placeholder='Enter your review...',
                                                     optionHeight=80,
                                                     style={'width': '100%', 'height': 50})    ,   
                                                ]),
                                        html.Div([html.Button(children='Find Review', id='button_review',n_clicks=0,
                                        style={'height':'40px','backgroundColor':'#4CAF50'})],className='text-center')
                                        ])
                                ),
                        
                        html.Div(html.Div([html.H2(children=None, id='result',style={'text-align':'center'})]))
                    ]
                ,style={'margin':'0 auto'}), html.Hr(style={'height':'10px'}),  html.Hr(),
                html.Div(children=[
                    dcc.Dropdown(
                        id='demo-dropdown1',
                        options=[
                            {'label': '5', 'value': 5},
                            {'label': '4', 'value': 4},
                            {'label': '3', 'value': 3},
                            {'label': '2', 'value': 2},
                            {'label': '1', 'value': 1},
                        ],
                        value=5,
                        style={'margin-bottom':'30px'}
                    ),
                    dash_table.DataTable(id='dd-output-container1',
                                         data=scrape1.to_dict('records'),
                                         columns=[{'name': c,'id': c} for c in scrape1.columns.values],
                                         
                                         style_table={
                        'height': 300,
                    },
                    style_data={
                        'width': '50px', 'minWidth': '50px', 'maxWidth': '50px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                         'border': '1px solid blue' },
                    style_header={ 'border': '1px solid pink' },
                    
                    )
                    
                ],style={'margin-left':'60px','margin-right':'60px'})
            ]
        )
        return row
    
    elif tab == 'tab-3-example-graph':
        html.H3('Shoes'),
        row = html.Div(
            [
                 dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Pie Chart', id='pieh1',style={'text-align':'center', 'margin-top':'30px'})])),
                        dbc.Col(html.Div([html.H1(children='Product', id='line1',style={'text-align':'center', 'margin-top':'30px'})])),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure = graphs, style={'width': '100%','height':700,},className = 'd-flex justify-content-center')),
                        dbc.Col(html.Img(src = app.get_asset_url ('shoesproduct.jpg'),style={'width': '600px', 'height': '600px','margin':'0 auto', 'margin-top':'20px'},className = 'd-flex justify-content-center')),
                    ]
                ), html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Bar Graph', id='cloud',style={'text-align':'center'})])),
                        dbc.Col(html.Div([html.H1(children='World Cloud', id='drop-down',style={'text-align':'center'})])),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure = figs, style={'width': '100%','height':700,},className = 'd-flex justify-content-center')),
                        dbc.Col(html.Img(src = app.get_asset_url ('shoes_wordcloud.png'),style={'width': '800px', 'height': '800px','display':'block','margin':'0 auto'})),
                    ]
                ), html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(html.Div([html.H1(children='Check Review', id='pieh1',style={'margin-left':'20px','text-align':'center'})]))
                    ]
                ),
                html.Br(),
                html.Div(
                    [
                        html.Div(
                                dbc.Container([
                                        dbc.FormGroup([
                                                    dcc.Dropdown(id='dropdown', 
                                                     options=[{'label': i, 'value': i} for i in scrape2.Review.unique()], 
                                                     placeholder='Enter your review...',
                                                     optionHeight=80,
                                                     style={'width': '100%', 'height': 50})    ,   
                                                ]),
                                        html.Div([html.Button(children='Find Review', id='button_review',n_clicks=0,
                                        style={'height':'40px','backgroundColor':'#4CAF50'})],className='text-center')
                                        ])
                                ),
                        
                        html.Div(html.Div([html.H2(children=None, id='result',style={'text-align':'center'})]))
                    ]
                ,style={'margin':'0 auto'}), html.Hr(style={'height':'10px'}),  html.Hr(),
                html.Div(children=[
                    dcc.Dropdown(
                        id='demo-dropdown2',
                        options=[
                            {'label': '5', 'value': 5},
                            {'label': '4', 'value': 4},
                            {'label': '3', 'value': 3},
                            {'label': '2', 'value': 2},
                            {'label': '1', 'value': 1},
                        ],
                        value=5,
                        style={'margin-bottom':'30px'}
                    ),
                    dash_table.DataTable(id='dd-output-container2',
                                         data=scrape2.to_dict('records'),
                                         columns=[{'name': c,'id': c} for c in scrape2.columns.values],
                                         
                                         style_table={
                        'height': 300,
                    },
                    style_data={
                        'width': '50px', 'minWidth': '50px', 'maxWidth': '50px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                         'border': '1px solid blue' },
                    style_header={ 'border': '1px solid pink' },
                    
                    )
                    
                ],style={'margin-left':'60px','margin-right':'60px'})
            ]
        )
        return row

@app.callback(
    Output('result', 'children'),
    [
    Input('button_review', 'n_clicks'),
    State('dropdown','value'),

    ]
    )   
def update_app_ui(n_click,textarea_value):
    result_list = check_review(textarea_value)
    if n_click > 0:
        if (result_list[0] == 0 ):
            result = 'Negative'
        elif (result_list[0] == 1 ):
            result = 'Positive'
        else:
            result = 'Unknown'
        
    return result 

@app.callback(
    dash.dependencies.Output('dd-output-container', 'data'),
    [dash.dependencies.Input('demo-dropdown', 'value')])

def update_output_clothes(value):
    dfs = scrape.loc[scrape['Rating'] == value]
    return dfs.to_dict('records') 

@app.callback(
    dash.dependencies.Output('dd-output-container1', 'data'),
    [dash.dependencies.Input('demo-dropdown1', 'value')])
def update_output_jewellery(value):
    dfs = scrape1.loc[scrape1['Rating'] == value]
    return dfs.to_dict('records') 

@app.callback(
    dash.dependencies.Output('dd-output-container2', 'data'),
    [dash.dependencies.Input('demo-dropdown2', 'value')])
def update_output_shoes(value):
    dfs = scrape2.loc[scrape2['Rating'] == value]
    return dfs.to_dict('records') 

def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)
    
def main():  
    
    load_model()
    open_browser()
    global app
    app.layout = create_app_ui()
    app.title = 'Sentiments Analysis with Insights'
    app.run_server()

if __name__ == '__main__':
    main()