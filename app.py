###################################################
# Outlines the UI created with Dash               #
# for OCR text extraction                         #
###################################################

# Load the necessary modules 
import datetime
import dash
from dash.dependencies import Input, Output, State
from dash_extensions import Download
from dash import dcc, html
import io
import os
import base64
from PIL import Image
import numpy as np
from OCR import get_text
from transformers_model import Predictor as TransformersPredictor


# Load models first 
transformers_predictor = TransformersPredictor()

# Global variables
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, title='Hateful Meme Classification', external_stylesheets=external_stylesheets, suppress_callback_exceptions=True, prevent_initial_callbacks=True)

app.layout = html.Div([

    ### code for the HEADER
    html.Div([
        html.H1(
            "Hateful Meme Classification",
            id='title-header',
            style={
                'textAlign': 'center'
            }
        ), html.H5(
            "A CS 6422 Project",
            style={
                'textAlign': 'center'
            }
        ), 
    ],  style={
            'textAlign': 'center',
            'padding-top': 15,
            'padding-bottom': 15
        }),
    ### end code for the HEADER
  
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
        },
        # Do not allow multiple files to be uploaded
        multiple=False
    ), dcc.Loading(
            id="loading-1",
            type="cube",
            fullscreen=True,
            children=html.Div(id="output-image-upload"),
    )], style={
        'text-align': 'center',
        'margin-left': '20%',
        'margin-right': '20%',
        'padding-top': 0
    }),
])


def parse_contents(contents, filename, date):
    """
    Converts input image encodes as b64 string to 
    numpy array then runs OCR.py on that array and 
    returns a Div with the image and extracted text
    contents: image inputed by user as b64 
    filename: filename for image
    date: date uploaded
    Div: returns Div with image and extracted text
    """
    b64_img = contents.split(',')[1]
    decoded = base64.b64decode(b64_img)
    buffer = io.BytesIO(decoded)
    im = Image.open(buffer)
    im.save("uploaded_image.png")
    arr = np.asarray(im)
    text = get_text(arr)
    contents = resize_input(im)

    classification, transformers_probs = transformers_predictor.evaluate("uploaded_image.png",text)
    
    return html.Div([
        html.H5('Original Filename: ' + filename, id='filename', key=filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={'padding-top': '1%'}),
        html.Br(),html.Br(),
        html.H5("Detected Text: " + text, id='txt', key=text),
        html.H5("Image + Text Classification: " + classification, id='classification', key=classification),
        html.H5("Image + Text Probability: " + str(transformers_probs), id='transformers_probs', key=str(transformers_probs)),
        #html.Div([html.Button("Download .txt File", id="btn"), Download(id="download")]),
        html.Hr(),
    ], style={
        'text-align': 'center',
    })

def resize_input(im):
    """
    Helper function that resizes an image
    if it is too big to fit on the screen
    im: image inputed by the user
    resized_im: resized image
    """
    resized_im = im
    if im.size[0] > 1000:
        ratio = im.size[0] / 1000
        resized_im = im.resize((round(im.size[0] / ratio), round(im.size[1] / ratio)))
    return resized_im

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(contents, filename, date):
    """
    Updates the screen based on the image 
    and extracted text
    contents: image as b64
    filename: filename of image
    date: date uploaded
    children: returned Div from image and extraacted text
    """
    if contents is not None:
        children = [parse_contents(contents, filename, date)]
        return children

@app.callback(Output("download", "data"), [Input("btn", "n_clicks")], State("txt", "key"), State('filename', 'key'),)
def download_text(n_clicks, text, filename):
    """
    Downloads the text as a txt file
    n_clicks: number of times download button
    has been pressed
    text: extracted text
    filename: filename of image
    returns a txt file with the same name as filename
    but with a .txt extension 
    """
    return dict(content=text, filename=filename+'.txt')

if __name__ == '__main__':
    app.run_server(debug=True)