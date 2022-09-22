from transformers import pipeline
import gradio as gr
from gradio.mix import Parallel, Series

import os
import warnings
warnings.filterwarnings("ignore")

# select Summariser Models that are top downloaded.Initialise the parallel interface

os.environ["CURL_CA_BUNDLE"]=""

io1 = gr.Interface.load('huggingface/sshleifer/distilbart-cnn-12-6')
io2 = gr.Interface.load("huggingface/facebook/bart-large-cnn")
io3 = gr.Interface.load("huggingface/google/pegasus-xsum")  
io4 = gr.Interface.load("huggingface/sshleifer/distilbart-cnn-6-6")                   

iface = Parallel(io1, io2, io3, io4,
                 theme='huggingface', 
                 inputs = gr.inputs.Textbox(lines = 10, label="Text"))

# iface.launch()

if __name__ == "__main__":
    app, local_url, share_url = iface.launch(share=True)