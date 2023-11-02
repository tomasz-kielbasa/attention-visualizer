import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
args = parser.parse_args()

model_name = args.model_name
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        model_name,
    )

@torch.no_grad()
@st.cache_data()
def get_attention_weights_and_tokens(text):
    tokenized = tokenizer(text, return_tensors='pt')
    tokens = [tokenizer.decode(token) for token in tokenized.input_ids[0]]
    tokenized = tokenized.to(device)
    output = model(**tokenized, output_attentions=True)
    attentions = [attention.to(torch.float32) for attention in output.attentions]
    return attentions, tokens

model = load_model()
tokenizer = load_tokenizer()

st.title('Attention visualizer')
text = st.text_area('Write your text here and see attention weights.')
layer = st.slider(
    'Which layer do you want to see?',
    min_value=1,
    max_value=model.config.num_hidden_layers
) - 1

head = st.select_slider(
    'Which head do you want to see?',
    options = ['Average'] + list(range(1, model.config.num_attention_heads + 1))
)
if text:
    attentions, tokens = get_attention_weights_and_tokens(text)
    if head == 'Average':
        weights = attentions[layer].cpu()[0].mean(dim=0)
    else:
        weights = attentions[layer].cpu()[0][head - 1]
    fig = px.imshow(
        weights,
    )
    fig.update_layout(xaxis={
            'ticktext': tokens,
            'tickvals': list(range(len(tokens))),
        }, yaxis={
            'ticktext': tokens,
            'tickvals': list(range(len(tokens))),
        },
        height=800,
    )

    st.plotly_chart(fig)