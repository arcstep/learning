import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="qwen3:30b-a3b", token="***").launch()