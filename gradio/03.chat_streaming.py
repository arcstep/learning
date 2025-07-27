import time
import gradio as gr

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.1)
        yield "You typed: " + message[: i+1]

gr.ChatInterface(
    fn=slow_echo, 
    type="messages",
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="快点问我问题", container=False, scale=7),
    title="智能助手",
    description="我是一个智能助手，你可以问我任何问题",
    theme="ocean",
    examples=["你好", "我帅吗?", "你想跟我学做菜不?"],
    cache_examples=True,
).launch()