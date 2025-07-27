import gradio as gr
import time

custom_css = """
/* 完全移除整个 footer */
footer {
    display: none !important;
}
"""

def echo(message, history, system_prompt, tokens):
    response = f"System prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i+1]

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 我的应用")
    system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
    slider = gr.Slider(10, 100, render=False)

    gr.ChatInterface(
        echo, additional_inputs=[system_prompt, slider], type="messages"
    )

    gr.HTML("""
    <div style="text-align: center; padding: 15px; color: #999;">
        © 2023 我的应用 | 版本 1.0
    </div>
    """)

demo.launch()