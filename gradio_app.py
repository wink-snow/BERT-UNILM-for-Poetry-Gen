import gradio as gr
from poem_master import PoemMaster
import traceback
import os

poem_generator = None
initialization_error_message = None

try:
    print("正在初始化 PoemMaster 以用于 Gradio 应用...")
    poem_generator = PoemMaster()
    print("PoemMaster 初始化成功。")
except Exception as e:
    initialization_error_message = f"模型初始化失败: {e}\n详细错误信息请查看控制台。"
    print("--- PoemMaster 初始化错误 ---")
    print(initialization_error_message)
    traceback.print_exc()
    print("-----------------------------")

def generate_poem_for_interface(poem_title: str, 
                                poem_style_pretty: str, 
                                top_k: int, 
                                top_p: float, 
                                temperature: float):
    """
    Gradio 调用的函数，用于生成诗歌。
    """
    if poem_generator is None:
        return f"错误：模型未能成功加载。\n{initialization_error_message}"
    
    if not poem_title or not poem_title.strip():
        return "错误：诗题不能为空，请输入诗题。"

    # PoemMaster 的 generate 方法期望的输入格式是 "诗题&&格律"
    text_input = f"{poem_title.strip()}&&{poem_style_pretty}"
    
    print(f"Gradio 请求: 诗题='{poem_title}', 格律='{poem_style_pretty}', Top-k={top_k}, Top-p={top_p}, Temperature={temperature}")
    print(f"格式化后送入 PoemMaster 的输入: '{text_input}'")

    try:
        generated_poem = poem_generator.generate(
            text_input=text_input,
            top_k=int(top_k),       # Gradio slider 可能返回 float
            top_p=float(top_p),
            temperature=float(temperature)
        )
        return generated_poem
    except ValueError as ve: 
        return f"输入错误: {str(ve)}"
    except Exception as e:
        print(f"在 Gradio 界面生成诗歌时发生错误: {e}")
        traceback.print_exc()
        return f"生成诗歌时发生内部错误: {str(e)}"

if poem_generator and hasattr(poem_generator, 'poem_type_mapping'):
    available_styles = list(poem_generator.poem_type_mapping.keys())
    default_style = available_styles[0] if available_styles else "五言绝句"
else:
    available_styles = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]
    default_style = "五言绝句"
    if initialization_error_message is None: 
        initialization_error_message = "警告：模型组件未完全加载，格律列表可能不完整或功能受限。"

input_poem_title = gr.Textbox(
    label="诗题 (Poem Title)", 
    placeholder="例如：大雪满边城"
)
input_poem_style = gr.Dropdown(
    label="格律 (Poem Style)", 
    choices=available_styles, 
    value=default_style
)
input_top_k = gr.Slider(
    minimum=1, 
    maximum=50, 
    value=8, 
    step=1, 
    label="Top-k",
    info="控制采样时考虑的最高概率候选词的数量。"
)
input_top_p = gr.Slider(
    minimum=0.01, 
    maximum=1.0, 
    value=0.95, 
    step=0.01, 
    label="Top-p (Nucleus Sampling)",
    info="控制采样时考虑的累积概率阈值，值越小生成的文本越集中。"
)
input_temperature = gr.Slider(
    minimum=0.1, 
    maximum=2.0, 
    value=0.8, 
    step=0.05, 
    label="Temperature (随机性)",
    info="控制生成文本的随机性，值越高越随机，越低越保守。"
)

output_generated_poem = gr.Textbox(
    label="生成的诗歌 (Generated Poem)", 
    lines=10,               
    interactive=False      
)

example_list = [
    ["大雪满边城", "五言绝句", 8, 0.95, 0.8],
    ["春江花月夜", "七言绝句", 10, 0.9, 0.75],
    ["登高", "五言律诗", 5, 0.92, 1.0],
    ["秋日即景", "七言律诗", 8, 0.95, 0.85],
    ["黄鹤楼", "七言律诗", 8, 0.95, 0.9],
]

# --- Gradio Interface ---
# 可选: gr.themes.Default(), gr.themes.Monochrome(), gr.themes.Soft(), gr.themes.Glass()
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
).set(
    # button_primary_background_fill="*primary_500",
    # button_primary_text_color="white",
)


interface = gr.Interface(
    fn=generate_poem_for_interface,
    inputs=[
        input_poem_title, 
        input_poem_style, 
        input_top_k, 
        input_top_p, 
        input_temperature
    ],
    outputs=output_generated_poem,
    title=" AI 古诗生成器 ",
    description=(
        "输入诗歌的主题（诗题）并选择期望的格律，AI 将会为您创作一首古诗。\n"
        "模型基于 bert4torch 和 UNILM 进行微调。初次加载可能需要一些时间，请耐心等待。\n"
        f"{(initialization_error_message) if initialization_error_message else ''}" 
    ),
    examples=example_list,
    theme=theme, 
    allow_flagging='never', 
    css="footer {display: none !important}"
)

if __name__ == '__main__':
    print("准备启动 Gradio 应用...")
    if poem_generator is None and initialization_error_message:
        print(f"警告: PoemMaster 未能成功初始化。Gradio 界面将启动，但生成功能将显示错误信息。")
        print(f"初始化错误详情: {initialization_error_message}")
    
    interface.launch()
    # interface.launch(share=True) 
    # interface.launch(server_name="0.0.0.0", server_port=7860)