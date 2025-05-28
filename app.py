import gradio as gr
from poem_master import PoemMaster
import traceback
import os

poem_generator = None
initialization_error_message = None

try:
    print("正在初始化 PoemMaster 以用于 Gradio 应用...")
    poem_generator = PoemMaster(device='cpu')
    # poem_generator = PoemMaster()
    print("PoemMaster 初始化成功。")
except Exception as e:
    initialization_error_message = f"模型初始化失败: {e}\n详细错误信息请查看控制台。"
    print("--- PoemMaster 初始化错误 ---")
    print(initialization_error_message)
    traceback.print_exc()
    print("-----------------------------")

def generate_poem(
        poem_title: str, 
        poem_style_pretty: str, 
        top_k: int, 
        top_p: float, 
        temperature: float):
    """
    Gradio 调用的函数，用于生成诗歌。
    """
    if poem_generator is None:
        return  """<div style="color: red; text-align: center; font-size: 1.2em;">
                    <strong>错误：</strong>模型未能成功加载。<br>{}
                  </div>""".format(initialization_error_message if initialization_error_message else "")
    
    if not poem_title or not poem_title.strip():
        return """<div style="color: red; text-align: center; font-size: 1.2em;">
                    <strong>错误：</strong>诗题不能为空，请输入诗题。
                  </div>"""


    # PoemMaster 的 generate 方法期望的输入格式是 "诗题&&格律"
    text_input = f"{poem_title.strip()}&&{poem_style_pretty}"
    
    print(f"[Gradio 请求] 诗题='{poem_title}', 格律='{poem_style_pretty}', Top-k={top_k}, Top-p={top_p}, Temperature={temperature}")
    print(f"[Gradio 请求] 格式化后送入 PoemMaster 的输入: '{text_input}'")

    try:
        generated_poem = poem_generator.generate(
            text_input=text_input,
            top_k=int(top_k),       # Gradio slider 可能返回 float
            top_p=float(top_p),
            temperature=float(temperature)
        )
        print(f"[Gradio 响应] 生成的诗歌: {generated_poem}")
        # 格式化生成内容
        processed_poem = generated_poem.replace('\r\n', '\n').replace('\r', '\n')
        formatted_body = processed_poem.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n")
        lines = [line.strip() for line in formatted_body.strip().split('\n') if line.strip()]
        html_poem_body = "<br>".join(lines)
        
        markdown_output = f"""
            <div style="text-align: center; margin-bottom: 10px;">
            <h2 style="display: inline-block; margin-bottom: 15px; font-size: 2.2em; font-family: 'KaiTi', 'STKaiti', '楷体', serif; font-weight: bold; color: #333;">
                {poem_title.strip()}
            </h2>
            </div>
            <div style="font-size: 1.6em; line-height: 2.2em; text-align: center; white-space: pre-line; font-family: 'FangSong', 'STFangsong', '仿宋', serif; color: #444;">
                {html_poem_body}
            </div>
        """

        return markdown_output
    
    except ValueError as ve: 
        return f"""<div style="color: red; text-align: center; font-size: 1.2em;">
                    <strong>输入错误:</strong> {str(ve)}
                  </div>"""
    except Exception as e:
        print(f"在 Gradio 界面生成诗歌时发生错误: {e}")
        traceback.print_exc()
        return f"""<div style="color: red; text-align: center; font-size: 1.2em;">
                    <strong>生成诗歌时发生内部错误:</strong> {str(e)}
                  </div>"""

if poem_generator and hasattr(poem_generator, 'poem_type_mapping'):
    available_styles = list(poem_generator.poem_type_mapping.keys())
    default_style = available_styles[0] if available_styles else "五言绝句"
else:
    available_styles = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]
    default_style = "五言绝句"
    if initialization_error_message is None: 
        initialization_error_message = "警告：模型组件未完全加载，格律列表可能不完整或功能受限。"

custom_css = """
.gradio-container { 
    background-color: #fdfcf5 !important; /* Added !important for safety */
    font-family: 'Segoe UI', Tahoma, sans-serif, 'Noto Sans SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
}
textarea, input, .gradio-dropdown > div > input { 
    border-radius: 8px !important; 
    border: 1px solid #e0e0e0 !important;
}
button.custom-btn { 
    background-color: #6c5ce7 !important;
    color: white !important; 
    border-radius: 8px !important;
    padding: 10px 15px !important;
    font-size: 1.05em !important;
    border: none !important;
}
button.custom-btn:hover {
    background-color: #5a4bd7 !important;
}
.gradio-form > div > .label > span, .gradio-form > .label > span {
    background-color: #8a7ff0 !important;
    color: white !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    font-size: 0.9em !important;
    margin-bottom: 5px !important;
    display: inline-block !important;
}
div[data-testid="markdown"] { /* Affects Markdown components */
    padding: 15px;
    background-color: transparent; /* Ensure no conflicting background from markdown itself */
}
/* New styles for the main page title */
.page-main-title-container {
    text-align: center !important;
    margin-bottom: 10px !important;
    width: 100% !important;
}
.page-main-title-text {
    font-size: 2.5em !important;
    font-weight: bold !important;
    font-family: "KaiTi", "STKaiti", "楷体", "Songti SC", "STSong", "SimSun", serif !important; /* Keep KaiTi, add robust fallbacks */
    display: inline-block !important;
    color: #333 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(
    primary_hue="purple", secondary_hue="blue" 
).set(
    block_title_text_weight="600",
    block_label_text_weight="500",
    body_background_fill="#fdfcf5" 
)) as app:
    gr.HTML("""
        <div class='page-main-title-container'>
            <h1 class='page-main-title-text'>🪶 AI 古诗生成器</h1>
        </div>
    """)
    # gr.Markdown("<div style='text-align:center; margin-bottom:10px;'><span style='font-size:2.5em; font-weight:bold; font-family: \"KaiTi\", \"STKaiti\", \"楷体\", serif;'>🪶 AI 古诗生成器</span></div>")
    # gr.Markdown("<p style='text-align:center; font-size:1.1em; color:#555; margin-bottom:25px;'>输入一个诗题，选择格律，AI 将为您创作一首古诗。</p>")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2): 
            poem_title_input = gr.Textbox(
                label="诗题 (Poem Title)", 
                placeholder="例如：大雪满边城",
                elem_id="poem_title_input" 
            )
            poem_style_input = gr.Dropdown(
                label="格律 (Poem Style)", 
                choices=available_styles, 
                value=default_style,
                elem_id="poem_style_dropdown"
            )
            top_k_input = gr.Slider(
                minimum=1, 
                maximum=50, 
                value=8, 
                step=1, 
                label="Top-k",
                info="控制采样时考虑的最高概率候选词的数量。"
            )
            top_p_input = gr.Slider(
                minimum=0.01, 
                maximum=1.0, 
                value=0.95, 
                step=0.01, 
                label="Top-p (Nucleus Sampling)",
                info="控制采样时考虑的累积概率阈值"
            )
            temperature_input = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.8, 
                step=0.05, 
                label="Temperature (随机性)",
                info="控制生成文本的随机性，值越高越随机，越低越保守。"
            )
            generate_btn = gr.Button("生成诗歌", elem_classes="custom-btn") 
            
        with gr.Column(scale=3): 
            generated_poem_display = gr.Markdown(
                elem_id="generated_poem_output" # For specific CSS if needed
            )

    generate_btn.click(
        fn=generate_poem, 
        inputs=[poem_title_input, poem_style_input, top_k_input, top_p_input, temperature_input], 
        outputs=generated_poem_display
    )

    gr.Markdown("---") 
    
    gr.Markdown("<h3 style='font-family: \"Segoe UI\", Tahoma, sans-serif; color: #4a4a4a; margin-bottom: 10px; text-align:left;'><span style='background-color:#e6e0fa; padding: 3px 7px; border-radius:5px;'>≡ 点击示例快速体验：</span></h3>")

    gr.Examples(
        examples=[
            ["大雪满边城", "五言绝句", 8, 0.95, 0.8],
            ["大雪满边城", "七言绝句", 10, 0.9, 0.75],
            ["大雪满边城", "五言律诗", 8, 0.95, 0.85],
            ["大雪满边城", "七言律诗", 8, 0.95, 0.9],
            ["蜉蝣", "五言绝句", 10, 0.9, 0.8],
            ["剑来", "七言绝句", 10, 0.95, 0.8],
            ["大道朝天", "五言律诗", 12, 0.9, 0.75],
            ["岳阳楼宴客", "七言律诗", 8, 0.95, 0.9],
        ],
        inputs=[poem_title_input, poem_style_input, top_k_input, top_p_input, temperature_input],
        outputs=generated_poem_display,
        examples_per_page=4
    )
    
    gr.Markdown("---") # Separator
    gr.Markdown("<h3 style='font-family: \"Segoe UI\", Tahoma, sans-serif; color: #4a4a4a; margin-bottom:5px;'>📌 模型说明</h3>")
    gr.HTML("<div style='font-size:0.9em; color:#666; padding: 5px; border-left: 3px solid #8a7ff0; background-color:#f9f7ff; border-radius:4px;'><p style='margin:0;'>本模型基于 UNILM 与 bert4torch 微调。初次加载需时间。由于推理环境限制，耗时较长，请耐心等待。</p></div>")
    gr.Markdown("<p style='font-size:0.9em; color:#666;'>如果您对模型或代码有任何疑问或建议，请访问 <a href='https://github.com/wink-snow/BERT-UNILM-for-Poetry-Gen' target='_blank'>GitHub 仓库</a>。</p>")

if __name__ == '__main__':
    print("准备启动 Gradio 应用...")
    if poem_generator is None and initialization_error_message:
        print(f"警告: PoemMaster 未能成功初始化。Gradio 界面将启动，但生成功能将显示错误信息。")
        print(f"初始化错误详情: {initialization_error_message}")
    
    # interface.launch()
    # interface.launch(share=True) 
    # interface.launch(server_name="0.0.0.0", server_port=7860)
    app.launch()