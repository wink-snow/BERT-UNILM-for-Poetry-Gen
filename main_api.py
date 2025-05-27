from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from poem_master import PoemMaster

class PoemRequest(BaseModel):
    prompt: str # 格式: "诗题&&格律", e.g., "大雪满边城&&五言绝句"
    top_k: int = 8
    top_p: float = 0.95
    temperature: float = 0.8

class PoemResponse(BaseModel):
    input_prompt: str
    generated_poem: str
    model_device: str

poem_generator: Optional[PoemMaster] = None
model_device_name: str = "cpu"

@asynccontextmanager
async def startup_event(app: FastAPI):
    global poem_generator, model_device_name
    print("FastAPI application startup...")
    print("Initializing PoemMaster...")
    try:
        # poem_generator = PoemMaster(device='cpu') # 强制使用CPU
        poem_generator = PoemMaster()
        model_device_name = poem_generator.device
        print(f"PoemMaster initialized successfully on device: {model_device_name}.")
    except Exception as e:
        print(f"Error during PoemMaster initialization: {e}")
        poem_generator = None 
        import traceback
        traceback.print_exc()

    yield
    print("FastAPI application shutdown...")

app = FastAPI(
    title="AI Poem Generation API",
    description="一个使用bert4torch和UNILM模型生成古诗的API",
    version="0.1.0",
    lifespan=startup_event
)

@app.post("/generate_poem/", response_model=PoemResponse)
async def create_poem(request: PoemRequest):
    """
    接收诗歌创作请求，生成并返回诗歌。

    - **prompt**: 输入的文本，格式为 "诗题&&格律".
      例如: "大雪满边城&&五言绝句"
      支持的格律: 五言绝句, 七言绝句, 五言律诗, 七言律诗.
    - **top_k**: Top-k 采样参数.
    - **top_p**: Top-p (nucleus) 采样参数.
    - **temperature**: 温度参数，控制生成的多样性.
    """
    if poem_generator is None:
        raise HTTPException(status_code=503, detail="模型服务尚未准备好，请稍后再试或联系开发者检查服务器日志。")

    print(f"Received request: {request.prompt}")
    try:
        generated_text = poem_generator.generate(
            text_input=request.prompt,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature
        )
        return PoemResponse(
            input_prompt=request.prompt,
            generated_poem=generated_text,
            model_device=model_device_name
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during poem generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"作诗时发生内部错误: {str(e)}")

@app.get("/")
async def root():
    return {"message": "欢迎使用 AI 作诗 API! 请访问 /docs 查看API文档。"}

# --- How to Run ---
# pip install fastapi uvicorn[standard]
# 在终端中运行: uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
# 然后可以通过 http://localhost:8000/docs 访问API文档和测试界面。