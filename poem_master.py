import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from utils.auto_poem import AutoPoem 
from typing import Dict, Optional

class PoemMaster:
    def __init__(
            self,
            config_path: str = './pretrained_weights/bert_chinese_wwm/config.json',
            checkpoint_path: str = './pretrained_weights/bert_chinese_wwm/pytorch_model.bin',
            dict_path: str = './pretrained_weights/bert_chinese_wwm/vocab.txt',
            model_weights_path: str = './best_model.pt',
            device: Optional[str] = None
        ):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[PoemMaster] Using device: {self.device}")

        print("[PoemMaster] Loading vocabulary...")
        token_dict, keep_tokens = load_vocab(
            dict_path=dict_path,
            simplified=True,
            startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
        )
        self.tokenizer = Tokenizer(token_dict, do_lower_case=True)
        print("[PoemMaster] Vocabulary loaded.")

        print("[PoemMaster] Building model...")
        self.model = build_transformer_model(
            config_path,
            checkpoint_path,
            application='unilm',
            keep_tokens=keep_tokens,
            with_mlm=True,
            add_trainer=True 
        ).to(self.device)
        print(f"[PoemMaster] Loading fine-tuned weights from {model_weights_path}...")
        self.model.load_weights(model_weights_path)
        self.model.eval()
        print("[PoemMaster] Model built and weights loaded.")

        self.poem_type_mapping = {
            "五言绝句": "jueju_5",
            "七言绝句": "jueju_7",
            "五言律诗": "lishi_5",
            "七言律诗": "lishi_7",
        }
        
        self.autopoems: Dict[str, AutoPoem] = {}
        print("[PoemMaster] Initializing AutoPoem instances...")
        for pretty_name, internal_name in self.poem_type_mapping.items():
            print(f"[PoemMaster] Initializing AutoPoem for {pretty_name} ({internal_name})...")
            self.autopoems[internal_name] = AutoPoem(
                model=self.model,
                tokenizer=self.tokenizer,
                bos_token_id=self.tokenizer._token_start_id,
                eos_token_id=self.tokenizer._token_end_id,
                max_length=128,
                max_new_tokens=84, # 确保此值足够生成最长的诗
                device=self.device,
                poem_type=internal_name
            )
        print("[PoemMaster] All AutoPoem instances initialized.")

    def generate(self, text_input: str, top_k: int = 8, top_p: float = 0.95, temperature: float = 1) -> str:
        """
        生成诗歌。
        Parameters:
            text_input: 输入字符串，格式为 "诗题&&格律"，例如 "大雪满边城&&五言绝句"
            top_k: 生成时考虑的候选词数量
            top_p: 生成时考虑的候选词概率阈值
            temperature: 生成时使用的温度参数，控制生成的随机性
        Raises:
            ValueError: 如果输入格式错误或指定的格律不受支持
            RuntimeError: 如果未能找到对应的 AutoPoem 实例
        """
        try:
            title, poem_style_pretty = text_input.split("&&")
            title = title.strip()
            poem_style_pretty = poem_style_pretty.strip()
        except ValueError:
            raise ValueError("输入格式错误，应为 '诗题&&格律', 例如: '大雪满边城&&五言绝句'")

        internal_poem_type = self.poem_type_mapping.get(poem_style_pretty)
        if not internal_poem_type:
            supported_types = ", ".join(self.poem_type_mapping.keys())
            raise ValueError(f"不支持的格律: {poem_style_pretty}. 支持的格律有: {supported_types}")

        autopoem_instance = self.autopoems.get(internal_poem_type)
        if not autopoem_instance:
            # This case should not happen if __init__ is correct
            raise RuntimeError(f"未能找到格律 {internal_poem_type} 对应的 AutoPoem 实例。")
        
        print(f"[PoemMaster] Generating poem with title='{title}', style='{poem_style_pretty}' (internal: '{internal_poem_type}')")
        try:
            output = autopoem_instance.generate(text_input, top_k=top_k, top_p=top_p, temperature=temperature)
            print(f"[PoemMaster] Generation completed successfully.")
        except Exception as e:
            print(f"[PoemMaster] Error during generation: {e}")
        return output

if __name__ == '__main__':
    print("--- PoemMaster Test ---")
    try:
        master = PoemMaster()
        
        test_inputs = [
            "大雪满边城&&五言绝句",
            "春江花月&&七言绝句",
            "登高望远&&五言律诗",
            "秋日&&七言律诗"
        ]
        
        for test_input in test_inputs:
            print(f"\n输入: {test_input}")
            try:
                poem = master.generate(test_input, top_k=8, top_p=0.95, temperature=1)
                print(f"输出:\n{poem}")
            except Exception as e:
                print(f"生成错误: {e}")
                import traceback
                traceback.print_exc()

        print("\n测试错误输入:")
        try:
            master.generate("无效输入")
        except ValueError as e:
            print(f"捕获到预期的错误: {e}")

        try:
            master.generate("测试&&未知格律")
        except ValueError as e:
            print(f"捕获到预期的错误: {e}")
            
    except Exception as e:
        print(f"初始化 PoemMaster 时发生错误: {e}")
        import traceback
        traceback.print_exc()