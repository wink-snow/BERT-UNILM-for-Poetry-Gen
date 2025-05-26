from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
import torch

from utils.auto_poem import AutoPoem

config_path = './pretrained_weights/bert_chinese_wwm/config.json'
checkpoint_path = './pretrained_weights/bert_chinese_wwm/pytorch_model.bin'
dict_path = './pretrained_weights/bert_chinese_wwm/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,
    with_mlm=True,
    add_trainer=True
).to(device)

model.load_weights('./best_model.pt')

autopoems = [
    AutoPoem(model, tokenizer, bos_token_id=tokenizer._token_start_id, eos_token_id=tokenizer._token_end_id,
            max_length=128, max_new_tokens=84, device=device, poem_type='jueju_5'),
    AutoPoem(model, tokenizer, bos_token_id=tokenizer._token_start_id, eos_token_id=tokenizer._token_end_id,
            max_length=128, max_new_tokens=84, device=device, poem_type='jueju_7'),
    AutoPoem(model, tokenizer, bos_token_id=tokenizer._token_start_id, eos_token_id=tokenizer._token_end_id,
            max_length=128, max_new_tokens=84, device=device, poem_type='lishi_5'),
    AutoPoem(model, tokenizer, bos_token_id=tokenizer._token_start_id, eos_token_id=tokenizer._token_end_id,
            max_length=128, max_new_tokens=84, device=device, poem_type='lishi_7')
]

test_seq = ["大雪满边城&&五言绝句", '大雪满边城&&七言绝句', '大雪满边城&&五言律诗', '大雪满边城&&七言律诗']
for seq, autopoem in zip(test_seq, autopoems):
    print("输入：%s" % seq)
    try:
        output = autopoem.generate(seq, top_k=8, top_p=0.95, temperature=0.8)
        print("输出：%s" % output)
    except Exception as e:
        print(f"Error during generation for '{seq}': {e}")
        import traceback
        traceback.print_exc()
    