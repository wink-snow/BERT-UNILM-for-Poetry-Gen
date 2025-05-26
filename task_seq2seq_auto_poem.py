#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.callbacks import Callback
from bert4torch.losses import CausalLMLoss
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import glob
import os
import time

from utils.load_data import load_poems_from_csv

DATA_ROOT = './data/processed/'
# DATA_ROOT = './test/data/'

# 基本参数
maxlen = 128
batch_size = 128
epochs = 6
LR = 1e-6
USE_SCHEDULER = False
GRAD_ACCUMULATION_STEPS = 1  # 梯度累积步数

TOP_K = 5  # beam search时的top_k

# bert配置
config_path = './pretrained_weights/bert_chinese_wwm/config.json'
checkpoint_path = './pretrained_weights/bert_chinese_wwm/pytorch_model.bin'
dict_path = './pretrained_weights/bert_chinese_wwm/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class TrainDataset(ListDataset):
    @staticmethod
    def load_data(file_paths):
        result = []
        for file_path in file_paths:
            result.extend(load_poems_from_csv(file_path, mode='train'))
        return result
    def __getitem__(self, index):
        item = self.data[index]
        token_ids, segment_ids = tokenizer.encode(item['input'], item['output'], maxlen=maxlen)
        return {'token_ids': token_ids, 'segment_ids': segment_ids}

    def __len__(self):
        return len(self.data)
    
class TestDataset(ListDataset):
    @staticmethod
    def load_data(file_paths):
        result = []
        for file_path in file_paths:
            result.extend(load_poems_from_csv(file_path, mode='val'))
        return result
    
    def __getitem__(self, index):
        item = self.data[index]
        token_ids, segment_ids = tokenizer.encode(item['input'], item['output'], maxlen=maxlen)
        return {'token_ids': token_ids, 'segment_ids': segment_ids}

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    for item in batch:
        batch_token_ids.append(item['token_ids'])
        batch_segment_ids.append(item['segment_ids'])
    
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long) # device 在 fit 中处理
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long) 
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(
    TrainDataset(glob.glob(os.path.join(DATA_ROOT, '*.csv'))), 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=max(1, os.cpu_count() // 2),
    collate_fn=collate_fn,
    pin_memory=True if device == 'cuda' else False,
    drop_last=True
) 

val_dataloader = DataLoader(
    TestDataset(glob.glob(os.path.join(DATA_ROOT, '*.csv'))), 
    batch_size=batch_size // 4, 
    shuffle=False, 
    num_workers=max(1, os.cpu_count() // 2),
    collate_fn=collate_fn,
    pin_memory=True if device == 'cuda' else False,
    drop_last=False
)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    add_trainer=True
).to(device)

try:
    sample_batch_input, _ = next(iter(train_dataloader))
    # summary 需要输入在正确的设备上
    sample_batch_input = [t.to(device) for t in sample_batch_input]
    summary(model, input_data=sample_batch_input)
except Exception as e:
    print(f"Could not generate model summary: {e}")

grad_accumulation_steps = GRAD_ACCUMULATION_STEPS # 梯度累积步数
loss = CausalLMLoss(offset=True, logits_index=1, ignore_index=tokenizer.token_to_id('[PAD]')) 
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2) 
num_training_steps = epochs * len(train_dataloader) // grad_accumulation_steps
num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )
scheduler = LambdaLR(optimizer, lr_lambda)

model.compile(
    loss=loss, 
    optimizer=optimizer,
    scheduler=scheduler if USE_SCHEDULER else None,
    clip_grad_norm=1.0,
    # mixed_precision=True,
    grad_accumulation_steps=grad_accumulation_steps
)

class AutoPoemForTrain(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids_source, segment_ids_source = inputs  # inputs: [CLS] text_a [SEP]

        true_generated_ids = output_ids[:, 1:] 

        token_ids_combined = torch.cat([token_ids_source, true_generated_ids], 1)
        
        segment_ids_generated = torch.ones_like(true_generated_ids, device=segment_ids_source.device, dtype=segment_ids_source.dtype)
        segment_ids_combined = torch.cat([segment_ids_source, segment_ids_generated], 1)
        
        _, y_pred = model.predict([token_ids_combined, segment_ids_combined])
        
        return y_pred[:, -1, :]

    def generate(self, text, top_k=1, top_p=0.95): 
        max_c_len = maxlen - self.max_new_tokens
        # tokenizer.encode(text) -> [CLS] text [SEP]
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len) 
        token_ids_tensor = torch.tensor([token_ids], device=self.device)
        segment_ids_tensor = torch.tensor([segment_ids], device=self.device)
        
        search_results = self.beam_search(inputs=[token_ids_tensor, segment_ids_tensor], top_k=top_k, top_p=top_p)
        
        if isinstance(search_results, tuple) and len(search_results) == 2:
            output_ids = search_results[0][0] 
        else:
            output_ids = search_results[0] 

        return tokenizer.decode(output_ids.cpu().numpy())


auto_poem = AutoPoemForTrain(bos_token_id=tokenizer._token_start_id, eos_token_id=tokenizer._token_end_id, max_new_tokens=84, device=device)

test_seq = ["月&&五言绝句", "听琴&&七言绝句", "破虏歌&&五言律诗", "送君&&七言律诗"]
t = time.localtime()
log_path = '/root/tf-logs/' + str(t.tm_year) + '_' + str(t.tm_mon) + '_' + str(t.tm_mday) + \
    '_' + str(t.tm_hour) + '_' + str(t.tm_min) + '_' + str(t.tm_sec)
os.makedirs(log_path, exist_ok=True)
tb_writer = SummaryWriter(log_path)

def just_show(seq: list[str] = test_seq):
    for s in seq:
        print(u'输入: %s' % s)
        try:
            output = auto_poem.generate(s, top_k=TOP_K, top_p=0.95) # 传入 top_p
            print(u'输出: %s\n' % output)
        except Exception as e:
            print(f"Error during generation for '{s}': {e}")
            import traceback
            traceback.print_exc()

class Evaluator(Callback):
    def __init__(self):
        self.lowest_val_loss = 1e10

    def on_train_begin(self, logs = None):
        try:
            self.lowest_val_loss = self.calculate_validation_loss()
        except Exception as e:
            print(f"Error during initial validation loss calculation: {e}")
            self.lowest_val_loss = 1e10
        print(f'Initial validation loss: {self.lowest_val_loss:.6f}')

    def calculate_validation_loss(self):
        tr_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch = model._move_to_model_device(batch)
                val_X, val_y = batch
                _, y_pred = model.predict(val_X)
                loss = model.criterion(y_pred, val_y)
                tr_loss += loss.item()
        model.train()
        return tr_loss / len(val_dataloader)

    def on_epoch_end(self, steps, epoch, logs=None):
        if logs is None:
            logs = {}
        print(f'Epoch {epoch + 1} - Evaluating on validation set...')

        val_loss = self.calculate_validation_loss()
        if tb_writer is not None:
            tb_writer.add_scalar('val_loss', val_loss, epoch)
            tb_writer.add_scalar('train_loss', logs.get('loss', 0), epoch)
            tb_writer.flush()

        print(f'Epoch {epoch + 1} - Validation loss: {val_loss:.6f}')

        if val_loss < self.lowest_val_loss:

            model.save_weights('./best_model.pt')
            print(f'Validation loss decreased ({self.lowest_val_loss:.6f} --> {val_loss:.6f}).  Saving model to ./best_model.pt')
            self.lowest_val_loss = val_loss
        else:
            print(f'Validation loss did not improve ({self.lowest_val_loss:.6f} --> {val_loss:.6f}).  Not saving model.')

        just_show()

if __name__ == '__main__':
    try:
        model.load_weights('./best_model.pt')
    except:
        print('No pretrained model found, training from scratch or pretrained BERT.')

    just_show()
    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    if os.path.exists('./best_model.pt'):
        try:
            model.load_weights('./best_model.pt')
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Could not load weights in 'else' block: {e}")
    else:
        print("best_model.pt not found in 'else' block. Model will use initial weights.")