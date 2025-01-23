from tokenizers import Tokenizer
from transformers import BertTokenizer, AutoTokenizer
import torch
from gensim.utils import simple_preprocess

# tokenizer_bert = BertTokenizer.from_pretrained('../viet_bert_tokenizer', do_lower_case = False)
tokenizer_phobert = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast = False)

def token(text, max_len=35):
    text = text
    text = ' '.join(simple_preprocess(text))
    text = ' '.join(tokenizer_phobert.tokenize(text))
    encoding = tokenizer_phobert.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )

    return {
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_masks': encoding['attention_mask'].flatten()
    }