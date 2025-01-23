import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, phoBert_model_name='vinai/phobert-base', dropout=0.2, num_classes=5):
        super(BertClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained(phoBert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # Get the pooled output (equivalent to [CLS] token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and fully connected layer
        x = self.dropout(pooled_output)
        x = self.fc(x)

        return x