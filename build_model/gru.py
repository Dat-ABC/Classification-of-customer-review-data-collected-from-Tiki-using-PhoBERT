import torch.nn as nn
import torch

class GRU(nn.Module):
	def __init__(self, vocab_size, embedding_dim, gru_units=64, num_layers=1, dropout=0.2, activation=None, batch_normalization=False, bidirectional=False, output_dim=2):
		super(GRU, self).__init__()
		self.embedding=nn.Embedding(vocab_size, embedding_dim)
		self.gru=nn.GRU(embedding_dim, gru_units, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
		if bidirectional==True:
			self.dense1=nn.Linear(gru_units*2, gru_units)
		elif bidirectional==False:
			self.dense1=nn.Linear(gru_units, gru_units)
		self.bn=nn.BatchNorm1d(gru_units) if batch_normalization else nn.Identity()
		self.activation=activation
		self.dropout=nn.Dropout(dropout)
		self.dense2=nn.Linear(gru_units, output_dim)
	def forward(self, x):
		x = self.embedding(x)
		x, _ = self.gru(x)
		# x = torch.mean(x, dim=1)
		x=self.dense1(x[:, -1, :])
		x=self.bn(x)
		if self.activation is not None:
			x=self.activation(x)
		x=self.dropout(x)
		x=self.dense2(x)
		return x