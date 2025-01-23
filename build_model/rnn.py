import torch
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, rnn_units=64, num_layers=2, dropout=0.2, activation=None, batch_normalization= False, bidirectional=False, output_dim=3):
		super(RNN, self).__init__()
		self.embedding=nn.Embedding(vocab_size, embedding_dim)
		self.rnn=nn.RNN(embedding_dim, rnn_units, num_layers, dropout=dropout, bidirectional=bidirectional)
		if bidirectional==True:
			self.dense1=nn.Linear(rnn_units*2, rnn_units)
		elif bidirectional==False:
			self.dense1=nn.Linear(rnn_units, rnn_units)
		self.bn=nn.BatchNorm1d(rnn_units) if batch_normalization else nn.Identity()
		self.activation=activation
		self.dropout=nn.Dropout(dropout)
		self.dense2=nn.Linear(rnn_units, output_dim)
	def forward(self, x):
		x=self.embedding(x)
		x, _ = self.rnn(x)
		# x=self.dense1(x)
		x = self.dense1(x[:, -1, :])
		x=self.bn(x)
		if self.activation is not None:
			x=self.activation(x)
		x=self.dropout(x)
		x=self.dense2(x)
		return x