import torch
import torch.nn as nn

class LSTM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, lstm_units=64, num_layers=2, dropout=0.2, activation=None, batch_normalization=False, bidirectional=False, output_dim=5):
		super(LSTM, self).__init__()
		self.embedding=nn.Embedding(vocab_size, embedding_dim)
		self.lstm=nn.LSTM(embedding_dim, lstm_units, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
		if bidirectional:
			self.dense1=nn.Linear(lstm_units*2, lstm_units)
		else:
			self.dense1=nn.Linear(lstm_units, lstm_units)
		self.bn=nn.BatchNorm1d(lstm_units) if batch_normalization else nn.Identity()
		self.activation=activation
		self.dropout=nn.Dropout(dropout)
		self.dense2=nn.Linear(lstm_units, output_dim)
	def forward(self, inputs):
		X=self.embedding(inputs)
		X,_=self.lstm(X)
		X = self.dense1(X[:, -1, :])
		X=self.bn(X)
		if self.activation is not None:
			X=self.activation(X)
		X=self.dropout(X)
		X=self.dense2(X)
		return X
