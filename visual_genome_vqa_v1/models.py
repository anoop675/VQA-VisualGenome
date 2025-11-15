import torch
from torch import nn

# Cross Attention Fusion Model
class CrossAttentionBlock(nn.Module):
	def __init__(self, d_model, nhead=8, dim_ff=2048, dropout=0.1):
		super().__init__()
		self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
		self.norm1 = nn.LayerNorm(d_model)
		self.ff = nn.Sequential(
			nn.Linear(d_model, dim_ff),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(dim_ff, d_model)
		)
		self.norm2 = nn.LayerNorm(d_model)

	def forward(self, q, kv, kv_key_padding_mask=None):
		attn_out, _ = self.mha(query=q, key=kv, value=kv, key_padding_mask=kv_key_padding_mask)
		q = self.norm1(q + attn_out)
		q2 = self.ff(q)
		return self.norm2(q + q2)

class CrossAttentionFusionNetwork(nn.Module):
	def __init__(self, d_img=768, d_txt=768, d=512, n_heads=8, num_answers=1000, num_cross_layers=2, dropout=0.3):
		super().__init__()
		self.P_img = nn.Linear(d_img, d)
		self.P_txt = nn.Linear(d_txt, d)
		self.cross_blocks = nn.ModuleList([CrossAttentionBlock(d_model=d, nhead=n_heads) for _ in range(num_cross_layers)])
		self.fc1 = nn.Linear(d, 256)
		self.bn1 = nn.BatchNorm1d(256)
		self.classifier = nn.Linear(256, num_answers)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

	def forward(self, image_embedding, text_embedding, image_mask=None, text_mask=None):
		I = self.P_img(image_embedding)  # [B, Ni, d]
		T = self.P_txt(text_embedding)   # [B, Nt, d]
		kv_mask = None
		if image_mask is not None:
			kv_mask = (image_mask == 0)
		Tq = T
		for blk in self.cross_blocks:
			Tq = blk(Tq, I, kv_key_padding_mask=kv_mask)
		pooled = Tq[:, 0, :]  # assumes CLS at 0
		x = self.relu(self.fc1(pooled))
		x = self.bn1(x)
		x = self.dropout(x)
		logits = self.classifier(x)
		return logits