import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import zipfile
import requests
from io import BytesIO
from zipfile import ZipFile
from collections import Counter
from tqdm import tqdm
import traceback
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOT = "./visual_genome_data" # directory where the JSON dumps will be placed
IMG_DIR = "./vg_images" # directory where images downloaded for experiments are stored

# Files contains all fused embeddings for train, validation and test sets
TRAIN_EMB = "train_seq_embeddings_vg.pt" 
VAL_EMB = "val_seq_embeddings_vg.pt"
TEST_EMB = "test_seq_embeddings_vg.pt"

VISUAL_GENOME_BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset" 
QUESTION_ANS_ZIP = "question_answers.json.zip"
IMAGE_DATA_ZIP = "image_data.json.zip"

SEED = 42 #for reproducibility
BATCH_EMB_BUILD = 24  # building embeddings batch (lower to avoid OOM)
TRAIN_BATCH = 8
VAL_BATCH = 16
EPOCHS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dataset class (Embeddings)
class EmbeddingDataset(Dataset):
	def __init__(self, path):
		dataset_metadata_dict = torch.load(path)
		self.text = dataset_metadata_dict["text"]
		self.img  = dataset_metadata_dict["img"]
		self.text_mask = dataset_metadata_dict.get("text_mask", torch.ones(self.text.shape[0], self.text.shape[1], dtype=torch.long))
		self.img_mask = dataset_metadata_dict.get("img_mask", torch.ones(self.img.shape[0], self.img.shape[1], dtype=torch.long))
		self.labels = dataset_metadata_dict["labels"]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return {
			"text_embedding": self.text[idx],   # (Nt, D_txt)
			"image_embedding": self.img[idx],   # (Ni, D_img)
			"text_mask": self.text_mask[idx],
			"image_mask": self.img_mask[idx],
			"label": self.labels[idx]
		}

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


# Utility function to download and unzip the JSON
def download_and_unzip_vg_jsons(target_dir=ROOT, timeout=60):
	os.makedirs(target_dir, exist_ok=True) # if root dir does't exist, create directory

	qa_zip_url = f"{VISUAL_GENOME_BASE}/{QUESTION_ANS_ZIP}" #url to the question-answer pairs (zip file)
	img_meta_zip_url = f"{VISUAL_GENOME_BASE}/{IMAGE_DATA_ZIP}" #url to the image metadata (zip file)
	qa_zip_path = os.path.join(target_dir, QUESTION_ANS_ZIP) #
	img_zip_path = os.path.join(target_dir, IMAGE_DATA_ZIP)

	# Helper function to download a file from a URL
	def fetch(url, out_path):
		print("Downloading ZIP file from: ", url)
		fetch_request = requests.get(url, timeout=timeout, stream=True) # send a GET request to download the file in streaming mode

		if fetch_request.status_code != 200:
			raise RuntimeError(f"Failed to download ZIP file at {url} with status {fetch_request.status_code}")

		with open(out_path, "wb") as f:
			for chunk in fetch_request.iter_content(chunk_size=8192): # write the file in chunks (to avoid loading everything into memory)
				if chunk: 
					f.write(chunk) # only write on non-empty chunks
		print("Saved ZIP file in path: ", out_path)

	if not os.path.exists(qa_zip_path):
		fetch(qa_zip_url, qa_zip_path)
	else:
		print("Path to the QA pairs already exists in ", qa_zip_path, "\n Using QA pairs from path")

	if not os.path.exists(img_zip_path):
		fetch(img_meta_zip_url, img_zip_path)
	else:
		print("Path to the images metadata already exists in ", img_zip_path, "\n Using images metadata from path")

	# Loop over both downloaded ZIP files to extract their contents
	for z in (qa_zip_path, img_zip_path):
		print("Unzipping ZIP file: ", z)
		with ZipFile(z, 'r') as zip_ref:
			zip_ref.extractall(target_dir)

	print("Download & unzip done, JSON files saved in ", target_dir)

# Function to load Visual Genome QA from local JSON into a flat pandas dataframe
def load_vg_qa_local(json_path, limit=2000):
	if not os.path.exists(json_path):
		raise FileNotFoundError(f"File not found: {json_path}")

	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	records = []
	for item in data:
		image_id = item.get("id", None)
		for qa_pair in item.get("qas", []):
			records.append({
				"image_id": image_id,
				"question": str(qa_pair["question"]).strip(),
				"answer": str(qa_pair["answer"]).strip().lower()
			})
			if limit and len(records) >= limit: break
		if limit and len(records) >= limit: break

	df = pd.DataFrame(records)
	print(f"Loaded {len(df)} QA pairs from Visual Genome")
	return df

# Load image metadata mapping
def load_vg_image_data_from_local(local_dir=ROOT):
	local_img_meta = None
	p = os.path.join(local_dir, "image_data.json")
 
	if os.path.exists(p):
		local_img_meta = p

	if local_img_meta is None:
		print("No image_data.json found in", local_dir, ", returing empty metadata")
		return {}

	with open(local_img_meta, "r", encoding="utf-8") as f:
		data = json.load(f)
  
	mapping = {}
	for rec in data:
		image_id = int(rec.get("image_id") or rec.get("id"))
		mapping[image_id] = {"url": rec.get("url"), "width": rec.get("width"), "height": rec.get("height")}
	
	print("Loaded image metadata for", len(mapping), "images")
	return mapping

# Pre-download images for a DataFrame (subset)
def predownload_images_for_df(df, image_meta_map, out_dir=IMG_DIR, max_images=1000):
	os.makedirs(out_dir, exist_ok=True)
	image_ids = list(dict.fromkeys(df["image_id"].tolist()))
	cnt = 0

	for img_id in tqdm(image_ids, desc="predownloading images"):
		if cnt >= max_images:
			break

		meta = image_meta_map.get(int(img_id))
  
		if not meta:
			continue

		url = meta.get("url")
		if not url:
			continue

		out_path = os.path.join(out_dir, f"{int(img_id)}.jpg")
		if os.path.exists(out_path):
			cnt += 1
			continue

		try:
			r = requests.get(url, timeout=10)
   
			if r.status_code == 200:
				with open(out_path, "wb") as f:
					f.write(r.content)
				cnt += 1
		except Exception:
			continue
	print("Downloaded", cnt, "images to", out_dir)

# ---------------------------
# Build sequence-level embeddings (BERT tokens and ViT patch tokens)
# Tolerant: uses local IMG_DIR images first; if missing, tries image_meta_map URL (download)
# ---------------------------
def build_and_save_embeddings(df, tokenizer, img_preprocessor, text_encoder, img_encoder, image_meta_map, out_path, 
							  local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD, max_text_len=64):
	"""
	Builds and saves sequence-level embeddings with a fixed text token length.

	- text_seq: (N, max_text_len, D_txt)
	- text_mask: (N, max_text_len)
	- img_seq:  (N, Ni, D_img)  (Ni fixed for ViT)
	- img_mask: (N, Ni)
	"""
	text_encoder.eval()
	img_encoder.eval()
	text_encoder.to(device)
	img_encoder.to(device)
	os.makedirs(local_image_dir, exist_ok=True)

	all_text_seq = []
	all_img_seq = []
	all_text_masks = []
	all_img_masks = []
	all_labels = []
	dropped = 0

	for i in tqdm(range(0, len(df), batch_size), desc=f"Building {out_path}"):
		batch = df.iloc[i:i+batch_size]
		qs = batch["question"].tolist()
		image_ids = batch["image_id"].tolist()

		imgs = []
		valid_indices = []
		for idx, img_id in enumerate(image_ids):
			local_path = os.path.join(local_image_dir, f"{int(img_id)}.jpg")
			img_path = None
			if os.path.exists(local_path):
				img_path = local_path
			else:
				meta = image_meta_map.get(int(img_id), {})
				url = meta.get("url")
				if url:
					try:
						# try to download on demand
						r = requests.get(url, timeout=10)
						if r.status_code == 200:
							with open(local_path, "wb") as f:
								f.write(r.content)
							img_path = local_path
					except Exception:
						img_path = None
			if not img_path or not os.path.exists(img_path):
				dropped += 1
				continue
			try:
				img = Image.open(img_path).convert("RGB")
				imgs.append(img)
				valid_indices.append(idx)
			except Exception:
				dropped += 1
				continue

		if len(imgs) == 0:
			continue

		batch_qs = [qs[k] for k in valid_indices]
		batch_labels = batch["label"].values[valid_indices].tolist()

		with torch.no_grad():
			# Tokenize with fixed max length so all batches have same Nt
			tokenized_qs = tokenizer(batch_qs,
				padding="max_length",
				truncation=True,
				max_length=max_text_len,
				return_tensors="pt")
			tokenized_qs = {k: v.to(device) for k, v in tokenized_qs.items()}
			text_outputs = text_encoder(**tokenized_qs)
			# text_outputs.last_hidden_state -> (B, max_text_len, D_txt)
			text_seq = text_outputs.last_hidden_state.detach().cpu()
			text_mask = tokenized_qs["attention_mask"].detach().cpu()  # (B, max_text_len)

			# Images -> ViT patch tokens (Ni fixed)
			img_inputs = img_preprocessor(images=imgs, return_tensors="pt")
			img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
			img_outputs = img_encoder(**img_inputs)
			img_seq = img_outputs.last_hidden_state.detach().cpu()  # (B, Ni, D_img)
			img_mask = torch.ones(img_seq.shape[0], img_seq.shape[1], dtype=torch.long)

		all_text_seq.append(text_seq)
		all_img_seq.append(img_seq)
		all_text_masks.append(text_mask)
		all_img_masks.append(img_mask)
		all_labels.append(torch.tensor(batch_labels, dtype=torch.long))

	if len(all_labels) == 0:
		raise RuntimeError("No embeddings were created (no valid images found). Check local images or image_meta_map URLs.")

	# Now concatenation will succeed because every text_seq has shape (B, max_text_len, D)
	text_seq = torch.cat(all_text_seq, dim=0)
	img_seq = torch.cat(all_img_seq, dim=0)
	text_masks = torch.cat(all_text_masks, dim=0)
	img_masks = torch.cat(all_img_masks, dim=0)
	labels = torch.cat(all_labels, dim=0)

	torch.save({"text": text_seq, "img": img_seq, "text_mask": text_masks, "img_mask": img_masks, "labels": labels}, out_path)
	print("Saved embeddings:", out_path, "Dropped samples during build:", dropped)

def build_vg_dataframe(local_dir=ROOT, limit=None):
	qa_json = os.path.join(local_dir, "question_answers.json")
 
	if not os.path.exists(qa_json):
		raise FileNotFoundError(f"Please put 'question_answers.json' into {local_dir} or run download_and_unzip_vg_jsons().")
  
	print("Loading QA from", qa_json)
	df = load_vg_qa_local(qa_json)
	df["question"] = df["question"].astype(str)
	df["answer"] = df["answer"].astype(str).str.lower().str.strip()
	df = df[df["question"].str.len() > 0]

	if limit:
		df = df.sample(n=min(limit, len(df)), random_state=SEED).reset_index(drop=True)
	else:
		df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
  
	n = len(df)
	n_train = int(0.8 * n)
	n_val = int(0.1 * n)
	train_df = df.iloc[:n_train].reset_index(drop=True)
	val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
	test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
	print("Split sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
 
	return train_df, val_df, test_df

def map_label(ans): 
	return ans_to_labels.get(ans, -1)

def evaluate(model, val_dataloader, criterion, device=DEVICE):
	model.eval()
	total_val_loss = 0.0
	predictions = []
	true_vals = []
	conf = []

	with torch.no_grad():
		for batch in val_dataloader:
			batch = {k: v.to(device) for k, v in batch.items()}
			inputs = {
				'image_embedding': batch['image_embedding'],
				'text_embedding': batch['text_embedding'],
				'image_mask': batch.get('image_mask', None),
				'text_mask': batch.get('text_mask', None)
			}

			outputs = model(**inputs)
			labels = batch['label']
			loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
			total_val_loss += loss.item()
			probs = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
			outputs = outputs.argmax(-1)
			predictions.append(outputs.detach().cpu().numpy())
			true_vals.append(labels.cpu().numpy())
			conf.append(probs)
   
	loss_val_avg = total_val_loss / len(val_dataloader)
	predictions = np.concatenate(predictions, axis=0)
	true_vals = np.concatenate(true_vals, axis=0)
	conf = np.concatenate(conf, axis=0)
 
	return loss_val_avg, predictions, true_vals, conf

def train_loop(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=EPOCHS, device=DEVICE):
	best_val_acc = 0.0

	for epoch in range(1, epochs+1):
		model.train()
		total_train_loss = 0.0
		train_predictions = []
		train_true_vals = []

		for batch in tqdm(train_dataloader, desc=f"Train epoch {epoch}"):
			batch = {k: v.to(device) for k, v in batch.items()}
			inputs = {
				'image_embedding': batch['image_embedding'],
				'text_embedding': batch['text_embedding'],
				'image_mask': batch.get('image_mask', None),
				'text_mask': batch.get('text_mask', None)
			}

			labels = batch['label']
			outputs = model(**inputs)
			loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()
			total_train_loss += loss.item()
			train_predictions.append(outputs.argmax(-1).detach().cpu().numpy())
			train_true_vals.append(labels.cpu().numpy())
   
		train_preds = np.concatenate(train_predictions, axis=0)
		train_trues = np.concatenate(train_true_vals, axis=0)
		train_acc = accuracy_score(train_preds, train_trues)
		val_loss, val_preds, val_trues, _ = evaluate(model, val_dataloader, criterion, device=device)
		val_acc = accuracy_score(val_preds, val_trues)
		print(f"Epoch {epoch}: train_loss={total_train_loss/len(train_dataloader):.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
		
    if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), "best_model_vg_fusion.pth")
			print("Saved best model with val_acc:", best_val_acc)
   
	return best_val_acc

def vqa_pipeline(limit_examples=2000, top_k=1000, max_images_to_download=1000, download_jsons=True): #limiting to only 2000 examples for experimenting (default), None for full dataset (appox 108000 examples)
	try:
    #--------------------------------------------------Data Preparation------------------------------------------------------------------
		# Downloading JSONS for extracting QA pairs and images metadata (set to False, if already downloaded locally)
		if download_jsons:
			print("Downloading Visual Genome JSONs")
			download_and_unzip_vg_jsons(ROOT)
		
		# Building dataframes (based on limit size)
		train_df, val_df, test_df = build_vg_dataframe(local_dir=ROOT, limit=limit_examples)
		
		# Load image meta mapping
		image_meta_map = load_vg_image_data_from_local(local_dir=ROOT)
		
		# Build top-K answer vocabulary (for classifying answers)
		all_answers = pd.concat([train_df["answer"], val_df["answer"], test_df["answer"]])
		ans_counts = Counter(all_answers.tolist())
		most_common = [a for a, _ in ans_counts.most_common(top_k)]
		ans_to_labels = {a: i for i, a in enumerate(most_common)}
		label_to_ans = {i: a for a, i in ans_to_labels.items()}
		print("Top-K answer vocab size:", len(ans_to_labels))
		
    train_df["label"] = train_df["answer"].apply(map_label)
		val_df["label"] = val_df["answer"].apply(map_label)
		test_df["label"] = test_df["answer"].apply(map_label)
		# drop -1s
		train_df = train_df[train_df["label"] >= 0].reset_index(drop=True)
		val_df = val_df[val_df["label"] >= 0].reset_index(drop=True)
		test_df = test_df[test_df["label"] >= 0].reset_index(drop=True)
		print("After mapping to top-K -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
		
		# Pre-download images subset for speed
		print("Pre-downloading up to", max_images_to_download, "images for training/val/test")
		combined_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
		predownload_images_for_df(combined_df, image_meta_map, out_dir=IMG_DIR, max_images=max_images_to_download)
  

    #--------------------------------------------------Feature Engineering------------------------------------------------------------------
		# Load backbones (frozen)
		tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # loads BERT's tokenizer to tokenize the question strings into many lowercased tokens and convert them into token IDs that BERT understands
		text_encoder_bert = AutoModel.from_pretrained("bert-base-uncased") # loads a pre-trained BERT model used for converting the token IDs and produces dense vector embeddings for each token
		for p in text_encoder_bert.parameters(): # leaving the parameters (tensors) of the BERT model unchanged (frozen) (to avoid re-training it when the entire fusion network learns)
			p.requires_grad = False # No need to compute gradients for each tensor during backpropagation

		img_preprocessor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k") # loads ViT’s feature extractor that normalizes & resizes images to 224×224 patches
		img_encoder_vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k") #loads the ViT model
		for p in img_encoder_vit.parameters(): # leaving the parameters (tensors) of the ViT model unchanged (frozen) (to avoid re-training it when the entire fusion network learns)
			p.requires_grad = False ## No need to compute gradients for each tensor during backpropagation
		
		# Build embeddings files (sequence-level)
		if not os.path.exists(TRAIN_EMB):
			build_and_save_embeddings(train_df, tokenizer, img_preprocessor, text_encoder_bert, img_encoder_vit, image_meta_map, TRAIN_EMB, local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD)
		if not os.path.exists(VAL_EMB):
			build_and_save_embeddings(val_df, tokenizer, img_preprocessor, text_encoder_bert, img_encoder_vit, image_meta_map, VAL_EMB, local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD)
		if not os.path.exists(TEST_EMB):
			build_and_save_embeddings(test_df, tokenizer, img_preprocessor, text_encoder_bert, img_encoder_vit, image_meta_map, TEST_EMB, local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD)
		
    #-----------------------------------------------------Model Initialization-----------------------------------------------------------------
		# Datasets / Dataloaders
		train_dataset = EmbeddingDataset(TRAIN_EMB)
		val_dataset = EmbeddingDataset(VAL_EMB)
		test_dataset = EmbeddingDataset(TEST_EMB)
		train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=2, pin_memory=True)
		val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)
		test_dataloader = DataLoader(test_dataset, batch_size=VAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)
		
		# Model and Training setup
		num_answers_total = len(ans_to_labels)
		model = CrossAttentionFusionNetwork(d_img=768, d_txt=768, d=512, n_heads=8, num_answers=num_answers_total, num_cross_layers=2, dropout=0.3)
		model.to(DEVICE)
		criterion = nn.CrossEntropyLoss()
		train_steps = len(train_dataloader) * EPOCHS
		warm_steps = max(1, int(train_steps * 0.1))
		optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5, eps=1e-8)
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=train_steps)
		
    #-----------------------------------------------------Model Training + Testing-----------------------------------------------------------------
		# Train
		best_val = train_loop(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=EPOCHS, device=DEVICE)
		print("Best val acc:", best_val)
		
		# Test
		model.load_state_dict(torch.load("best_model_vg_fusion.pth", map_location=DEVICE))
		test_loss, preds, truths, conf = evaluate(model, test_dataloader, criterion, device=DEVICE)
		test_acc, test_prec, test_recall, test_f1 = accuracy_score(preds, truths), precision_score(preds, truths), recall_score(preds, truths), f1_score(preds, truths)
		print("Test loss:", test_loss)
		print("Test performance:")
		print(f"Accuracy: {test_acc:.3f}, Precision: {test_prec:.3f}, Recall: {test_recall:.3f}, F1 Score: {test_f1:.3f}")

	except Exception as e:
		print("Error in VQA pipeline due to error:", e)
		print(traceback.format_exc())

if __name__ == "__main__":
	# TODO: adjust params to scale up training
	vqa_pipeline(limit_examples=2000, top_k=1000, max_images_to_download=1000, download_jsons=True) #small params for quick experiments, can you give me a good way to modularize this (DON'T change the code)
