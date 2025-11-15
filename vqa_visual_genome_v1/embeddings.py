import os
from PIL import Image
import torch
from tqdm import tqdm


from config import IMG_DIR, BATCH_EMB_BUILD, DEVICE

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
			tokenized_qs = tokenizer(batch_qs, padding="max_length", truncation=True, max_length=max_text_len, return_tensors="pt")
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