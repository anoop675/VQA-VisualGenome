import os
import random
import traceback

import numpy as np
import pandas as pd
import torch

from config import SEED, TRAIN_EMB, VAL_EMB, TEST_EMB, IMG_DIR, DEVICE, TRAIN_BATCH, VAL_BATCH, EPOCHS
from data_utils import download_and_unzip_vg_jsons, load_vg_qa_local, load_vg_image_data_from_local
from embeddings import predownload_images_for_df, build_and_save_embeddings
from dataset import EmbeddingDataset
from models import CrossAttentionFusionNetwork
from train import train_loop, evaluate

from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import Counter
from config import ROOT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_vg_dataframe(local_dir=ROOT, limit=None):
    qa_json = os.path.join(local_dir, "question_answers.json")

    if not os.path.exists(qa_json):
        raise FileNotFoundError(
            f"Please put 'question_answers.json' into {local_dir} or run download_and_unzip_vg_jsons().")

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

def vqa_pipeline(limit_examples=2000, top_k=1000, max_images_to_download=1000,
                 download_jsons=True):  # limiting to only 2000 examples for experimenting (default), None for full dataset (appox 108000 examples)
    try:
        # --------------------------------------------------Data Preparation------------------------------------------------------------------
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

        # --------------------------------------------------Feature Engineering------------------------------------------------------------------
        # Load backbones (frozen)
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")  # loads BERT's tokenizer to tokenize the question strings into many lowercased tokens and convert them into token IDs that BERT understands
        text_encoder_bert = AutoModel.from_pretrained(
            "bert-base-uncased")  # loads a pre-trained BERT model used for converting the token IDs and produces dense vector embeddings for each token
        for p in text_encoder_bert.parameters():  # leaving the parameters (tensors) of the BERT model unchanged (frozen) (to avoid re-training it when the entire fusion network learns)
            p.requires_grad = False  # No need to compute gradients for each tensor during backpropagation

        img_preprocessor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k")  # loads ViT’s feature extractor that normalizes & resizes images to 224×224 patches
        img_encoder_vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")  # loads the ViT model
        for p in img_encoder_vit.parameters():  # leaving the parameters (tensors) of the ViT model unchanged (frozen) (to avoid re-training it when the entire fusion network learns)
            p.requires_grad = False  ## No need to compute gradients for each tensor during backpropagation

        # Build embeddings files (sequence-level)
        if not os.path.exists(TRAIN_EMB):
            build_and_save_embeddings(train_df, tokenizer, img_preprocessor, text_encoder_bert, img_encoder_vit, image_meta_map,
                                      TRAIN_EMB, local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD)
        if not os.path.exists(VAL_EMB):
            build_and_save_embeddings(val_df, tokenizer, img_preprocessor, text_encoder_bert, img_encoder_vit, image_meta_map,
                                      VAL_EMB, local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD)
        if not os.path.exists(TEST_EMB):
            build_and_save_embeddings(test_df, tokenizer, img_preprocessor, text_encoder_bert, img_encoder_vit, image_meta_map,
                                      TEST_EMB, local_image_dir=IMG_DIR, device=DEVICE, batch_size=BATCH_EMB_BUILD)

        # -----------------------------------------------------Model Initialization-----------------------------------------------------------------
        # Datasets / Dataloaders
        train_dataset = EmbeddingDataset(TRAIN_EMB)
        val_dataset = EmbeddingDataset(VAL_EMB)
        test_dataset = EmbeddingDataset(TEST_EMB)
        train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=2, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=VAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)

        # Model and Training setup
        num_answers_total = len(ans_to_labels)
        model = CrossAttentionFusionNetwork(d_img=768, d_txt=768, d=512, n_heads=8, num_answers=num_answers_total,
                                            num_cross_layers=2, dropout=0.3)
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        train_steps = len(train_dataloader) * EPOCHS
        warm_steps = max(1, int(train_steps * 0.1))
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=train_steps)

        # -----------------------------------------------------Model Training + Testing-----------------------------------------------------------------
        # Train
        best_val = train_loop(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=EPOCHS,
                              device=DEVICE)
        print("Best val acc:", best_val)

        # Test
        model.load_state_dict(torch.load("best_model_vg_fusion.pth", map_location=DEVICE))
        test_loss, preds, truths, conf = evaluate(model, test_dataloader, criterion, device=DEVICE)
        test_acc, test_prec, test_recall, test_f1 = accuracy_score(preds, truths), precision_score(preds, truths), recall_score(
            preds, truths), f1_score(preds, truths)
        print("Test loss:", test_loss)
        print("Test performance:")
        print(f"Accuracy: {test_acc:.3f}, Precision: {test_prec:.3f}, Recall: {test_recall:.3f}, F1 Score: {test_f1:.3f}")

    except Exception as e:
        print("Error in VQA pipeline due to error:", e)
        print(traceback.format_exc())

if __name__ == "__main__":
    vqa_pipeline(limit_examples=2000, top_k=1000, max_images_to_download=1000, download_jsons=True)