#!/usr/bin/env python
# -*- coding: utf-8 -*-

# classifier_odsbahia-ptbr.py
# --- lê phrases.xlsx, classifica com o modelo ODSBahia e gera phrases_classifiedODS.xlsx ---

import argparse
import os
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "odsbahia/odsbahia-ptbr"
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_CODE_COL = "code"
DEFAULT_TEXT_COL = "phrase"
DEFAULT_INPUT = "phrases.xlsx"
DEFAULT_OUTPUT = "phrases_classifiedODS.xlsx"
THRESHOLD = 0.7


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_batch(texts: List[str], tokenizer, model, device, max_len: int):
    if not texts:
        return []

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        # scores contínuos por rótulo
        probs = torch.sigmoid(logits)

    results = []
    for row in probs.cpu():
        results.append([float(v.item()) for v in row])
    return results


def run(input_path, code_column, text_column, output_path, batch_size, max_len):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    print(f"Lendo: {input_path}")
    df = pd.read_excel(input_path)

    for col in (code_column, text_column):
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não existe. Colunas disponíveis: {list(df.columns)}"
            )

    # ==============================
    # GERAR code SEQUENCIAL QUANDO VIER EM BRANCO
    # ==============================
    raw_codes = df[code_column].astype(str).tolist()
    codes = []
    counter = 1
    for c in raw_codes:
        c_strip = c.strip()
        # trata vazio e NaN (convertido para string)
        if c_strip == "" or c_strip.lower() == "nan":
            codes.append(str(counter))
            counter += 1
        else:
            codes.append(c_strip)

    texts = df[text_column].fillna("").astype(str).tolist()

    tokenizer, model, device = load_model()
    id2label = model.config.id2label
    num_labels = model.config.num_labels

    # tenta primeiro índice int; se der KeyError, cai pra string
    try:
        label_order = [id2label[i] for i in range(num_labels)]
    except KeyError:
        label_order = [id2label[str(i)] for i in range(num_labels)]

    print("Labels do modelo na ordem dos logits:")
    print(label_order)

    print(f"Classificando {len(texts)} phrases...")
    all_scores = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        all_scores.extend(
            classify_batch(batch, tokenizer, model, device, max_len)
        )
        print(f"  {end}/{len(texts)}")

    # monta dataframe com scores por ODS
    out_df = pd.DataFrame(all_scores, columns=label_order)
    out_df.insert(0, code_column, codes)

    # calcula most_similar e maximum_similarity
    most_similar = []
    maximum_similarity = []

    for scores in all_scores:
        max_val = max(scores)
        max_idx = scores.index(max_val)
        label = label_order[max_idx]
        if max_val >= THRESHOLD:
            most_similar.append(label)
        else:
            most_similar.append("None_indeterminate")
        maximum_similarity.append(round(max_val, 4))

    out_df["most_similar"] = most_similar
    out_df["maximum_similarity"] = maximum_similarity

    out_path = output_path if output_path else DEFAULT_OUTPUT
    print(f"Salvando -> {out_path}")
    out_df.to_excel(out_path, index=False)
    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--code-column", default=DEFAULT_CODE_COL)
    p.add_argument("--text-column", default=DEFAULT_TEXT_COL)
    p.add_argument("--output")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.input, a.code_column, a.text_column, a.output, a.batch_size, a.max_len)
