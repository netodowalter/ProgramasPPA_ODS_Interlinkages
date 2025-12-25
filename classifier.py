import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)

categories_df = pd.read_excel("categories.xlsx")
categories_codes = categories_df['code'].tolist()
categories_description = categories_df['description'].tolist()
categories_embeddings = model.encode(categories_description, convert_to_tensor=True, device=device)

df = pd.read_excel("phrases.xlsx")

resultados = []

for phrase in df["phrase"]:
    phrase_embedding = model.encode(phrase, convert_to_tensor=True, device=device)
    similaridades = util.pytorch_cos_sim(phrase_embedding, categories_embeddings)[0]
    similaridade_dict = {categories_codes[i]: round(similaridades[i].item(), 4) for i in range(len(categories_codes))}
    max_category = max(similaridade_dict, key=similaridade_dict.get)
    max_value = similaridade_dict[max_category]
    categories_classificado = max_category if max_value >= 0.7 else "None_indeterminate"
    resultados.append({
        "Phrase": phrase,
        **similaridade_dict,
        "Most similar category": categories_classificado,
        "Maximum similarity": max_value
    })

df_resultado = pd.DataFrame(resultados)
df_resultado.to_excel("phrases_classified.xlsx", index=False)
print("Classification completed! Results saved in the same folder")