import os
import warnings
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Model names and save directory
generation_model_name = "gpt2"  # Replace with your generation model of choice
entailment_model_name = "microsoft/deberta-large-mnli"
save_dir = "saved_models"

os.makedirs(save_dir, exist_ok=True)

gen_model_path = os.path.join(save_dir, generation_model_name.replace('/', '_'))
entailment_model_path = os.path.join(save_dir, entailment_model_name.replace('/', '_'))

# Load or download generation model and tokenizer
gen_pipe = pipeline(
    "text-generation",
    model=generation_model_name,
    tokenizer=generation_model_name,
    device=device,
)
gen_pipe.model.save_pretrained(gen_model_path)
gen_pipe.tokenizer.save_pretrained(gen_model_path)

# Load entailment model and tokenizer explicitly with from_pt=True for PyTorch weights
# Load model and tokenizer normally (PyTorch weights, no from_pt argument)
model = AutoModelForSequenceClassification.from_pretrained(entailment_model_name)
tokenizer = AutoTokenizer.from_pretrained(entailment_model_name)

# Create pipeline specifying PyTorch framework explicitly
entailment_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    framework="pt"
)


# Functions following the semantic entropy hallucination detection pipeline

def generate_samples(prompt, num_samples=10):
    outputs = gen_pipe(prompt, max_length=100, num_return_sequences=num_samples, do_sample=True)
    return [out['generated_text'] for out in outputs]

def check_entailment(sent1, sent2):
    result = entailment_pipe([{"text": sent1, "text_pair": sent2}])[0]
    return result['label'] == "ENTAILMENT" and result['score'] > 0.9

def cluster_by_bidirectional_entailment(outputs):
    clusters = []
    for out in outputs:
        placed = False
        for cluster in clusters:
            if check_entailment(out, cluster[0]) and check_entailment(cluster[0], out):
                cluster.append(out)
                placed = True
                break
        if not placed:
            clusters.append([out])
    return clusters

def semantic_entropy(clusters, base=2):
    total = sum(len(cluster) for cluster in clusters)
    proportions = np.array([len(cluster) / total for cluster in clusters if len(cluster) > 0])
    entropy = -np.sum(proportions * np.log(proportions) / np.log(base))
    return entropy

def detect_hallucination(prompt, entropy_threshold=0.8, num_samples=10):
    outputs = generate_samples(prompt, num_samples)
    clusters = cluster_by_bidirectional_entailment(outputs)
    ent = semantic_entropy(clusters, base=2)
    hallucinated = ent > entropy_threshold
    return hallucinated, ent, clusters

# Example usage
prompt = "Explain the theory of relativity."
hallucinated, entropy_value, clusters = detect_hallucination(prompt)

print(f"Hallucination Detected: {hallucinated}")
print(f"Semantic Entropy (bits): {entropy_value:.3f}")
print(f"Number of Clusters: {len(clusters)}")
