from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
import torch

logging.set_verbosity_error()  # Suppress warnings

# Use a model already fine-tuned for MNLI
model_name = "microsoft/deberta-base-mnli"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def isEntailment(premise, hypothesis, threshold=0.7):
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = logits.softmax(dim=1)
    return probs[0][2] >= threshold

def BidirectionalEntailment(premise, hypothesis, threshold=0.7):
    return isEntailment(premise, hypothesis, threshold) and isEntailment(hypothesis, premise, threshold)

def cluster_by_entailment(responses, threshold=0.7):
    clusters = []
    for resp in responses:
        matched = False
        for cluster in clusters:
            cluster_rep = cluster[0]
            if BidirectionalEntailment(resp, cluster_rep, threshold):
                cluster.append(resp)
                matched = True
                break
        if not matched:
            clusters.append([resp])
    return clusters

if __name__ == "__main__":
    responses = [
        "The cat is on the roof.",
        "The dog is in the garden.",
        "The cat is sitting on the roof.",
        "The dog is playing in the garden.",
    ]
    for resp1 in responses:
        for resp2 in responses:
            print(f"Entailment('{resp1}', '{resp2}') = {isEntailment(resp1, resp2)}")
    