from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging
import torch
logging.set_verbosity_error()  # Suppress warnings
model_name='microsoft/deberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels = 3,from_tf=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def isEntailment(premise, hypothesis,threshold=0.7):

    inputs = tokenizer(premise, hypothesis, return_tensors='pt',padding=True,truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = logits.softmax(dim=1)
    return probs
    
def BidirectionalEntailment(premise, hypothesis, threshold=0.7):
    return isEntailment(premise, hypothesis, threshold) and isEntailment(hypothesis, premise, threshold)
def cluster_by_entailment(responses, threshold=0.7):
    clusters = []

    for resp in responses:
        matched = False

        # Compare with each existing cluster's representative
        for cluster in clusters:
            cluster_rep = cluster[0]  # Use first element as the cluster representative
            if BidirectionalEntailment(resp, cluster_rep, threshold):
                cluster.append(resp)    # Add to existing cluster
                matched = True
                break  # No need to check other clusters
        
        # If no matching cluster found, start a new one
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
            
            
    # print(BidirectionalEntailment("The cat is on the roof.", "The cat is sitting on the roof."))
    