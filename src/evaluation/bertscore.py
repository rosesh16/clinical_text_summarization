from bert_score import score

def compute_bertscore(preds, refs):
    P, R, F1 = score(
        preds,
        refs,
        lang="en",
        verbose=False
    )
    return F1.mean().item()