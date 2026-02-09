print(">>> USING rouge_l.py FROM:", __file__)

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scores = []
    for p, r in zip(preds, refs):
        score = scorer.score(r, p)["rougeL"].fmeasure
        scores.append(score)

    return sum(scores) / len(scores)