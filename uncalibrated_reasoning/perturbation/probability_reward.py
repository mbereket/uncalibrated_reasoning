import numpy as np

def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, task_name=None) -> list[float]:
    scores = []
    for solution_str, ground_truth in zip(solution_strs, ground_truths):
        scores.append(compute_score_helper(solution_str, ground_truth))
    return scores
        
def compute_score_helper(solution_str, ground_truth):
    pred = extract_between(solution_str, "<answer>", "</answer>")
    try:
        pred = float(pred)
        pred = int(pred) # round down
        if pred <= 0 or pred >= 100:
            pred = None
    except:
        pred = None

    probs = np.linspace(0.01, 0.99, 99)
    if pred is None:
        return np.log(probs[0])
    else:
        ground_truth = float(ground_truth)

        pred_idx = pred - 1
        pred_prob = probs[pred_idx]
        reward = ground_truth * np.log(pred_prob) + (1 - ground_truth) * np.log(1 - pred_prob)
        return reward


def extract_between(text, start, end):
    pieces = text.split(start)
    if len(pieces) < 2:
        return None
    else:
        return pieces[1].split(end)[0]
