# Interview Practice Set B — Research-Oriented (PhD-Style)

## 1. Motivation & Research Thinking
- What types of ML research interest you the most?
- How do you think about forming hypotheses in ML evaluation?

## 2. Deep Dive: Tracking & CV
- Compare these: MOTA, MOTP, IDF1.
- Why is IDF1 considered more reliable than MOTA in some cases?
- How would you detect identity switches in tracking?
- How do occlusions impact tracking performance?

## 3. Benchmarking Design
- Suppose you are creating a new benchmark for drone tracking — what are the key considerations?
- How do you ensure fairness across models that expect different input sizes?
- How do you detect annotation noise or inconsistencies?
- What is data leakage? Give a real CV example.

## 4. PyTorch / Implementation
- How would you compute IoU between predicted and ground truth boxes?
- Describe how you structure a Dataset class for video-based data.
- What are common sources of nondeterminism in PyTorch?
- What does torch.backends.cudnn.benchmark = True/False do?

## 5. Debugging / Experiments
- A model suddenly loses 20% mAP after small refactoring. How do you debug?
- If your experiment results seem too good to be true, how would you verify them?
- How would you test robustness to frame drops or motion blur?

## 6. Behavioural
- Tell me about a time when you created a system or pipeline others relied on.
- Describe a challenging ML debugging moment.
- What skills do you hope to gain in this co-op?
