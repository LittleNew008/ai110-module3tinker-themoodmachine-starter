"""
Train the ML model on the existing dataset (dataset.py) and test both the
ML model and the rule based model on a separate hand-written test set.

The test posts below are NOT in dataset.py, so neither model has seen them.

Run:  python holdout_eval.py
"""

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer

# (post, true label)
TEST_CASES = [
    # positive
    ("I got the job!! so happy right now", "positive"),
    ("What a great day, the sun is out and I feel good", "positive"),
    ("Lowkey obsessed with this new album, it's amazing", "positive"),
    ("Can't wait for the trip, super excited 😂", "positive"),
    ("Dinner with my family was so much fun :)", "positive"),
    ("this pizza is fire no cap", "positive"),
    # negative
    ("I hate waiting in line, so boring", "negative"),
    ("Feeling sad and alone tonight :(", "negative"),
    ("My phone broke and I'm so angry", "negative"),
    ("That movie was awful, total waste of money", "negative"),
    ("I am not having a good day at all", "negative"),
    ("so stressed about finals I could cry 😭", "negative"),
    # neutral
    ("The meeting is at 3pm in room 204", "neutral"),
    ("Just took the bus to campus", "neutral"),
    ("Watching the game tonight", "neutral"),
    ("It is what it is", "neutral"),
    # mixed
    ("Happy it's the weekend but sad I have to study", "mixed"),
    ("Tired but proud, that was a hard workout", "mixed"),
    ("Excited and nervous about the interview", "mixed"),
    ("The food was good but the service was terrible", "mixed"),
    # tricky: sarcasm / slang
    ("Oh great, another flat tire. Just what I needed", "negative"),
    ("I just love when my laptop dies mid-presentation", "negative"),
    ("ngl that exam was lowkey easy", "positive"),
    ("bruh this is so mid 💀", "negative"),
]


def main() -> None:
    texts = [t for t, _ in TEST_CASES]
    true = [l for _, l in TEST_CASES]

    vectorizer = CountVectorizer()
    model = LogisticRegression(max_iter=1000)
    model.fit(vectorizer.fit_transform(SAMPLE_POSTS), TRUE_LABELS)

    analyzer = MoodAnalyzer()
    rule_preds = [analyzer.predict_label(t) for t in texts]
    ml_preds = list(model.predict(vectorizer.transform(texts)))

    print(f"Trained ML on {len(SAMPLE_POSTS)} posts; testing on {len(TEST_CASES)} new posts\n")
    for t, y, r, m in zip(texts, true, rule_preds, ml_preds):
        print(
            f"rule={'OK' if r == y else 'XX'}({r:8}) "
            f"ml={'OK' if m == y else 'XX'}({m:8}) true={y:8} | {t}"
        )

    n = len(TEST_CASES)
    print(f"\nRule based accuracy: {sum(r == y for r, y in zip(rule_preds, true)) / n:.2f}")
    print(f"ML accuracy:         {sum(m == y for m, y in zip(ml_preds, true)) / n:.2f}")

    print("\nPer-label accuracy (rule / ml):")
    for label in ["positive", "negative", "neutral", "mixed"]:
        idx = [i for i, y in enumerate(true) if y == label]
        r_ok = sum(rule_preds[i] == label for i in idx)
        m_ok = sum(ml_preds[i] == label for i in idx)
        print(f"  {label:8} {r_ok}/{len(idx)}  /  {m_ok}/{len(idx)}")

    print("\nML prediction counts:", dict(Counter(ml_preds)))


if __name__ == "__main__":
    main()
