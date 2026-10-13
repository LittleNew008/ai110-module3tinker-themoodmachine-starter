"""
5-fold cross validation comparing the rule based model and the ML model.

Every post is predicted by a model that never saw it during training, so
the scores reflect performance on unseen text (unlike ml_experiments.py,
which evaluates on the training data).

Run:  python cross_validation.py
"""

import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer

N_SPLITS = 5
SEEDS = [0, 1, 2, 3, 4]  # different shuffles, since the dataset is tiny


def cross_validate(seed: int, verbose: bool = False) -> tuple:
    """Return (rule_correct, ml_correct) over all posts for one 5-fold split."""
    analyzer = MoodAnalyzer()
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    rule_correct = ml_correct = 0

    for train_idx, test_idx in kfold.split(SAMPLE_POSTS):
        train_texts = [SAMPLE_POSTS[i] for i in train_idx]
        train_labels = [TRUE_LABELS[i] for i in train_idx]

        vectorizer = CountVectorizer()
        model = LogisticRegression(max_iter=1000)
        model.fit(vectorizer.fit_transform(train_texts), train_labels)

        for i in test_idx:
            text, true = SAMPLE_POSTS[i], TRUE_LABELS[i]
            rule_pred = analyzer.predict_label(text)
            ml_pred = model.predict(vectorizer.transform([text]))[0]
            rule_correct += rule_pred == true
            ml_correct += ml_pred == true
            if verbose:
                print(
                    f"rule={'OK' if rule_pred == true else 'XX'}({rule_pred:8}) "
                    f"ml={'OK' if ml_pred == true else 'XX'}({ml_pred:8}) "
                    f"true={true:8} | {text}"
                )

    return rule_correct, ml_correct


def main() -> None:
    warnings.filterwarnings("ignore")
    total = len(SAMPLE_POSTS)

    print(f"=== {N_SPLITS}-fold cross validation on {total} posts (seed 0) ===")
    cross_validate(SEEDS[0], verbose=True)

    print(f"\n=== Accuracy across {len(SEEDS)} different shuffles ===")
    rule_scores, ml_scores = [], []
    for seed in SEEDS:
        rule, ml = cross_validate(seed)
        rule_scores.append(rule / total)
        ml_scores.append(ml / total)
        print(f"seed {seed}: rule={rule / total:.2f}  ml={ml / total:.2f}")

    print(
        f"\nmean: rule={sum(rule_scores) / len(SEEDS):.2f}  "
        f"ml={sum(ml_scores) / len(SEEDS):.2f}"
    )


if __name__ == "__main__":
    main()
