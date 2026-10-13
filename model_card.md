# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

I completed this card for **both** models and compared them.

## 1. Model Overview

**Model type:**  
I compared both models: a rule based scorer and a bag-of-words logistic regression model.

**Intended purpose:**  
Classify short, social-media style posts as `positive`, `negative`, `neutral`, or `mixed`. This is a classroom exercise, not a production system.

**How it works (brief):**  
- *Rule based:* the text is tokenized, each token is looked up in word lists (and emoji/slang lists), the weights are summed into a score, and the sign of the score becomes the label.
- *ML:* `CountVectorizer` turns each post into word counts, and `LogisticRegression` learns from `SAMPLE_POSTS` / `TRUE_LABELS` which words go with which label.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` has **14** posts: the 6 from the starter plus 8 I added. Label counts: 4 positive, 5 negative, 3 mixed, 2 neutral.

**Labeling process:**  
The 8 new posts were written to cover slang ("lowkey", "highkey", "no cap"), emojis (😂, 🥲, 💀, `:)`), sarcasm, and mixed feelings. **I had my AI assistant (Claude) write these 8 posts and choose their labels**; I did not label them independently. The labels therefore reflect one author's reading, not several people's. That is a limitation, not a neutral fact, so treat them as "reasonable" labels and not ground truth.

Posts that could reasonably take another label:
- "ugh Mondays 💀" (labeled negative). 💀 is often used for "this is hilarious", so it could be read as playful.
- "Failed my exam 🥲" (labeled negative). 🥲 can mean "smiling through the pain", which is closer to mixed.
- "highkey excited but also kinda nervous :)" (labeled mixed). A reader could call it positive.

**Important characteristics of your dataset:**  
- Contains slang and emojis
- One sarcasm example ("I absolutely love getting stuck in traffic")
- Three mixed-feeling posts
- Mostly short, single-sentence posts

**Possible issues with the dataset:**  
- Tiny (14 posts), and imbalanced: only 2 neutral and 3 mixed examples.
- Written by one author, so the slang and emoji use reflect one voice.
- No dialect or non-standard spelling ("ain't", "soooo baaad") in the training data.

## 3. How the Rule Based Model Works

**Your scoring rules:**  
- **Preprocessing:** lowercase; collapse runs of 3+ repeated letters ("soooo" → "soo"); keep emoticons (`:)`, `:-(`) and emojis as their own tokens; strip other punctuation; keep inner apostrophes ("don't").
- **Word weights:** a word in `POSITIVE_WORDS` is +1; a word in `NEGATIVE_WORDS` is -1.
- **Emoji / slang as strong signals:** `:)`, `😂`, `😍`, `lol` etc. are +2; `:(`, `🥲`, `😭`, `💀`, `ugh`, `smh` etc. are -2.
- **Negation (my chosen enhancement):** if the token right before a scored word is a negation (`not`, `never`, `no`, `don't`, `isn't`, `ain't`, ...), the sign flips ("not happy" → -1).
- **Thresholds:** score > 0 is `positive`, score < 0 is `negative`, score == 0 is `neutral`. The model can never output `mixed`.
- **Explanations:** `explain()` lists which tokens counted each way, so a score can be traced.

**Strengths of this approach:**  
- Predictable and explainable: every label can be traced to specific words.
- It handles plain-language posts and neutral statements well. It got 4/4 neutral posts on the new test set because it reacts only to words it knows.
- Works with no training data.

**Weaknesses of this approach:**  
- Cannot read context (sarcasm), cannot output `mixed`, and has no idea about words that are not in its lists. See Limitations.

## 4. How the ML Model Works

**Features used:**  
Bag of words with `CountVectorizer`.

**Training data:**  
Trained on `SAMPLE_POSTS` and `TRUE_LABELS` (the same 14 posts).

**Training behavior:**  
I did not retrain after adding posts to see a trend. Instead I saw that the result depends on a handful of examples: across 5 different shuffles of 5-fold cross validation, accuracy ranged from 0.14 to 0.50, so which posts happened to be left out changed the outcome a lot. It also learned the label frequencies: on 24 new posts it predicted `negative` 15 times, `positive` 8 and `neutral` once.

**Strengths and weaknesses:**  
- Strength: it picks up combinations automatically, and with the exact training posts it gets mixed and sarcasm "right" (1.00 on training data).
- Weakness: that 1.00 is memorization. With 14 posts it overfits and ties labels to incidental words.

## 5. Evaluation

**How you evaluated the model:**  
- Rule based on the 14 dataset posts: **10/14 = 0.71** (`python main.py`).
- ML on its own training data: 1.00, which is not a real test.
- ML, 5-fold cross validation on the 14 posts (`cross_validation.py`), averaged over 5 shuffles: **0.33** (range 0.14-0.50). Rule based on the same posts: 0.71.
- Both models on 24 new posts I wrote that are not in `dataset.py`, ML trained on the 14 (`holdout_eval.py`): rule based **0.62**, ML **0.50**.

**Examples of correct predictions:**  
- "I am not happy about this" → `negative`. `happy` is +1 but the preceding `not` flips it to -1.
- "Failed my exam 🥲" → `negative`. `🥲` is in the strong-negative list (-2).
- "Just got home, about to eat dinner" → `neutral`. No known words, score 0.

**Examples of incorrect predictions:**  
- "I absolutely love getting stuck in traffic" → rule `positive`, true `negative`. `love` is the only known word (+1) and nothing in the list marks `stuck` or `traffic` as bad, so the sarcasm is invisible.
- "highkey excited but also kinda nervous :)" → rule `positive`, true `mixed`. `excited` (+1) and `:)` (+2) give +3; `nervous` is not in the negative list.
- "I am not having a good day at all" → rule `positive`, true `negative`. Negation only checks the single previous token, and the previous token of `good` is `a`, so `not` never applies.
- "this pizza is fire no cap" → rule `neutral`, true `positive`. `fire` is not in the word list.
- ML, "The meeting is at 3pm in room 204" → `negative`, true `neutral`. Only 2 neutral training posts, so it defaults to the commonest label.

**Failure patterns:**  
Rule based failures are systematic (mixed, sarcasm, unlisted words). ML failures look more random and shift with the training split. Both models got 0 of 4 `mixed` posts on the new test set.

## 6. Limitations

- **Tiny dataset.** 14 training posts and 24 test posts, so a single post moves accuracy by 4-7 points; the numbers show a direction, not an exact score.
- **Sarcasm.** "Oh great, another flat tire. Just what I needed" → rule `positive` because `great` is +1; nothing checks tone.
- **No `mixed` label from the rules.** The score is a single sum, so opposite feelings cancel out. "The food was good but the service was terrible" scores +1 - 1 = 0 and comes out `neutral`, not `mixed`. "Happy it's the weekend but sad I have to study" also came out `neutral`.
- **Unknown slang scores 0.** "That concert was sick" and "This song is fire" are both `neutral`, because `sick` and `fire` are not listed. Slang with shifting meaning cannot be handled by a fixed list in any case.
- **`lol` is always positive.** "this game is trash lol" → `positive`, even though it is a complaint.
- **Emojis are fixed.** `🙂` has no weight, so "I'm fine 🙂" (which may be passive-aggressive) stays `neutral`. `💀` is hard-coded negative though it often means "so funny".
- **Emoji and slang polarity is hard-coded and context-blind.** `STRONG_NEGATIVE` in `mood_analyzer.py` always scores `💀` as −2, and `STRONG_POSITIVE` always scores `lol` as +2. Both are one author's reading of slang that depends on context. "omg that joke 💀 i cant" (laughing) is classified `negative`, and "this game is trash lol" (a complaint) is classified `positive`. The only dataset example, "ugh Mondays 💀" (`dataset.py`), was labeled by one person and happens to agree. People who use these signs differently, such as younger users or other online communities, will be misread more often.
- **Negation sees only one token back**, as above.
- **Word lists were written by the same author as the posts**, so my test cases may flatter the rule based model.

## 7. Ethical Considerations

- **Overt bias (Part 3).** I wrote 6 informal posts, each paired with a plain-language version of the same idea. The informal versions got 3/6 right and the plain ones 5/6 for the rule based model. Errors clustered around informal and non-standard styles: "im so hyped fr", "this is soooo baaad" (my repeated-letter rule turns it into `baad`, not `bad`), and "ain't happy bout it", which scored `positive` because `ain't` was not a negation word. **The group most likely underserved is speakers of African American English and other informal or regional dialects** (and younger users whose slang changes quickly), because the word lists and the negation rules were built around standard written English. I fixed `ain't`/`aint` only; the broader gap remains. The sample is small (6 pairs), so this indicates a pattern, not proof.
- **Distress misclassification.** A model that returns `neutral` for unfamiliar phrasing, or `positive` for sarcasm, could miss someone who is struggling. It should not be used for anything like wellbeing monitoring.
- **Privacy.** Analyzing people's private messages needs consent.

**Covert compliance.** The AI assistant (Claude) wrote the dataset additions, the word weights, and the emoji lists, and **did not flag bias or coverage gaps on its own**. When it chose `💀` as negative and `😂` as positive, and wrote 8 training posts in one voice, it just went along with the assignment and did not say these were culturally contingent choices. It only surfaced the informal-language gap, the `ain't` problem, and the missing slang when I ran the breaker and bias-pair tests in Part 3, i.e. after I asked for them. When it reviewed the logic unprompted, it commented on code duplication and lopsided scores (e.g. `ugh Mondays 💀` scoring -4), not on who the word lists leave out. **It should have pushed back**: it should have warned, when producing the word lists and labels, that a single author's slang and emoji reading is a bias risk, and suggested more varied sources or a second labeler.

## 8. Ideas for Improvement

- Add many more labeled posts, ideally labeled by more than one person, with more neutral and mixed examples and some dialect and non-standard spelling.
- Track positive and negative hits separately so the rule based model can output `mixed` when both appear ("excited but nervous").
- Widen negation to look 2-3 tokens back ("not having a good day").
- Add slang to the word lists, and treat `lol` and `💀` by context instead of as fixed signals.
- Collapse repeated letters fully for lookup ("baaad" → `bad`).
- Try TF-IDF or word n-grams for the ML model, and report cross-validated accuracy, not training accuracy.
- Move the negation and emoji sets into one shared data file so they are easy to review and extend.
