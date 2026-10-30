# RAG-Enhanced Transformers for Fake News Detection in Short News Statements

## Problem statement
Short news statements and titles are often ambiguous. Polysemous words and missing context make them hard to classify. The task is to classify each statement as **Fake (1)** or **Real (0)**. The models get retrieved context: similar, previously fact-checked claims plus word-sense glosses.

## Dataset
- Source: https://github.com/ApnaClassroom/Dataset/blob/main/News.csv (`data/raw/News.csv`)
- 10,240 statements with columns `News` and `Fake`. The counts (5,752 `TRUE` / 4,488 `FALSE`) match the binarised **LIAR / PolitiFact** training split, and the `Fake` column is `TRUE` for *truthful* statements. The target is therefore `label = 1 (Fake)` when `Fake == FALSE`. See `data/README.md`.
- Cleaning: whitespace and quote normalisation; 21 rows removed (duplicate statements and statements with conflicting labels). This leaves 10,219 rows, split 70/15/15 (stratified, seed 42) into 7,153 / 1,533 / 1,533.

## IEEE paper
Anusha, K. M. Shenoy, S. N. Pai, "RAG-Enhanced DistilBERT Model for Contextual Understanding of Polysemy in Digital News Titles," *IEEE Access*, vol. 14, pp. 90425–90443, 2026, doi:10.1109/ACCESS.2026.3702098.
The full text could not be downloaded in the build environment, so the implementation follows the paper's abstract: DistilBERT plus retrieval-augmented context for polysemy disambiguation.

## Proposed system
1. **RAG retriever.** A TF-IDF (1–2 gram) cosine search over a knowledge base of *training-split* fact-checked claims and their verdicts returns the top 3 claims. Training rows use leave-one-out retrieval, so no label leaks.
2. **Polysemy module.** It finds the most polysemous content words (at least 4 WordNet senses). Each one is disambiguated with Lesk, using the statement **plus the retrieved evidence** as context.
3. **Transformer.** The input is `statement </s> Evidence: [fake] … | [real] … Senses: word: gloss`.
4. **Stronger models than the paper's:** RAG-RoBERTa-base, RAG-XLM-R-base and a weighted soft-voting ensemble.

### Algorithms / models trained
| Folder in `models/` | Model | Role |
|---|---|---|
| `tfidf_logreg` | TF-IDF (word + char) + Logistic Regression | classical baseline |
| `roberta_base` | RoBERTa-base on the statement only | standard transformer |
| `rag_roberta_base` | RAG-RoBERTa-base | proposed |
| `rag_distilroberta_kd` | 6-layer student, alternate-layer initialisation + knowledge distillation from RAG-RoBERTa | paper's RAG-DistilBERT recipe |
| `rag_xlmr_base` | RAG-XLM-RoBERTa-base | proposed |
| `ensemble_soft_voting` | weighted average of RAG-RoBERTa, RAG-XLM-R and TF-IDF (weights fitted on validation) | proposed |

**Deviations (environment constraints):**
- huggingface.co is blocked in the build environment, so DistilBERT, DeBERTa and sentence-transformers weights were unavailable. The official RoBERTa-base and XLM-R-base weights are downloaded from `download.pytorch.org` and converted to Hugging Face format inside the notebook (`models/pretrained_backbones/`).
- Training was CPU-only. Each run used 2 epochs, and the embeddings plus the lowest 4 encoder layers of the 12-layer models were frozen.

## Folder structure
```text
PROJECT_MAIN/
├── code/
│   ├── 01_Data_Preprocessing_and_Model_Training.ipynb   # full pipeline (executed, outputs included)
│   └── 02_Best_Model_Prediction.ipynb                   # independent inference with best model only
├── data/  raw/News.csv · processed/*.csv · README.md
├── models/
│   ├── tfidf_logreg/ roberta_base/ rag_roberta_base/ rag_distilroberta_kd/ rag_xlmr_base/ ensemble_soft_voting/
│   ├── rag_retriever/              # TF-IDF retriever, knowledge base, RAG config, label map
│   ├── pretrained_backbones/       # converted RoBERTa-base / XLM-R-base (created by the notebook)
│   ├── best_model/                 # best model + its retriever + best_model_info.json
│   └── model_evaluation_results/   # metrics CSVs, reports, confusion matrices, ROC, EDA plots, predictions
├── webapp/                         # Flask web application (see below)
├── requirements.txt
└── README.md
```
Transformer models are saved as `model.pth` (PyTorch state_dict), `config.json`, the tokenizer files and `metadata.json`. The scikit-learn pipeline is saved as `model.joblib`.

> **Note on the delivered zip:** the chat upload limit is 30 MB, so the transformer weight files (`model.pth`, 0.3–1.1 GB each) are not included. Everything else is: configs, tokenizers, retriever, TF-IDF model and all results. Run the training notebook once to regenerate the weights in `models/`, including `models/best_model/`. The prediction notebook and the web app need them.

## Installation
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
NLTK WordNet and stop-words download automatically on first run.

## How to run the training notebook
```bash
cd code
jupyter notebook 01_Data_Preprocessing_and_Model_Training.ipynb   # Run All
```
- A full CPU run takes about 4 hours on 2 cores; a GPU is used automatically if present.
- `FAST_DEV_RUN=1` runs a small smoke test.
- `RESUME_TRAINING=1` reuses models already completed by an interrupted run.

## How to run the prediction notebook
```bash
cd code
jupyter notebook 02_Best_Model_Prediction.ipynb
```
This notebook loads only `models/best_model/` and applies the same cleaning, retrieval and word-sense pipeline. It prints the label, P(Fake) and confidence. Edit `user_input` in the last cell to try your own statement.

## Model results (held-out test set, 1,533 statements)
| Model | Accuracy | Precision | Recall | F1 (Fake) | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.6190 | 0.5632 | 0.5833 | 0.5731 | 0.6146 | 0.6412 |
| RoBERTa-base | 0.6314 | 0.5661 | **0.6815** | **0.6185** | 0.6310 | 0.6681 |
| RAG-RoBERTa-base | 0.6438 | 0.6154 | 0.5000 | 0.5517 | 0.6281 | 0.6877 |
| RAG-DistilRoBERTa-KD | **0.6588** | **0.6408** | 0.5045 | 0.5645 | **0.6421** | **0.7032** |
| RAG-XLM-R-base | 0.6164 | 0.5638 | 0.5521 | 0.5579 | 0.6096 | 0.6524 |
| Soft-voting ensemble | 0.6393 | 0.5917 | 0.5714 | 0.5814 | 0.6322 | 0.6773 |

Validation metrics, per-model reports and plots are in `models/model_evaluation_results/`.

## Best model
**`roberta_base`.** It was selected by the rule fixed before training: highest **validation** macro-F1 (0.6173). The ensemble scored 0.6168 and RAG-RoBERTa 0.6145, so this is a near tie. On test it reaches accuracy 0.6314, macro-F1 0.6310 and ROC-AUC 0.6681. It is saved in `models/best_model/`.

## Conclusion
- RAG context improved RoBERTa's test accuracy from 0.6314 to 0.6438 (+1.96% relative) and ROC-AUC from 0.668 to 0.688. It also lowered Fake-class recall at the 0.5 threshold, so macro-F1 did not improve.
- The distilled RAG student (the paper-style model) had the best test scores, but test data was not used for selection.
- The paper's +11.7% gain was not reproduced on this dataset and compute budget.
- LIAR-style claims are hard to verify from text alone. A larger, current fact-check knowledge base and full GPU fine-tuning are the most promising next steps.

---

## Web application (FactLens)
A Flask, SQLite, Bootstrap 5 and Chart.js app that serves the saved best model through `webapp/predictor.py`. It uses the same pipeline code as the notebooks.

```bash
pip install -r requirements.txt
cd webapp
python app.py            # → http://127.0.0.1:5000
```
- **Default admin:** username `admin`, password `Admin@123`. It is created automatically in `webapp/instance/factlens.db` on first start. Change it for real deployments.
- The database tables and the secret key are created on first start. Optional settings: the `FACTLENS_SECRET_KEY` and `FACTLENS_DB` environment variables, and `FACTLENS_SECURE_COOKIES=1` when serving over HTTPS. For production, use a WSGI server, e.g. `pip install waitress && waitress-serve --port 5000 app:app`.

**Features**
- **Public pages:** Home, About (paper, dataset, real test results), User Login, Admin Login, Register.
- **Registration:** name, email and password. New accounts get status `PENDING`, and pending users see a pending message instead of the dashboard. `REJECTED` users cannot log in. Public registration always creates role `user`.
- **User area:**
  - Dashboard with stats, verdict and per-day charts, model performance chart and an AJAX quick check.
  - Check News page showing the verdict, confidence, P(fake) and retrieved fact-checked claims.
  - History of the user's own records and activity, with a detail view.
  - Profile with name and password change.
- **Admin area:**
  - Dashboard with user and prediction stats, pending approvals and the latest activity.
  - User management: approve, reject and delete.
  - All prediction records and the full activity log.
- **Security:**
  - Passwords are hashed with Werkzeug scrypt.
  - Login and role checks are enforced on the server for every page and API. The user is reloaded from the database on each request, so rejecting a logged-in user takes effect immediately.
  - CSRF tokens protect all POSTs and the JSON API.
  - Sessions are HttpOnly and SameSite=Lax with a 2-hour lifetime, and the session is reset at login.
  - Inputs are validated, and security headers are set.
  - Users can open only their own records; other users' records return 404.

**Tested:**
- An automated end-to-end test (48 checks) passed against the real best model. It covered registration → pending → blocked login → admin login → approve/reject → user login → dashboard → prediction (form and API) → history.
- It also checked that other users' records are hidden, that anonymous, user-to-admin and CSRF-less requests are blocked, and that logout works.
- A browser run with Playwright showed no JavaScript errors.

**Known limitations:**
- The deployed model reads the statement alone. The app still shows the retrieved claims, clearly labelled "reference only".
- The model loads on the first prediction (about 3 s on the build machine); after that, CPU inference takes under 0.1 s per statement.
- There is no email verification and no login rate-limiting.
