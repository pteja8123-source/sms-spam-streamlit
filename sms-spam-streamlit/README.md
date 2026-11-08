# SMS Spam Detector — Local Streamlit Demo

Week 4 of the SkillAudit.ai internship: turn a trained classification model into a local application. Enter an SMS message and receive a **spam** or **legitimate (ham)** prediction with the model's spam score.

The app uses the **TF-IDF + balanced Logistic Regression** pipeline from my Week 2 SMS project. Logistic Regression achieved the higher spam F1-score in that comparison. The saved pipeline includes both preprocessing and the classifier, so the app uses the same feature transformation as training.

## Run locally

Use Python 3.12 in a virtual environment. Download this repository and open a terminal in the project folder.

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL shown in the terminal, normally `http://127.0.0.1:8501`. Enter a message or select an example, then click **Check message**. Stop the server with **Ctrl+C**. No live-server deployment is required.

**The trained model is included.** App predictions work offline after dependency installation; training is not required to start the demo. Run the app from this project folder so its Streamlit configuration is applied.

## Features

- SMS input and two example messages.
- Spam/ham prediction and a clearly labelled model spam score.
- Validation for blank input and text with no recognized model terms.
- Cached loading of the saved model.
- Model details and held-out metrics in an expandable section.
- Local inference without an external prediction API; entered messages are not saved by this app.

The displayed score is an uncalibrated model output, not a guarantee of safety. The model may make mistakes.

## Training and measured results

Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The training script downloads the official archive, removes exact duplicate rows, and uses a stratified 80/20 split with `random_state=42`.

- Raw rows: 5,574; duplicate rows removed: 403; cleaned rows: 5,171.
- Training messages: 4,136; held-out test messages: 1,035.
- TF-IDF: unigrams/bigrams, English stop words, `min_df=2`, `max_features=10000`, sublinear term frequency.
- Logistic Regression: `class_weight='balanced'`, `max_iter=2000`.

| Metric | Measured result |
|---|---:|
| Test accuracy | 97.49% |
| Spam precision | 90.70% |
| Spam recall | 89.31% |
| Spam F1 | 0.9000 |

The saved model was trained and evaluated during preparation of this app. It detected 117 spam messages, missed 14, and incorrectly flagged 12 ham messages. It remains fitted to the training subset, so these held-out scores describe the bundled model; it was not refitted on the test set. Saved/reloaded predictions were checked for equality.

## Reproduce the saved model

```bash
python train_model.py
```

The first run requires internet access. It caches the dataset under `data/` and writes the model and run metadata to `models/`. Restart Streamlit after retraining to refresh the cached pipeline. The model metadata records versions, dataset/model hashes, metrics, and the confusion matrix. Different package versions can change results or make a saved scikit-learn model incompatible; use the supplied requirements, or retrain in your own environment.

Only load trusted joblib/pickle artifacts. The application loads its own bundled model and does not accept model uploads.

## Verify the app

```bash
python -m unittest discover -s tests -v
```

The tests use Streamlit's AppTest to exercise blank input, ham and spam predictions, unfamiliar words, and the example selector without a browser.

All five interface tests passed during preparation. These are simulated interface checks; a browser screen recording is not included and still needs to be made for submission.

## Repository structure

```text
sms-spam-streamlit/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   ├── sms_pipeline.joblib
│   └── model_metadata.json
└── tests/
    └── test_app.py
```

Include the `models/` folder when uploading to GitHub. Downloaded dataset files, virtual environments, and private configuration should not be uploaded.

## Limitations and learning outcomes

This historical English-language dataset does not represent every modern spam campaign, language, or platform. Accuracy alone is insufficient with imbalanced data, and model scores are not calibrated risk estimates. Exact deduplication does not remove all near-duplicate campaigns. This project demonstrates how to save a complete pipeline, load it consistently, validate inputs, and separate training from inference in a simple local interface.

## Dataset attribution

Almeida, T. & Hidalgo, J. (2011). [SMS Spam Collection](https://doi.org/10.24432/C5CC84). UCI Machine Learning Repository. Dataset license: CC BY 4.0.
