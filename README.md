🩺 Breast Cancer Diagnosis Classifier

An end-to-end machine learning project that predicts whether a breast tumor is malignant or benign from measurements of cell nuclei. It covers the complete ML workflow: data exploration, model comparison, hyperparameter tuning, evaluation, interpretability, and model persistence.

⚠️ Disclaimer: This project is for educational purposes only. It is not a medical device and must not be used for real diagnosis.

Show Image Show Image Show Image

📚 Documentation
Page	What's inside
Installation	Setup, requirements, troubleshooting
Dataset	Data source, feature descriptions, class balance
Methodology	Pipeline design, validation strategy, tuning
Results	Metrics, confusion matrix, feature importance
Usage & API	Running the script, loading the model, batch predictions
FAQ	Common questions and limitations
Contributing	How to contribute

📌 Highlights
Compares 4 models using stratified 5-fold cross-validation
Uses a scikit-learn Pipeline, so scaling happens inside each CV fold (no data leakage)
Automated hyperparameter tuning with GridSearchCV
Evaluation with classification report, confusion matrix, and ROC curve
Model-agnostic interpretability via permutation importance
Saves the trained model with joblib for reuse

⚡ Quick Start
bash
git clone https://github.com/<your-username>/breast-cancer-classifier.git
cd breast-cancer-classifier

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python cancer_classifier.py
The script prints results to the console and writes three files: evaluation.png, feature_importance.png, and cancer_model.joblib.

📈 Results at a Glance
Final model (tuned Logistic Regression, C=1) on the 20% held-out test set (114 samples):
Metric	Score
Accuracy	0.982
ROC-AUC	0.995
Malignant recall	0.976 (41 of 42 caught)
Benign recall	0.986 (71 of 72 caught)

Show Image Show Image

See Results for the full breakdown.

📁 Project Structure
.
├── cancer_classifier.py     # Full ML pipeline
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── docs/
│   ├── 01-installation.md
│   ├── 02-dataset.md
│   ├── 03-methodology.md
│   ├── 04-results.md
│   ├── 05-usage.md
│   └── 06-faq.md
├── evaluation.png           # Generated on run
├── feature_importance.png   # Generated on run
└── cancer_model.joblib      # Generated on run
🛠️ Tech Stack

Python · scikit-learn · pandas · Matplotlib · joblib

🌱 Roadmap
 Add SHAP values for per-patient explanations
 Tune the decision threshold to prioritize malignant recall
 Try XGBoost / LightGBM or a small neural network
 Serve the model with FastAPI or Streamlit
 Add unit tests and GitHub Actions CI.
 
 
 By-- Saurabh Kumar Pal
