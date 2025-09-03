import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# load
df = pd.read_csv(r"E:\PYTHON\projects\SPAM\DATA\spam.csv",encoding="latin1")

# split
X_train, X_test, y_train, y_test = train_test_split(
    df["v2"], df["v1"], test_size=0.2,
    stratify=df["v1"], random_state=42
)

# pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", ngram_range=(1,2),
        min_df=2, max_df=0.9, sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000, class_weight="balanced", n_jobs=-1
    ))
])

# fit
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# save
joblib.dump(model, "models/spam_model.pkl")
print("Saved to models/spam_model.pkl")