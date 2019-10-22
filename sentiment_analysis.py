import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#### https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("SMSSpamCollection", sep="\t", header=None)
df.columns = ["label", "text"]
y = df.pop("label")

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42, stratify=y)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train["text"].values)
X_test = vectorizer.transform(X_test["text"].values)

X_train = pd.DataFrame(X_train.A, columns=vectorizer.get_feature_names())
X_test = pd.DataFrame(X_test.A, columns=vectorizer.get_feature_names())

lab = LabelEncoder()
y_train = lab.fit_transform(y_train)
y_test = lab.transform(y_test)

clr = LogisticRegression(solver="lbfgs")
clr = clr.fit(X_train, y_train)
print(accuracy_score(clr.predict(X_train), y_train))
print(accuracy_score(clr.predict(X_test), y_test))
