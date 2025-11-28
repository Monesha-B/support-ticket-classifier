import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load the data
df = pd.read_csv("data.csv")

# Step 2: Convert text column into numbers (TF-IDF)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X = vectorizer.fit_transform(df["text"])

# Step 3: Get the labels
y = df["label"]

print("Shape of numeric matrix:", X.shape)
print("Example labels:", y.head())


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Training complete.")

# --- SAVE model and vectorizer ---
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("Saved model.joblib and vectorizer.joblib")


# #  To see the  vectorised data

# import pandas as pd

# words = vectorizer.get_feature_names_out()
# df_vectors = pd.DataFrame(X.toarray(), columns=words)
# print(df_vectors.head())
# # to see the word clearly
# row_index = 2  # change this to any row you want
# row = df_vectors.iloc[row_index]

# # print words that actually appear in this sentence
# print(row[row > 0])

# row_1 = df_vectors.iloc[0]
# print(row_1[row_1 > 0])

from sklearn.linear_model import LogisticRegression

# After training:
words = vectorizer.get_feature_names_out()

# Get weight matrix (rows = classes, columns = words)
weights = model.coef_

# Put into a DataFrame for easy reading
import pandas as pd
weights_df = pd.DataFrame(weights, columns=words)

print("Classes:", model.classes_)
print(weights_df.head())  # first few columns

print("\n==============================")
print("   Support Ticket Classifier")
print("==============================")
print("""
This tool helps you automatically understand support messages.
It can classify your issue into one of these categories:

  • Billing
  • Bug / Technical Problem
  • Feature Request

How to use it:
  - Type your issue in a full sentence.
  - Try to be clear (e.g., “payment failed”, “app crashes”, “need dark mode”).
  - If your message is too short or unclear, I will ask you to type more.

Type 'q' anytime to quit.

""")


# ---- TEST THE MODEL ON NEW SENTENCES ----

while True:
    user_text = input("\nEnter a ticket (or 'q' to quit): ")

    # NEW PART: Stop empty input from going to the model
    if user_text.strip() == "":
        print("Please type something.")
        continue

    if user_text.lower().strip() == "q":
        break

    X_new = vectorizer.transform([user_text])
    pred = model.predict(X_new)[0]

    print("Predicted label:", pred)
