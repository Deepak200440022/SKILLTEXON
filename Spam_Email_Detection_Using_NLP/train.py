from data_preprocessing import vector, df, vectorizor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

X_train, X_test, y_train,y_test = train_test_split(vector, df["labels"], random_state=42, test_size=0.2, shuffle=True)

model = MultinomialNB()
model.fit(X_train,y_train)


# Save model and vectorizer
joblib.dump(model, "spam_classifier_model.joblib")
joblib.dump(vectorizor, "tfidf_vectorizer.joblib")
# Predict on test set
y_pred = model.predict(X_test)


y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
# Save stats
stats = {
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": report,
    "accuracy": accuracy_score(y_test, y_pred)
}

with open("model_stats.json", "w") as f:
    json.dump(stats, f)

# Evaluate
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(report)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
