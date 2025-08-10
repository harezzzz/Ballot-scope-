import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from textblob import TextBlob

# Load the dataset (Ensure the CSV file exists and has necessary columns)
try:
    df = pd.read_csv(r"C:\Users\haris\Downloads\election_data.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: The file 'election_data.csv' was not found.")
    exit()

# Display first few rows
print(df.head())

# Handling missing values
df.dropna(inplace=True)

# Identify categorical columns excluding the target variable
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encoding categorical variables dynamically
label_encoders = {}
for col in categorical_cols:
    if col != 'Party':  # Avoid re-encoding target variable here
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Encoding the target variable ('Party')
label_encoder = LabelEncoder()
df['Party'] = label_encoder.fit_transform(df['Party'])

# Feature Selection (Modify based on your dataset)
X = df.drop(columns=['Party'])  # Features
y = df['Party']  # Target variable

# Sentiment Analysis Function
def get_sentiment(text):
    analysis = TextBlob(str(text))  # Convert to string and analyze
    return 1 if analysis.sentiment.polarity > 0 else (-1 if analysis.sentiment.polarity < 0 else 0)

# Apply Sentiment Analysis if the 'tweets' column exists
if 'tweets' in df.columns:
    df['Sentiment'] = df['tweets'].apply(get_sentiment)
    X['Sentiment'] = df['Sentiment']

# Standardizing Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling Class Imbalance using SMOTE (Only if imbalance exists)
class_counts = Counter(y)
min_class_size = min(class_counts.values())

if min_class_size >= 2:  # Ensure there are enough samples for SMOTE
    smote = SMOTE(k_neighbors=min(3, min_class_size - 1), random_state=42)  # Adjust neighbors
    X_scaled, y = smote.fit_resample(X_scaled, y)
    print("✅ SMOTE applied to balance the dataset.")
else:
    print("⚠️ SMOTE skipped due to insufficient samples per class.")

# Splitting Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Model Selection
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

# Training and Evaluating Models    
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n🔹 Model: {name}")
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    print("🔢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Class Distribution After Resampling
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("📊 Class Distribution After Processing")
plt.xlabel("Party")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()