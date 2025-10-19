"""
This script performs multi-label text classification using both
TF-IDF/LinearSVC and BERT (Transfer Learning with Hugging Face Transformers).

NOTE: Ensure you have the following packages installed:
pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers torch tensorflow
"""

# --- 0. Setup and Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import jaccard_score, classification_report
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Download necessary NLTK data (required once)
try:
    nltk.download('punkt')
    nltk.download('stopwords')

except Exception as e:
    print(f"NLTK download error (safe to ignore if already downloaded): {e}")


# --- 1. Load the dataset ---
try:
    df = pd.read_csv('aa_dataset-tickets-multi-lang-5-2-50-version.csv')
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("ERROR: Dataset file not found.")
    print("Please make sure 'aa_dataset-tickets-multi-lang-5-2-50-version.csv' is in the correct directory.")
    exit() # Exit if the core data cannot be loaded


# --- 2. Data Preprocessing (NLP) ---

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['cleaned_text'] = df['text'].apply(clean_text)
print("\nCleaned Text Samples:")
print(df[['text', 'cleaned_text']].head())

# Tokenization and Stop Word Removal
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

df['processed_text'] = df['cleaned_text'].apply(tokenize_and_remove_stopwords)
print("\nProcessed Text Samples (Stopwords Removed):")
print(df[['cleaned_text', 'processed_text']].head())

# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['processed_text']).toarray()
print("\nShape of the TF-IDF matrix:", X.shape)

# Prepare Labels for Multi-Label Classification
tag_columns = [col for col in df.columns if col.startswith('tag_')]
df[tag_columns] = df[tag_columns].fillna('')
all_tags = sorted(list(set([tag for tags in df[tag_columns].values for tag in tags if tag != ''])))

for tag in all_tags:
    df[tag] = df[tag_columns].apply(lambda row: int(tag in row.values), axis=1)

print("\nMulti-Hot Encoded Labels Sample:")
print(df[all_tags].head())

y = df[all_tags].values
print("\nShape of the labels matrix:", y.shape)


# --- 3. Split the data into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# # Visualization of the split
# split_sizes = {'Train': X_train.shape[0], 'Test': X_test.shape[0]}
# plt.figure(figsize=(6, 4))
# sns.barplot(x=list(split_sizes.keys()), y=list(split_sizes.values()))
# plt.title('Distribution of Data Split')
# plt.ylabel('Number of Samples')
# plt_ax = plt.gca()
# plt_ax.bar_label(plt_ax.containers[0])
# plt.show()


# --- 4. Train the TF-IDF Multi-label Classification Model (LinearSVC) ---
print("\n--- Training TF-IDF/LinearSVC Model ---")
model_tfidf = OneVsRestClassifier(LinearSVC(random_state=42))
model_tfidf.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = model_tfidf.predict(X_test)

# Evaluate the model
jaccard_accuracy_test = jaccard_score(y_test, y_test_pred, average='samples')
print(f"TF-IDF/LinearSVC Model Jaccard Score (Sample Average) on Test Set: {jaccard_accuracy_test:.4f}")

# --- 5. BERT Transfer Learning Setup ---
print("\n--- Setting up BERT Model (Requires GPU for efficient training) ---")

# Choose a pre-trained BERT model name
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
num_labels = y_train.shape[1]

bert_model = None
try:
    # Attempt to load the TensorFlow model
    bert_model = TFBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        from_pt=False
    )
    print(f"BERT model loaded successfully (TensorFlow).")
except Exception as e:
    print(f"Error loading TensorFlow BERT model: {e}")
    try:
        # Fallback: Try loading from PyTorch and converting (requires PyTorch)
        bert_model = TFBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            from_pt=True
        )
        print("Successfully loaded from PyTorch and converted to TensorFlow.")
    except Exception as e_pt:
        print(f"Error loading and converting BERT model: {e_pt}")

if bert_model is None:
    print("FATAL: Failed to load BERT model. BERT steps will be skipped.")
    jaccard_accuracy_bert = 0.0 # Set a default for the comparison chart
else:
    # --- 6. Prepare data for BERT ---
    encoded_inputs = tokenizer(
        df['cleaned_text'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )

    # Split BERT-processed data
    input_ids_np = encoded_inputs['input_ids'].numpy()
    attention_mask_np = encoded_inputs['attention_mask'].numpy()
    token_type_ids_np = encoded_inputs['token_type_ids'].numpy()

    input_ids_train, input_ids_test, y_train_bert, y_test_bert = train_test_split(
        input_ids_np, y, test_size=0.2, random_state=42
    )
    attention_mask_train, attention_mask_test, _, _ = train_test_split(
        attention_mask_np, y, test_size=0.2, random_state=42
    )
    token_type_ids_train, token_type_ids_test, _, _ = train_test_split(
        token_type_ids_np, y, test_size=0.2, random_state=42
    )

    X_train_bert = {
        'input_ids': tf.constant(input_ids_train),
        'attention_mask': tf.constant(attention_mask_train),
        'token_type_ids': tf.constant(token_type_ids_train)
    }
    X_test_bert = {
        'input_ids': tf.constant(input_ids_test),
        'attention_mask': tf.constant(attention_mask_test),
        'token_type_ids': tf.constant(token_type_ids_test)
    }

    # --- 7. Build and train the BERT model ---
    print("\n--- Training BERT Model ---")
    bert_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.metrics.BinaryAccuracy()]
    )

    # NOTE: Training a BERT model can be very slow on a CPU.
    # Consider adjusting epochs (e.g., to 1) or reducing the dataset size
    # if you are running this without a GPU.
    history = bert_model.fit(
        X_train_bert,
        y_train_bert,
        epochs=2,
        batch_size=16,
        validation_data=(X_test_bert, y_test_bert)
    )

    # --- 8. Evaluate the BERT model ---
    print("\n--- Evaluating BERT Model ---")
    test_predictions = bert_model.predict(X_test_bert)
    # Check if the output is a tuple (logits, sequence_output, etc.) or just logits
    if isinstance(test_predictions, tuple):
        test_predictions_sigmoid = tf.sigmoid(test_predictions[0]).numpy()
    else:
        test_predictions_sigmoid = tf.sigmoid(test_predictions.logits).numpy()

    threshold = 0.5
    y_test_pred_bert = (test_predictions_sigmoid > threshold).astype(int)

    jaccard_accuracy_bert = jaccard_score(y_test_bert, y_test_pred_bert, average='samples')
    print(f"BERT Model Jaccard Score (Sample Average) on Test Set: {jaccard_accuracy_bert:.4f}")

    # --- 9. Compare Model Predictions ---
    print("\n--- Model Comparison ---")
    agreement_count = np.sum(np.all(y_test_pred == y_test_pred_bert, axis=1))
    total_samples = y_test_bert.shape[0]

    print(f"Number of samples where both models made the exact same multi-label predictions: {agreement_count} out of {total_samples}")
    print(f"Percentage of samples with exact agreement: {agreement_count / total_samples * 100:.2f}%")

    # --- 10. Visualization ---
    model_names = ['TF-IDF Model', 'BERT Model']
    jaccard_scores = [jaccard_accuracy_test, jaccard_accuracy_bert]

    # plt.figure(figsize=(8, 5))
    # sns.barplot(x=model_names, y=jaccard_scores)
    # plt.title('Comparison of Model Performance (Jaccard Score)')
    # plt.ylabel('Jaccard Score (Sample Average)')
    # plt.ylim(0, 1)

    # plt_ax = plt.gca()
    # plt_ax.bar_label(plt_ax.containers[0], fmt=':.4f')

    # plt.show()

print("\nScript execution finished.")