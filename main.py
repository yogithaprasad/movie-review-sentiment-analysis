import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_FILE_PATH = 'data/IMDB Dataset.csv'
SAMPLE_SIZE = 10000  # Increased sample size for better results, can be adjusted
OUTPUT_DIR = 'output_plots'

# --- Ensure NLTK data is downloaded ---
# This is a better way to handle NLTK downloads in a script
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# --- Helper Functions for Text Preprocessing ---
def clean_html(text):
    """Removes HTML tags from text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_characters(text):
    """Removes special characters, keeping alphanumeric and spaces."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def stem_words(text, stemmer):
    """Stems words in a text."""
    return ' '.join([stemmer.stem(word) for word in text.split()])

def preprocess_text(df):
    """Applies all preprocessing steps to the review column."""
    print("Starting text preprocessing...")
    # Initialize tools
    ps = PorterStemmer()
    english_stopwords = set(stopwords.words('english'))

    df['review'] = df['review'].apply(clean_html)
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(remove_special_characters)
    df['review'] = df['review'].apply(
        lambda text: ' '.join([word for word in text.split() if word not in english_stopwords])
    )
    df['review'] = df['review'].apply(lambda text: stem_words(text, ps))
    print("Text preprocessing complete.")
    return df

def train_and_evaluate(X, y):
    """Trains and evaluates the Naive Bayes models."""
    print("Splitting data and training models...")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'model_instance': model # Store the trained model instance
        }
        print(f"  - {name} trained. Accuracy: {accuracy:.4f}")
        
    return results

def plot_and_save_results(results):
    """Generates and saves plots for model comparison."""
    print("Generating and saving plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

    model_names = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]
    precisions = [res['precision'] for res in results.values()]
    recalls = [res['recall'] for res in results.values()]
    f1_scores = [res['f1_score'] for res in results.values()]

    # --- Plot 1: Accuracy Comparison ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_accuracy_comparison.png'))
    plt.close() # Close the plot to free memory

    # --- Plot 2: Precision, Recall, F1-Score Comparison ---
    metrics_df = pd.DataFrame({
        'Model': model_names * 3,
        'Metric': ['Precision'] * 3 + ['Recall'] * 3 + ['F1-Score'] * 3,
        'Score': precisions + recalls + f1_scores
    })
    plt.figure(figsize=(12, 7))
    sns.barplot(data=metrics_df, x='Model', y='Score', hue='Metric', palette='plasma')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_performance_metrics.png'))
    plt.close()
    
    # --- Plot 3: Confusion Matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices', fontsize=16)
    for i, (name, res) in enumerate(results.items()):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))
    plt.close()

    print(f"All plots saved successfully in '{OUTPUT_DIR}' folder.")


def main():
    """Main function to run the entire sentiment analysis pipeline."""
    # 1. Load and sample data
    df = pd.read_csv(DATA_FILE_PATH)
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})
    
    # 2. Preprocess text data
    df = preprocess_text(df)

    # 3. Vectorize text
    cv = CountVectorizer(max_features=5000) # Limit features to prevent memory issues
    X = cv.fit_transform(df['review']).toarray()
    y = df['sentiment'].values

    # 4. Train models and get results
    results = train_and_evaluate(X, y)

    # 5. Visualize and save plots
    plot_and_save_results(results)

if __name__ == "__main__":
    main()