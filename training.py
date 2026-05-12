import pickle
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

_training_translator = None


def translate_kannada_to_english(text):
    """
    Translate Kannada to English using the same local NLLB stack as the web app.
    Used when prepare_data(..., use_translation=True).
    """
    global _training_translator
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).strip()
    if len(text) < 2:
        return text
    try:
        if _training_translator is None:
            from config.settings import Settings
            from services.translation.nllb_translator import get_nllb_translator

            _training_translator = get_nllb_translator(Settings())
        return _training_translator.translate(text)
    except Exception as e:
        print(f"Local translation error for text: {text[:50]}... Error: {e}")
        return str(text)

def load_and_prepare_data(news_file):
    """
    Load Kannada dataset and prepare for training
    """
    print("Loading data...")
    
    # Load dataset
    news_df = pd.read_csv(news_file)
    print(f"Dataset shape: {news_df.shape}")
    print(f"Columns: {news_df.columns.tolist()}")
    
    # Display sample data
    print("\nSample data:")
    print(news_df.head())
    
    # Check label distribution
    print(f"\nLabel distribution:")
    print(news_df.iloc[:, 1].value_counts())
    
    return news_df

def preprocess_text(text):
    """
    Clean and preprocess text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove special characters and digits but keep Kannada characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def prepare_data(news_df, max_features=5000, max_len=100, use_translation=False):
    """
    Prepare data for LSTM model
    """
    print("Preparing data for training...")
    
    # Create a copy to avoid modifying original
    df = news_df.copy()
    
    # Preprocess text
    df['cleaned_text'] = df.iloc[:, 0].apply(preprocess_text)
    
    if use_translation:
        print("Translating Kannada text to English with local NLLB (slow on CPU)...")
        df["english_text"] = df["cleaned_text"].apply(translate_kannada_to_english)
        text_column = "english_text"
    else:
        text_column = "cleaned_text"
        print("Working with Kannada text directly...")
    
    # Remove empty texts
    df = df[df[text_column].str.len() > 0]
    
    # Prepare features and labels
    X = df[text_column].values
    y = df.iloc[:, 1].values
    
    print(f"Data shape after cleaning: {X.shape}")
    print(f"Label distribution: {pd.Series(y).value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, df

def create_lstm_model(max_features, max_len, num_classes=1):
    """
    Create LSTM model architecture
    """
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    
    # LSTM layers
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    
    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer
    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']  # Removed precision and recall
        )
    else:
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
    
    return model

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Real', 'Fake']):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics(y_true, y_pred):
    """
    Plot additional metrics
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    NEWS_FILE = 'kannada_fake_news_dataset.csv'  # Replace with your file path
    MAX_FEATURES = 5000  # Reduced for Kannada text
    MAX_LEN = 100
    BATCH_SIZE = 32
    EPOCHS = 15
    USE_TRANSLATION = False  # Set to False to work with Kannada text directly
    
    # Load data
    news_df = load_and_prepare_data(NEWS_FILE)
    
    # Prepare data
    X_train, X_test, y_train, y_test, tokenizer, processed_df = prepare_data(
        news_df, MAX_FEATURES, MAX_LEN, use_translation=USE_TRANSLATION
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Save processed data for inspection
    processed_df.to_csv('processed_kannada_news.csv', index=False)
    print("Processed data saved to 'processed_kannada_news.csv'")
    
    # Create model
    model = create_lstm_model(MAX_FEATURES, MAX_LEN)
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_kannada_news_model.h5', monitor='val_accuracy', 
                                save_best_only=True, mode='max', verbose=1)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_metrics(y_test, y_pred)
    
    # Save final model and artifacts
    model.save('kannada_news_lstm_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save configuration
    config = {
        'max_features': MAX_FEATURES,
        'max_len': MAX_LEN,
        'use_translation': USE_TRANSLATION
    }
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print("\n✅ Model and artifacts saved successfully!")
    print("📁 Files saved:")
    print("   - kannada_news_lstm_model.h5 (Trained model)")
    print("   - best_kannada_news_model.h5 (Best model during training)")
    print("   - tokenizer.pkl (Text tokenizer)")
    print("   - model_config.pkl (Model configuration)")
    print("   - processed_kannada_news.csv (Processed data)")
    print("   - training_history.png (Training graphs)")
    print("   - confusion_matrix.png (Confusion matrix)")
    print("   - performance_metrics.png (Performance metrics)")

if __name__ == "__main__":
    main()
