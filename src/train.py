import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
import numpy as np
import optuna
import wandb
from wandb.integration.keras import WandbCallback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import *

# Enable dynamic memory growth for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# WandB Project Setup
WANDB_PROJECT = "news-recommendation"
WANDB_ENTITY = "augbirch"  # Replace with your WandB username or entity
wandb.login()

SIZE = "demo"
print("size: ", SIZE)
# Directory Setup
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
TRAIN_DIR = os.path.join(BASE_DIR, SIZE, 'train')
VALID_DIR = os.path.join(BASE_DIR, SIZE, 'validation')
ARTICLE_DIR = os.path.join(BASE_DIR, SIZE)
EMBEDDING_DIR = os.path.join(BASE_DIR, 'artifacts', 'Ekstra_Bladet_word2vec')

print("Starting loading data...")

# Load datasets
df_train_history = pd.read_parquet(os.path.join(TRAIN_DIR, 'history.parquet'))
df_train_behaviors = pd.read_parquet(os.path.join(TRAIN_DIR, 'behaviors.parquet'))
df_train_articles = pd.read_parquet(os.path.join(ARTICLE_DIR, 'articles.parquet'))
df_valid_history = pd.read_parquet(os.path.join(VALID_DIR, 'history.parquet'))
df_valid_behaviors = pd.read_parquet(os.path.join(VALID_DIR, 'behaviors.parquet'))
df_valid_articles = pd.read_parquet(os.path.join(ARTICLE_DIR, 'articles.parquet'))

# Filter and preprocess datasets
df_train_behaviors = filter_invalid_clicked_articles(df_train_behaviors)
df_valid_behaviors = filter_invalid_clicked_articles(df_valid_behaviors)

print("Loading embeddings...")
embedding_df = pd.read_parquet(EMBEDDING_DIR)

article_to_index = {article_id: idx for idx, article_id in enumerate(embedding_df['article_id'])}

n_trials = 30
n_epochs = 10

# Define Optuna objective
def objective(trial):
    # Suggest hyperparameters
    embedding_dim = trial.suggest_int("embedding_dim", 150, 300, step=15)
    num_heads = trial.suggest_int("num_heads", 4, 132, step=8)  
    attention_dim = trial.suggest_int("attention_dim", 16, 128, step=8)  
    batch_size = trial.suggest_int("batch_size", 4,132, step = 16) 
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-2, log=True)
    max_history_length = trial.suggest_int("max_history_length", 4, 64, step=4)
    max_articles_in_view = trial.suggest_int("max_articles_in_view", 4, 28, step=4)
    popularity_window_hours = trial.suggest_int("popularity_window_hours", 24, 72, step=8)
    top_N_popular_articles = trial.suggest_int("top_N_popular_articles", 4, 24, step=4)

    # Load embeddings
    num_articles = len(article_to_index)
    embedding_matrix = np.zeros((num_articles, embedding_dim))
    for idx, row in embedding_df.iterrows():
        embedding_matrix[article_to_index[row['article_id']]] = row['document_vector'][:embedding_dim]
    print("Embeddings loaded successfully")

    # Initialize WandB for this trial
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "attention_dim": attention_dim,
        "batch_size": batch_size,
        "initial_learning_rate": initial_learning_rate,
        "max_history_length": max_history_length,
        "max_articles_in_view": max_articles_in_view,
        "popularity_window_hours": popularity_window_hours,
        "top_N_popular_articles": top_N_popular_articles,
    })

    # Prepare datasets
    train_dataset, train_user_histories_tensor, user_id_to_index = prepare_data(
        df_train_history, df_train_behaviors, df_train_articles, article_to_index, embedding_matrix,
        max_history_length, popularity_window_hours, top_N_popular_articles, batch_size=batch_size
    )
    validation_dataset, _, _ = prepare_data(
        df_valid_history, df_valid_behaviors, df_valid_articles, article_to_index, embedding_matrix,
        max_history_length, popularity_window_hours, top_N_popular_articles, batch_size=batch_size
    )

    # Build and compile model
    model = NewsRecommendationModel(
        user_histories_tensor=train_user_histories_tensor,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        attention_dim=attention_dim
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.AUC(name="auc")])

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=n_epochs,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5),
            WandbCallback(save_model=False)
        ],
        verbose=1
    )

    # Get best validation AUC
    best_val_auc = max(history.history['val_auc'])

    # Log best validation AUC to WandB
    wandb.log({"best_val_auc": best_val_auc})
    wandb.finish()

    return best_val_auc


# Optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, timeout=360000)

# Log best trial
best_trial = study.best_trial
print("Best Trial:")
print(f"  Value: {best_trial.value}")
print(f"  Params: {best_trial.params}")

wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="best_trial_summary")
wandb.log({"best_trial_value": best_trial.value, "best_trial_params": best_trial.params})
wandb.finish()
