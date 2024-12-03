import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import optuna
import tensorflow as tf
import wandb
from utils import *

from optuna.integration import TFKerasPruningCallback

# WandB Project Setup
WANDB_PROJECT = "news-recommendation"
WANDB_ENTITY = "augbirch"  # Replace with your WandB username or team name
wandb.login()

SIZE = "demo"

# Directory Setup
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
TRAIN_DIR = os.path.join(BASE_DIR, SIZE, 'train')
VALID_DIR = os.path.join(BASE_DIR, SIZE, 'validation')
ARTICLE_DIR = os.path.join(BASE_DIR, SIZE)
EMBEDDING_DIR = os.path.join(BASE_DIR, 'artifacts', 'Ekstra_Bladet_word2vec')

# Load the datasets
df_train_history = pd.read_parquet(os.path.join(TRAIN_DIR, 'history.parquet'))
df_train_behaviors = pd.read_parquet(os.path.join(TRAIN_DIR, 'behaviors.parquet'))
df_train_articles = pd.read_parquet(os.path.join(ARTICLE_DIR, 'articles.parquet'))

df_valid_history = pd.read_parquet(os.path.join(VALID_DIR, 'history.parquet'))
df_valid_behaviors = pd.read_parquet(os.path.join(VALID_DIR, 'behaviors.parquet'))
df_valid_articles = pd.read_parquet(os.path.join(ARTICLE_DIR, 'articles.parquet'))

embedding_df = pd.read_parquet(EMBEDDING_DIR)
embedding_dim = len(embedding_df['document_vector'].iloc[0])

article_to_index = {article_id: idx for idx, article_id in enumerate(embedding_df['article_id'])}

num_articles = len(article_to_index)
embedding_matrix = np.zeros((num_articles, embedding_dim))
for idx, row in embedding_df.iterrows():
    embedding_matrix[article_to_index[row['article_id']]] = np.array(row['document_vector'])

epochs = 10
N_trials = 30

def objective(trial):
    # Suggest hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [300])
    num_heads = trial.suggest_int("num_heads", 4, 32, step=4)
    attention_dim = trial.suggest_int("attention_dim", 16, 64, step=16)
    top_N_popular_articles = trial.suggest_int("top_N_popular_articles", 5, 20, step=5)
    max_history_length = trial.suggest_int("max_history_length", 16, 64, step=16)
    popularity_window_hours = trial.suggest_int("popularity_window_hours", 24, 72, step=24)
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-4, 1e-2, log=True)

    # Initialize WandB for this trial
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config={
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "attention_dim": attention_dim,
        "top_N_popular_articles": top_N_popular_articles,
        "max_history_length": max_history_length,
        "popularity_window_hours": popularity_window_hours,
        "initial_learning_rate": initial_learning_rate,
    })

    # Prepare datasets
    train_dataset, train_user_histories_tensor, user_id_to_index = prepare_data(
        df_train_history, df_train_behaviors, df_train_articles,
        article_to_index, embedding_matrix, max_history_length,
        popularity_window_hours, top_N_popular_articles, is_training=True
    )
    validation_dataset, _, _ = prepare_data(
        df_valid_history, df_valid_behaviors, df_valid_articles,
        article_to_index, embedding_matrix, max_history_length,
        popularity_window_hours, top_N_popular_articles, is_training=False
    )

    # Build the model
    model = NewsRecommendationModel(
        user_histories_tensor=train_user_histories_tensor,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        attention_dim=attention_dim
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, clipnorm=5.0)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    auc_metric = tf.keras.metrics.AUC(name="auc")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_auc = 0
        for step, (inputs, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            auc_metric.update_state(labels, predictions)

        train_auc = auc_metric.result().numpy()
        auc_metric.reset_state()

        # Validation Loop
        val_auc = 0
        for val_inputs, val_labels in validation_dataset:
            val_predictions = model(val_inputs, training=False)
            auc_metric.update_state(val_labels, val_predictions)
        val_auc = auc_metric.result().numpy()
        auc_metric.reset_state()

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "loss": loss.numpy(),
        })

        trial.report(val_auc, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.log({"best_val_auc": val_auc})
    wandb.finish()
    return val_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_trials, timeout=3600)

print("Best Trial:")
trial = study.best_trial
wandb.log({"best_trial_value": trial.value, "best_trial_params": trial.params})
print(f"Best Trial Value: {trial.value}")
print("Best Trial Parameters:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")
