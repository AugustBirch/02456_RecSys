import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import optuna
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import *

from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping

SIZE = "demo"

# Get the directory of the current script and move one level up
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
TRAIN_DIR = os.path.join(BASE_DIR, SIZE, 'train')
VALID_DIR = os.path.join(BASE_DIR, SIZE, 'validation')
ARTICLE_DIR = os.path.join(BASE_DIR, SIZE)

TEST_ARTICLES_DIR = os.path.join(BASE_DIR, 'ebnerd_testset')
TEST_DIR = os.path.join(BASE_DIR, 'ebnerd_testset', 'test')

EMBEDDING_DIR = os.path.join(BASE_DIR, 'artifacts', 'Ekstra_Bladet_word2vec')

LOG_FILE = "optuna_training_log.txt"

# Clear previous log file content
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log_to_file(message):
    with open(LOG_FILE, "a") as file:
        file.write(message + "\n")

print("Starting loading data...")
log_to_file("Starting loading data...")

# Load the datasets
df_train_history = pd.read_parquet(os.path.join(TRAIN_DIR, 'history.parquet'))
df_train_behaviors = pd.read_parquet(os.path.join(TRAIN_DIR, 'behaviors.parquet'))
df_train_articles = pd.read_parquet(os.path.join(ARTICLE_DIR, 'articles.parquet'))

df_valid_history = pd.read_parquet(os.path.join(VALID_DIR, 'history.parquet'))
df_valid_behaviors = pd.read_parquet(os.path.join(VALID_DIR, 'behaviors.parquet'))
df_valid_articles = pd.read_parquet(os.path.join(ARTICLE_DIR, 'articles.parquet'))

df_test_history = pd.read_parquet(os.path.join(TEST_DIR, 'history.parquet'))
df_test_behaviors = pd.read_parquet(os.path.join(TEST_DIR, 'behaviors.parquet'))
df_test_articles = pd.read_parquet(os.path.join(TEST_ARTICLES_DIR, 'articles.parquet'))

print("Data loaded successfully")
log_to_file("Data loaded successfully")

# EMBEDDINGS OF ARTICLES
print("Loading embeddings...")
log_to_file("Loading embeddings...")

embedding_df = pd.read_parquet(EMBEDDING_DIR)
embedding_dim = len(embedding_df['document_vector'].iloc[0])

article_to_index = {article_id: idx for idx, article_id in enumerate(embedding_df['article_id'])}

num_articles = len(article_to_index)
embedding_matrix = np.zeros((num_articles, embedding_dim))
for idx, row in embedding_df.iterrows():
    embedding_matrix[article_to_index[row['article_id']]] = np.array(row['document_vector'])
print("Embeddings loaded successfully")
log_to_file("Embeddings loaded successfully")

max_history_length = 32
popularity_window_hours = 48
top_N_popular_articles = 10
epochs = 10
N_trials = 30

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

def objective(trial):
    embedding_dim = trial.suggest_categorical("embedding_dim", [300])
    num_heads = trial.suggest_int("num_heads", 4, 32, step=4)
    attention_dim = trial.suggest_int("attention_dim", 16, 64, step=16)
    dropout_rate1 = trial.suggest_float("dropout_rate1", 0.1, 0.5)
    dropout_rate2 = trial.suggest_float("dropout_rate2", 0.1, 0.5)
    initial_learning_rate = trial.suggest_loguniform("initial_learning_rate", 1e-4, 1e-2)

    log_to_file(f"Starting trial with parameters: embedding_dim={embedding_dim}, num_heads={num_heads}, "
                f"attention_dim={attention_dim}, dropout_rate1={dropout_rate1}, "
                f"dropout_rate2={dropout_rate2}, initial_learning_rate={initial_learning_rate}")

    model = NewsRecommendationModel(
        user_histories_tensor=train_user_histories_tensor,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        attention_dim=attention_dim,
        dropout_rate1=dropout_rate1,
        dropout_rate2=dropout_rate2
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    auc_metric = tf.keras.metrics.AUC(name="auc")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        log_to_file(f"Epoch {epoch + 1}/{epochs}")
        for step, (inputs, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(
                    inputs,
                    dropout_rate1=dropout_rate1,
                    dropout_rate2=dropout_rate2,
                    training=True
                )
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            auc_metric.update_state(labels, predictions)

        train_auc = auc_metric.result().numpy()
        log_to_file(f"Training AUC after epoch {epoch + 1}: {train_auc:.4f}")
        auc_metric.reset_state()

        for val_inputs, val_labels in validation_dataset:
            val_predictions = model(
                val_inputs,
                dropout_rate1=dropout_rate1,
                dropout_rate2=dropout_rate2,
                training=False
            )
            auc_metric.update_state(val_labels, val_predictions)

        val_auc = auc_metric.result().numpy()
        log_to_file(f"Validation AUC after epoch {epoch + 1}: {val_auc:.4f}")
        auc_metric.reset_state()

        trial.report(val_auc, epoch)
        if trial.should_prune():
            log_to_file("Trial pruned")
            raise optuna.exceptions.TrialPruned()

    return val_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_trials, timeout=3600)

print("Best Trial:")
trial = study.best_trial
log_to_file(f"Best Trial Value: {trial.value}")
log_to_file("Best Trial Parameters:")
for key, value in trial.params.items():
    log_to_file(f"  {key}: {value}")
