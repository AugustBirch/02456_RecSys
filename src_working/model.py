# Import required libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import *


# Enable dynamic memory growth for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

SIZE = "demo"

# Get the directory of the current script and move one level up
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'data')
TRAIN_DIR = os.path.join(BASE_DIR, SIZE, 'train')
VALID_DIR = os.path.join(BASE_DIR, SIZE, 'validation')
ARTICLE_DIR = os.path.join(BASE_DIR, SIZE)

TEST_ARTICLES_DIR = os.path.join(BASE_DIR, 'ebnerd_testset')
TEST_DIR = os.path.join(BASE_DIR, 'ebnerd_testset', 'test')

EMBEDDING_DIR = os.path.join(BASE_DIR, 'artifacts', 'Ekstra_Bladet_word2vec')
# /work3/s204129/DL2024/data/artifacts/Ekstra_Bladet_word2vec/document_vector.parquet

print("Starting loading data...")

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

# Ensure consistent data types for merging in train dataset
df_train_behaviors['article_id'] = df_train_behaviors['article_id'].fillna('-1').astype(str)
df_train_articles['article_id'] = df_train_articles['article_id'].astype(str)

# Ensure consistent data types for merging in validation dataset
df_valid_behaviors['article_id'] = df_valid_behaviors['article_id'].fillna('-1').astype(str)
df_valid_articles['article_id'] = df_valid_articles['article_id'].astype(str)

# Ensure consistent data types for merging in test dataset
# df_test_behaviors['article_id'] = df_test_behaviors['article_id'].fillna('-1').astype(str)
df_test_articles['article_id'] = df_test_articles['article_id'].astype(str)

# Filter invalid rows from behaviors dataset
df_train_behaviors = filter_invalid_clicked_articles(df_train_behaviors)
df_valid_behaviors = filter_invalid_clicked_articles(df_valid_behaviors)
# df_test_behaviors = filter_invalid_clicked_articles(df_test_behaviors)

print("Data loaded successfully")

# DEFINE ALL THE HYPERPARAMETERS
embedding_dim = 300             # NOT USED.  Dimension of the article embedding vectors            
num_heads = 16                  # Number of attention heads in the attention layer
attention_dim = 32              # Dimension of the attention space
batch_size = 64                 # NOT USED.  Number of samples used in each training iteration
epochs_num = 10                 # Number of times the model will iterate over the entire training dataset
initial_learning_rate=0.001     # Initial value of learning rate (learning rate is dynamically set by the scheduler)
max_history_length = 32         # Maximum length of user history considered by the model
max_articles_in_view = 10       # Maximum number of articles in a user's viewing session (if applicable)
popularity_window_hours = 48    # Number of hours to consider for popularity calculation
top_N_popular_articles = 32     # Number of top popular articles to consider

# EMBEDDINGS OF ARTICLES
print("Loading embeddings...")
# Import the embedding fle provided by the competition organizers
embedding_df = pd.read_parquet(EMBEDDING_DIR)

# # Check the embedding vectors dimension
# embedding_dim = len(embedding_df['document_vector'].iloc[0])

# Mapping article_id -> embedding index
article_to_index = {article_id: idx for idx, article_id in enumerate(embedding_df['article_id'])}

# Initialisation of embedding matrix
num_articles = len(article_to_index)
embedding_matrix = np.zeros((num_articles, embedding_dim))

# Puopulate the embedding matrix
for idx, row in embedding_df.iterrows():
    embedding_matrix[article_to_index[row['article_id']]] = np.array(row['document_vector'])
print("Embeddings loaded successfully")

train_dataset, train_user_histories_tensor, user_id_to_index = prepare_data(df_train_history, df_train_behaviors, 
                                                                            df_train_articles, article_to_index, 
                                                                            embedding_matrix, max_history_length, 
                                                                            popularity_window_hours, top_N_popular_articles, 
                                                                            is_training=True)

validation_dataset, _, _ = prepare_data(df_valid_history, df_valid_behaviors, 
                                        df_valid_articles, article_to_index, embedding_matrix, max_history_length, 
                                        popularity_window_hours, top_N_popular_articles, is_training=False)

# Create a model instance
model = NewsRecommendationModel(
    user_histories_tensor=train_user_histories_tensor,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    attention_dim=attention_dim
)

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Create a model instance
model = NewsRecommendationModel(
                                user_histories_tensor=train_user_histories_tensor,
                                embedding_dim=embedding_dim,
                                num_heads=num_heads,
                                attention_dim=attention_dim
                              )

# Define the loss function
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Define the scheduler (to dynamically set the optimal learning rate)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9)

# Create optimizer using above scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Define the callback (reduces the learning rate when the validation loss stops to decrease)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

# Compile the model
model.compile(optimizer=optimizer,  # Use the optimizer instance
              loss=loss_fn,
              metrics=[tf.keras.metrics.AUC(name="auc")])

# Train the model
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs_num,
    callbacks=[reduce_lr]
)
