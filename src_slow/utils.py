# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LayerNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC

import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm

# Function to transform the history of the user into tensor containing the articles embeddings
def process_user_history(df_history, article_to_index, embedding_matrix, max_history_length):

    # Initialize a list to store padded embeddings
    article_embeddings = []

    for article_ids in df_history['article_id_fixed']:
        # Collect embeddings for valid article IDs
        embeddings = [embedding_matrix[article_to_index[article_id]]
                      for article_id in article_ids if article_id in article_to_index]

        # Pad or truncate to the fixed history length
        if len(embeddings) > max_history_length:
            embeddings = embeddings[:max_history_length]
        elif len(embeddings) < max_history_length:
            embeddings += [np.zeros(embedding_matrix.shape[1])] * (max_history_length - len(embeddings))

        article_embeddings.append(embeddings)

    # Convert to a NumPy array and ensure correct dtype
    padded_array = np.array(article_embeddings, dtype=np.float32)

    # Map user to index
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(df_history['user_id'].unique())}

    return tf.convert_to_tensor(padded_array), user_id_to_index

def compute_article_age(df_behaviors, df_articles):
    # Explode 'article_ids_inview' to have one article per row
    df_behaviors_exploded = df_behaviors.explode('article_ids_inview')

    # Merge with articles to get 'published_time', specify suffixes to avoid column name conflicts
    df_merged = df_behaviors_exploded.merge(
        df_articles[['article_id', 'published_time']],
        left_on='article_ids_inview',
        right_on='article_id',
        how='left',
        suffixes=('', '_article')  # Specifying suffixes
    )

    # Convert timestamps to datetime
    df_merged['impression_time'] = pd.to_datetime(df_merged['impression_time'])
    df_merged['published_time'] = pd.to_datetime(df_merged['published_time'])

    # Compute article age in hours
    df_merged['article_age'] = (df_merged['impression_time'] - df_merged['published_time']).dt.total_seconds() / 3600.0
    df_merged['article_age'] = df_merged['article_age'].fillna(0)

    # Handle missing 'article_ids_clicked' in test set
    if 'article_ids_clicked' in df_merged.columns:
        agg_dict = {
            'user_id': 'first',
            'article_id': 'first',  # From df_behaviors_exploded
            'session_id': 'first',
            'article_ids_inview': list,
            'article_ids_clicked': 'first',
            'article_age': list,
            'impression_time': 'first',
            'hour_of_day': 'first',
            'day_of_week': 'first',
        }
    else:
        agg_dict = {
            'user_id': 'first',
            'article_id': 'first',  # From df_behaviors_exploded
            'session_id': 'first',
            'article_ids_inview': list,
            'article_age': list,
            'impression_time': 'first',
            'hour_of_day': 'first',
            'day_of_week': 'first',
        }

    df_grouped = df_merged.groupby('impression_id').agg(agg_dict).reset_index()

    return df_grouped

def compute_session_time_features(df_behaviors):
    df_behaviors['impression_time'] = pd.to_datetime(df_behaviors['impression_time'])
    df_behaviors['hour_of_day'] = df_behaviors['impression_time'].dt.hour
    df_behaviors['day_of_week'] = df_behaviors['impression_time'].dt.weekday  # 0=Monday, 6=Sunday
    return df_behaviors

def compute_user_activity_features(df_behaviors):
    df_behaviors = df_behaviors.sort_values(['user_id', 'impression_time'])
    df_behaviors['time_since_last_session'] = df_behaviors.groupby('user_id')['impression_time'].diff().dt.total_seconds().fillna(0)
    return df_behaviors

def pad_or_truncate_list(lst, target_length, padding_value):
    lst = list(lst)
    if len(lst) > target_length:
        return lst[:target_length]
    else:
        return lst + [padding_value] * (target_length - len(lst))
    
def build_in_session_histories(df_behaviors):
    # Sort by user_id, session_id, and impression_time
    df_behaviors = df_behaviors.sort_values(['user_id', 'session_id', 'impression_time'])

    # Initialize a dictionary to store in-session histories
    in_session_histories = {}

    # Group by session
    grouped = df_behaviors.groupby(['user_id', 'session_id'])

    # Iterate over each session
    for (user_id, session_id), group in grouped:
        viewed_articles = []
        for idx, row in group.iterrows():
            # Store the current viewed articles
            in_session_histories[idx] = list(viewed_articles)

            # Update the viewed_articles list with the current article
            article_id = row['article_id']
            if article_id is not None:
                viewed_articles.append(article_id)
    return in_session_histories

def generate_session_labels(df_behaviors, article_to_index, embedding_matrix, max_articles_in_view=10, max_in_session_history=5, max_popularity_articles=10):
    session_data = []

    for _, row in df_behaviors.iterrows():
        user_id = row['user_id']
        impression_id = row['impression_id']
        articles_in_view = np.array(row['article_ids_inview'])
        articles_clicked = set(row['article_ids_clicked']) if row['article_ids_clicked'] is not None else set()
        in_session_history = row['in_session_history']

        # Generate embeddings for in-session history
        in_session_embeddings = [
            embedding_matrix[article_to_index.get(article_id, 0)]
            for article_id in in_session_history
        ]

        # Pad or truncate to max_in_session_history
        in_session_embeddings = pad_or_truncate_list(in_session_embeddings, max_in_session_history, np.zeros(embedding_matrix.shape[1]))

        # Popularity articles embeddings
        popularity_articles = row['popularity_articles']
        popularity_embeddings = [
            embedding_matrix[article_to_index.get(article_id, 0)]
            for article_id in popularity_articles
        ]
        popularity_embeddings = pad_or_truncate_list(popularity_embeddings, max_popularity_articles, np.zeros(embedding_matrix.shape[1]))

        # Existing code for article embeddings and labels
        embeddings = [
            embedding_matrix[article_to_index.get(article_id, 0)]
            for article_id in articles_in_view
        ]
        embeddings = pad_or_truncate_list(embeddings, max_articles_in_view, np.zeros(embedding_matrix.shape[1]))

        labels = np.isin(articles_in_view, list(articles_clicked)).astype(int)
        labels = pad_or_truncate_list(labels, max_articles_in_view, 0)

        session_data.append({
            'user_id': user_id,
            'impression_id': impression_id,
            'article_embeddings': embeddings,
            'in_session_embeddings': in_session_embeddings,
            'popularity_embeddings': popularity_embeddings,
            'labels': labels,
            # Include other features as needed
        })

    return pd.DataFrame(session_data)

def prepare_test_sessions(df_behaviors, article_to_index, embedding_matrix, max_articles_in_view=10, max_in_session_history=5, max_popularity_articles=10):
    session_data = []

    for _, row in df_behaviors.iterrows():
        user_id = row['user_id']
        impression_id = row['impression_id']
        articles_in_view = np.array(row['article_ids_inview'])

        # In-session history embeddings
        in_session_history = row.get('in_session_history', [])
        in_session_embeddings = [
            embedding_matrix[article_to_index.get(article_id, 0)]
            for article_id in in_session_history
        ]
        in_session_embeddings = pad_or_truncate_list(in_session_embeddings, max_in_session_history, np.zeros(embedding_matrix.shape[1]))

        # Popularity articles embeddings
        popularity_articles = row.get('popularity_articles', [])
        popularity_embeddings = [
            embedding_matrix[article_to_index.get(article_id, 0)]
            for article_id in popularity_articles
        ]
        popularity_embeddings = pad_or_truncate_list(popularity_embeddings, max_popularity_articles, np.zeros(embedding_matrix.shape[1]))

        # Article embeddings
        embeddings = [
            embedding_matrix[article_to_index.get(article_id, 0)]
            for article_id in articles_in_view
        ]
        embeddings = pad_or_truncate_list(embeddings, max_articles_in_view, np.zeros(embedding_matrix.shape[1]))

        session_data.append({
            'user_id': user_id,
            'impression_id': impression_id,
            'article_embeddings': embeddings,
            'in_session_embeddings': in_session_embeddings,
            'popularity_embeddings': popularity_embeddings,
            # No labels since we don't have 'article_ids_clicked'
        })

    return pd.DataFrame(session_data)

def compute_popularity_features(df_behaviors, popularity_window_hours, top_n=10):
    """
    Adds a 'popularity_articles' column to df_behaviors, which lists the top N articles
    popular in the last 'popularity_window_hours' before each impression.
    """
    # Convert 'impression_time' to datetime
    df_behaviors['impression_time'] = pd.to_datetime(df_behaviors['impression_time'])

    # Create a DataFrame 'df_views' containing each viewed article with the corresponding time
    df_views = df_behaviors[['impression_time', 'article_id']].dropna(subset=['article_id'])

    # Set 'impression_time' as the index and sort it
    df_views.set_index('impression_time', inplace=True)
    df_views.sort_index(inplace=True)

    # Initialize a cache dictionary to store popularity for each time period
    popularity_cache = {}

    # Time window in Timedelta
    time_window = pd.Timedelta(hours=popularity_window_hours)

    # Iterate through unique impression times
    impression_times = df_behaviors['impression_time'].unique()

    for time in impression_times:
        if time in popularity_cache:
            continue
        # Define the time window
        start_time = time - time_window
        # Ensure that start_time and time are Timestamps
        start_time = pd.to_datetime(start_time)
        time = pd.to_datetime(time)
        # Filter views in the time window
        views_in_window = df_views.loc[start_time:time]
        # Count article views
        article_counts = views_in_window['article_id'].value_counts()
        # Get the top N most popular articles
        popular_articles = article_counts.head(top_n).index.tolist()
        # Store results in the cache
        popularity_cache[time] = popular_articles

    # Map popularity to df_behaviors
    df_behaviors['popularity_articles'] = df_behaviors['impression_time'].map(popularity_cache)

    # Replace NaN with an empty list if there are no popular articles
    df_behaviors['popularity_articles'] = df_behaviors['popularity_articles'].apply(lambda x: x if isinstance(x, list) else [])

    return df_behaviors

def create_tf_dataset(df_labeled_sessions, user_id_to_index, batch_size):
    user_indices = df_labeled_sessions['user_id'].map(user_id_to_index).fillna(0).astype(int).to_numpy()
    article_embeddings = np.stack(df_labeled_sessions['article_embeddings'].to_numpy())
    in_session_embeddings = np.stack(df_labeled_sessions['in_session_embeddings'].to_numpy())
    popularity_embeddings = np.stack(df_labeled_sessions['popularity_embeddings'].to_numpy())
    labels = np.stack(df_labeled_sessions['labels'].to_numpy())

    """
    # Convert to TensorFlow tensors and ensure they are on the CPU
    with tf.device('/CPU:0'):  # Force tensors to reside on the CPU
        user_indices = tf.convert_to_tensor(user_indices, dtype=tf.int32)
        article_embeddings = tf.convert_to_tensor(article_embeddings, dtype=tf.float32)
        in_session_embeddings = tf.convert_to_tensor(in_session_embeddings, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32) # Assuming labels are integers
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        ((user_indices, article_embeddings, in_session_embeddings, popularity_embeddings), labels)
    ).batch(batch_size)

    return dataset

def create_tf_dataset_for_prediction(df_sessions, user_id_to_index, batch_size):
    user_indices = df_sessions['user_id'].map(user_id_to_index).astype(int).to_numpy()
    article_embeddings = np.stack(df_sessions['article_embeddings'].to_numpy())
    in_session_embeddings = np.stack(df_sessions['in_session_embeddings'].to_numpy())
    popularity_embeddings = np.stack(df_sessions['popularity_embeddings'].to_numpy())

    dataset = tf.data.Dataset.from_tensor_slices((
        (user_indices, article_embeddings, in_session_embeddings, popularity_embeddings)
    ))

    return dataset.batch(batch_size)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LayerNormalization, Dropout
from tensorflow.keras.regularizers import l2

class UserEncoder(Model):
    def __init__(self, embedding_dim, num_heads, attention_dim, dropout_rate=0.2, **kwargs):
        super(UserEncoder, self).__init__(**kwargs)

        # Self-attention layer
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layer_norm1 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)

        # Additive attention layer
        self.additive_attention_dense = layers.Dense(embedding_dim, activation='tanh')
        self.layer_norm2 = layers.LayerNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)

        # Extra dense layer
        self.attention_score_dense = layers.Dense(1)
        self.softmax = layers.Softmax(axis=1)

    def call(self, inputs):

        # Self-attention layer
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.layer_norm1(attention_output)
        attention_output = self.dropout1(attention_output)

        # Additive attention layer
        additive_attention_output = self.additive_attention_dense(attention_output)
        additive_attention_output = self.layer_norm2(additive_attention_output)
        additive_attention_output = self.dropout2(additive_attention_output)

        # Dense layer
        attention_scores = self.attention_score_dense(additive_attention_output)
        attention_weights = self.softmax(attention_scores)

        weighted_output = tf.reduce_sum(attention_output * attention_weights, axis=1)
        return weighted_output
    
class InSessionEncoder(Model):
    def __init__(self, embedding_dim, num_heads, **kwargs):
        super(InSessionEncoder, self).__init__(**kwargs)
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.2)
        self.output_layer = layers.Dense(embedding_dim)

    def call(self, inputs):
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.layer_norm(attention_output)
        attention_output = self.dropout(attention_output)
        # Aggregate the outputs
        in_session_representation = tf.reduce_mean(attention_output, axis=1)
        in_session_representation = self.output_layer(in_session_representation)
        return in_session_representation
    
class PopularityEncoder(Model):
    def __init__(self, embedding_dim, num_heads, **kwargs):
        super(PopularityEncoder, self).__init__(**kwargs)
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.2)
        self.output_layer = layers.Dense(embedding_dim)

    def call(self, inputs):
        # inputs: tensor with dimensions: (batch_size, max_popularity_articles, embedding_dim)
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.layer_norm(attention_output)
        attention_output = self.dropout(attention_output)
        # Aggregation of outputs
        popularity_representation = tf.reduce_mean(attention_output, axis=1)
        popularity_representation = self.output_layer(popularity_representation)
        return popularity_representation
    
class ClickPredictor(Model):
    def __init__(self, input_dim, dropout_rate1, dropout_rate2, **kwargs):
        super(ClickPredictor, self).__init__(**kwargs)
        self.dense1 = layers.Dense(input_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout1 = layers.Dropout(rate=dropout_rate1)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(rate=dropout_rate2)
        self.dense3 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, dropout_rate1=None, dropout_rate2=None, training=None):
        x = self.dense1(inputs)
        if training and dropout_rate1 is not None:
            x = tf.nn.dropout(x, rate=dropout_rate1)
        x = self.dense2(x)
        if training and dropout_rate2 is not None:
            x = tf.nn.dropout(x, rate=dropout_rate2)
        click_probability = self.dense3(x)
        return click_probability





    
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import AUC

class NewsRecommendationModel(Model):
    def __init__(self, user_histories_tensor, embedding_dim, num_heads, attention_dim, dropout_rate1, dropout_rate2, **kwargs):
        super(NewsRecommendationModel, self).__init__(**kwargs)
        self.user_histories_tensor = user_histories_tensor
        self.user_encoder = UserEncoder(embedding_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim)
        self.in_session_encoder = InSessionEncoder(embedding_dim=embedding_dim, num_heads=num_heads)
        self.popularity_encoder = PopularityEncoder(embedding_dim=embedding_dim, num_heads=num_heads)

        # Self-attention layer
        self.self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.layer_norm = LayerNormalization()
        self.dropout_layer = Dropout(0.0)  # Default to no dropout, handled manually
        self.embedding_dim = embedding_dim

        # Dense layers for projection
        self.user_projection = layers.Dense(embedding_dim, activation='relu')
        self.article_projection = layers.Dense(embedding_dim, activation="relu")

        # Click predictor
        self.click_predictor = ClickPredictor(input_dim=embedding_dim, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)


    def call(self, inputs, dropout_rate1=0.2, dropout_rate2=0.2, training=None):
        user_indices, article_embeddings, in_session_embeddings, popularity_embeddings = inputs

        # Project article embeddings
        article_embeddings = self.article_projection(article_embeddings)

        # User representation
        user_histories = tf.gather(self.user_histories_tensor, user_indices)
        user_representation = self.user_encoder(user_histories)

        # In-session representation
        in_session_representation = self.in_session_encoder(in_session_embeddings)

        # Popularity representation
        popularity_representation = self.popularity_encoder(popularity_embeddings)

        # Combine user representations
        combined_user_representation = tf.concat(
            [user_representation, in_session_representation, popularity_representation], axis=-1
        )

        # Project to embedding_dim
        combined_user_representation = self.user_projection(combined_user_representation)

        # Prepare data for attention
        batch_size = tf.shape(article_embeddings)[0]
        num_articles = tf.shape(article_embeddings)[1]

        # Expand dimensions and concatenate with articles
        user_representation_expanded = tf.expand_dims(combined_user_representation, axis=1)
        sequence = tf.concat([user_representation_expanded, article_embeddings], axis=1)

        # Apply self-attention
        attention_output = self.self_attention(sequence, sequence)
        if training:
            attention_output = tf.nn.dropout(attention_output, rate=dropout_rate1)
        attention_output = self.layer_norm(sequence + attention_output)

        # Extract article representations after attention (skipping the first element, which is the user representation)
        article_attention_output = attention_output[:, 1:, :]

        # Flatten and predict click probabilities
        article_flat = tf.reshape(article_attention_output, [-1, self.embedding_dim])
        click_probabilities_flat = self.click_predictor(
            article_flat, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2, training=training
        )
        click_probabilities = tf.reshape(click_probabilities_flat, [batch_size, num_articles])

        return click_probabilities



def prepare_data(df_history, df_behaviors, df_articles, article_to_index, embedding_matrix, max_history_length, popularity_window_hours, top_N_popular_articles, is_training=True):

    # Compute temporal features
    df_behaviors = compute_session_time_features(df_behaviors)
    df_behaviors = compute_user_activity_features(df_behaviors)
    df_behaviors = compute_article_age(df_behaviors, df_articles)

    # Compute and add in-session history
    in_session_histories = build_in_session_histories(df_behaviors)
    df_behaviors['in_session_history'] = df_behaviors.index.map(in_session_histories)

    # Compute popularity features
    df_behaviors = compute_popularity_features(df_behaviors, popularity_window_hours, top_N_popular_articles)

    # Prepare user histories
    user_histories_tensor, user_id_to_index = process_user_history(
        df_history, article_to_index, embedding_matrix, max_history_length
    )

    # Generate session labels including temporal features
    df_labeled_sessions = generate_session_labels(df_behaviors, article_to_index, embedding_matrix)

    # Create dataset including temporal features
    dataset = create_tf_dataset(df_labeled_sessions, user_id_to_index, batch_size=32)

    return dataset, user_histories_tensor, user_id_to_index

def train_with_gradient_monitoring(model, dataset, validation_data, optimizer, loss_fn, epochs, log_interval=1):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = []
        gradient_stats = []

        for step, (inputs, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(labels, predictions)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Analyze gradients
            gradient_magnitudes = [tf.reduce_max(tf.abs(grad)).numpy() if grad is not None else 0 for grad in gradients]
            gradient_stats.append({
                'mean': np.mean(gradient_magnitudes),
                'max': np.max(gradient_magnitudes),
                'min': np.min(gradient_magnitudes),
                'std': np.std(gradient_magnitudes),
            })

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss.append(loss.numpy())

            # Log every `log_interval` steps
            if step % log_interval == 0:
                print(f"Step {step}, Loss: {loss.numpy():.4f}, Gradient Mean: {gradient_stats[-1]['mean']:.4e}")

        # Epoch summary
        avg_loss = np.mean(epoch_loss)
        avg_gradients = {
            'mean': np.mean([g['mean'] for g in gradient_stats]),
            'max': np.mean([g['max'] for g in gradient_stats]),
            'min': np.mean([g['min'] for g in gradient_stats]),
            'std': np.mean([g['std'] for g in gradient_stats]),
        }
        print(f"Epoch {epoch + 1} Summary: Loss = {avg_loss:.4f}, Gradient Stats: {avg_gradients}")

        # Validation AUC
        val_auc = tf.keras.metrics.AUC(name='val_auc')
        for val_inputs, val_labels in validation_data:
            val_predictions = model(val_inputs, training=False)
            val_auc.update_state(val_labels, val_predictions)
        print(f"Validation AUC: {val_auc.result().numpy():.4f}")

