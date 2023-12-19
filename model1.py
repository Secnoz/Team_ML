import tensorflow as tf
import tensorflow.datasets as tfds
import tensorflow.recommenders as tfrs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your data
data = pd.read_csv('data_rating.csv')

# Encode the user_id and product columns to create unique integer indices
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

data['user_id'] = user_encoder.fit_transform(data['user_id'])
data['product'] = product_encoder.fit_transform(data['product'])

# Split the data into a train and test set
train, test = train_test_split(data, test_size=0.2)

# Convert the pandas dataframe to a tensorflow dataset
train = tf.data.Dataset.from_tensor_slices(dict(train))
test = tf.data.Dataset.from_tensor_slices(dict(test))

# Build flexible representation models.
user_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=unique_user_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])
product_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=unique_product_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_product_titles) + 1, embedding_dimension)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    train.batch(128).map(product_model)
  )
)

# Create a retrieval model.
model = tfrs.Model(user_model, product_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train.
model.fit(train.batch(4096), validation_data=test.batch(4096), epochs=3)

# Set up retrieval using trained representations.
index = tfrs.layers.ann.BruteForce(model.user_model)
index.index_from_dataset(
  tf.data.Dataset.zip((train.map(lambda x: x["product"]), train.batch(100).map(model.product_model)))
)

# Get recommendations.
_, titles = index(np.array([42]))
print(f"Recommendations for user 42: {titles[0, :3]}")