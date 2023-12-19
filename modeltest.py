import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv("data_rating3.csv")  # Replace "your_dataset.csv" with the actual filename

# Encode user and product IDs
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

data['user_id'] = user_encoder.fit_transform(data['user_id'])
data['product'] = product_encoder.fit_transform(data['product'])

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Define the model
def create_model(num_users, num_products, embedding_size=50):
    user_input = Input(shape=(1,), name='user_input')
    product_input = Input(shape=(1,), name='product_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
    product_embedding = Embedding(input_dim=num_products, output_dim=embedding_size)(product_input)

    user_flat = Flatten()(user_embedding)
    product_flat = Flatten()(product_embedding)

    concat = Concatenate()([user_flat, product_flat])
    dense1 = Dense(512, activation='relu')(concat)
    dense2 = Dense(256, activation='relu')(dense1)
    output = Dense(1)(dense2)

    model = Model(inputs=[user_input, product_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# Get the number of unique users and products
num_users = data['user_id'].nunique()
num_products = data['product'].nunique()

# Create and train the model
model = create_model(num_users, num_products)
model.fit([train['user_id'], train['product']], train['rating'], epochs=20, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
result = model.evaluate([test['user_id'], test['product']], test['rating'])
print(f"Test Loss: {result}")

model.save("recommendation_model.h5")
