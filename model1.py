import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Membaca data dari file CSV
df = pd.read_csv('data_rating.csv')  # Gantilah 'nama_file.csv' dengan nama file CSV Anda

# Encoding user_id dan product
df['user_id'] = df['user_id'].astype("category").cat.codes
df['product'] = df['product'].astype("category").cat.codes

# Membagi data menjadi data latihan dan data pengujian
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Jumlah unik user dan product
num_users = len(df['user_id'].unique())
num_products = len(df['product'].unique())

# Membuat model collaborative filtering sederhana
user_input = Input(shape=(1,))
product_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 10)(user_input)
product_embedding = Embedding(num_products, 10)(product_input)

user_flat = Flatten()(user_embedding)
product_flat = Flatten()(product_embedding)

dot_product = Dot(axes=1)([user_flat, product_flat])

model = Model(inputs=[user_input, product_input], outputs=dot_product)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Melatih model
model.fit([train_data['user_id'], train_data['product']], train_data['rating'], epochs=100, validation_split=0.2)

# Evaluasi model
result = model.evaluate([test_data['user_id'], test_data['product']], test_data['rating'])
print(f'Loss on test data: {result}')
