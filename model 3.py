import pandas as pd
import numpy as np
import openpyxl
# Data user_id yang Anda miliki
# Data user_id yang Anda miliki
user_ids_existing = [
    48732, 47639, 69501, 80854, 15839, 37475, 49937, 35055, 53084, 21021,
    92812, 93646, 2211, 88076, 44513, 43155, 33008, 59306, 24201, 96251,
    70377, 29864, 11230, 1534, 64817, 18492, 78474, 87237, 82489, 76338,
    48553, 8963, 25821, 17423, 34604, 92534, 86691, 37555, 9000, 66093,
    1811, 30056, 70246, 62441, 64232, 54726, 9591, 62019, 47725, 28517
]
# Buat data dummy dengan user_id yang sudah ada
np.random.seed(42)  # Untuk hasil yang konsisten
user_ids = np.repeat(user_ids_existing, 1000)
products = np.random.choice(['ayam', 'itik', 'kerbau', 'sapi', 'kuda', 'unggas lainnya', 'kelinci', 'domba', 'kambing'], size=len(user_ids))
ratings = np.random.randint(2, 6, size=len(user_ids))

# Buat DataFrame
df = pd.DataFrame({'user_id': user_ids, 'product': products, 'rating': ratings})

# Tampilkan DataFrame
print(df)
df.to_csv('data_rating3.csv', index=False)
