import pandas as pd

# Load separate files
fake = pd.read_csv('dataset/Fake.csv')
real = pd.read_csv('dataset/True.csv')

# Add labels
fake['label'] = 'FAKE'
real['label'] = 'True'

# Combine
df = pd.concat([fake, real], ignore_index=True)

# Optional: ensure column is named 'text'
if 'text' not in df.columns:
    # Adjust this to your column name
    df.rename(columns={df.columns[0]: 'text'}, inplace=True)

# Save combined CSV
df.to_csv('dataset/fake_and_real_news.csv', index=False)
print(" Combined dataset created as fake_and_real_news.csv")
