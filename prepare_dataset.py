import pandas as pd

fake_df = pd.read_csv("data/WELFake_Dataset.csv")
science_df = pd.read_csv("data/real_science_news.csv")

print("WELFake Columns:", fake_df.columns)
print("Science Columns:", science_df.columns)

fake_df = fake_df[['title', 'text', 'label']]
fake_df['text'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')
fake_df['label'] = fake_df['label'].map({0: "FAKE", 1: "REAL"})
fake_df = fake_df[['text', 'label']]

science_df = science_df[['Text']]
science_df = science_df.rename(columns={'Text': 'text'})
science_df['label'] = "REAL"

final_df = pd.concat([fake_df, science_df], ignore_index=True)

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv("data/news.csv", index=False)

print("✅ news.csv created successfully")
print(final_df.head())
print("\nLabel distribution:")
print(final_df['label'].value_counts())
