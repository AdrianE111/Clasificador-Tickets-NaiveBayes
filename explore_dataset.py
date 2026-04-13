from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')

# Get the training split
df = pd.DataFrame(dataset['train'])

print('Dataset shape:', df.shape)
print('Columns:', df.columns.tolist())
print()
print('Sample rows:')
print(df.head())
print()
print('Unique categories:')
print(df['category'].value_counts())
print()
print('Total categories:', df['category'].nunique())