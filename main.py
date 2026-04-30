import pandas as pd 
import numpy as np
import transformers

def main():
    print("Hello from recipe-generator!")
    df = pd.read_csv("dataset/Multi_Cuisine_Recipe_Dataset.csv")
    df.head()
    print(df.columns)
    print(df.describe())
    print(df.isnull().sum())
    print("Loading pre-trained model...")
    model = transformers.AutoModelForCausalLM.from_pretrained("model")
    tokenizer = transformers.AutoTokenizer.from_pretrained("model")
    print("Model loaded successfully!")

if __name__ == "__main__":
    main()
