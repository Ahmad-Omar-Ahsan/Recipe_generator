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
    x = "Generate a recipe for a vegan pasta dish with tomatoes and basil."
    inputs = tokenizer(x, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Recipe:")
    print(generated_text)
    x_ing = df['ingredients']
    x_step = df['step']

if __name__ == "__main__":
    main()
