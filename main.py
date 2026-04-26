import pandas as pd 
import numpy as np
import transformers

def main():
    print("Hello from recipe-generator!")
    df = pd.read_csv("dataset/Multi_Cuisine_Recipe_Dataset.csv")
    df.head()

if __name__ == "__main__":
    main()
