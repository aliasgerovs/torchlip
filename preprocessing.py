import pandas as pd

def preprocessor(df):
    df = pd.read_csv("captions.txt")
    df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
    df.to_csv("captions.csv", index=False)
    df = pd.read_csv("captions.csv")
    image_path = "/content/Images"
    captions_path = "/content"
    return df