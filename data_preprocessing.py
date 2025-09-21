import pandas as pd

input_path = "F:/DSIT/1st_Semester/Deep_Neural_Networks/archive/Dataset.csv"
output_path = "F:/DSIT/1st_Semester/Deep_Neural_Networks/archive/Dataset_preprocessed.csv"

df = pd.read_csv(input_path)
df = df.fillna(0)
df.to_csv(output_path, index=False)

print("Saved: ", output_path)
