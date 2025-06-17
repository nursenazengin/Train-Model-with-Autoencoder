import pandas as pd 

df = pd.read_csv('train.csv')

#fail = df[["Failure"]]

del_col = ["Failure"]

df = df.drop(columns=del_col)


#df['Failure'] = df['Failure'].apply(lambda x: 0 if x == 0.0 else 1)


# print(df['Failure Type'].value_counts())

df.to_csv('predict.csv', index=False)

