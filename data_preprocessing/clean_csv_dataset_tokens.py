import os
import json
import pandas as pd

 
if __name__ == "__main__":
    csv_file = input("Please input filename of csv to process (i.e. filename.csv): ") 
    print(csv_file)
    output_filename = input("Please input filename of output csv file (i.e. filename.csv): ")
    print(output_filename)

    # csv_file = 'train.csv'
    # output_filename = "train_cleaned.csv"

    print('Loading csv.')
    df = pd.read_csv(csv_file)
    for col in ['tokens','trailing_whitespace','labels']:
        df[col] = df[col].apply(lambda x: eval(x))
    print('Sample data.')
    print(df.head(3))

    # Populate list with specific characters to remove from string in token
    char_to_remove = [
    '''   
        
      ''',
    '''   
     ''',
     '​',' ','â€œ‹','Â­â€','­','™','Â','â','€','œ','‹']

    # Cleaning all tokens. Omitted from cleaning: full_text. Can edit code to clean full_text too. 
    # Loop through every single element in all token list
    for i in df.index:
        print("Processing row: ", i)
        for j in range(len(df['tokens'].iloc[i])):
            for char in char_to_remove:
                if char in df['tokens'].iloc[i][j]:
                    df['tokens'].iloc[i][j] = df['tokens'].iloc[i][j].replace(char,'')

    df.to_csv(output_filename, index=False)

    print("Generated cleaned csv file.")

"""
#trash strings: ​,  , â€œ‹, \n\n, ' ', '    ', '   ', \n,­,™,Â­â€,Â­â€,

    
    
   ,

  ,
 Â,

    
   ,

Document 1339
NAME
Ruben Pabon,
Georgia,
Bethany,
"""