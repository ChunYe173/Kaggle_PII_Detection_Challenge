import os
import json
import pandas as pd

 
if __name__ == "__main__":
    csv_file = r'C:\Users\abome\source\repos\venv_pii_ner\dataset\wnut17train_cleaned.csv' # append path to other dataset json files to this list
    output_filename = "wnut17train_cleaned.json"

    df = pd.read_csv(csv_file)
    df['labels'] = df['labels'].apply(lambda x: eval(x))
    print(df.head(3))

    new_json_string = df.to_json(orient='records')
    type(new_json_string) # string
    new_json_file = json.loads(new_json_string)
    type(new_json_file) # list
    type(new_json_file[0]) # dict
    print(new_json_file[0])

    for i in range(len(new_json_file)):
      new_json_file[i]['tokens'] = eval(new_json_file[i]['tokens'])
      new_json_file[i]['trailing_whitespace'] = eval(new_json_file[i]['trailing_whitespace'])
      new_json_file[i]['labels'] = eval(new_json_file[i]['labels'])

    with open(output_filename, "w") as outfile:
       json.dump(new_json_file, outfile)

    print("Generated new json file.")
