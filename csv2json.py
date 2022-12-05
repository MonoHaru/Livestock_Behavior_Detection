import pandas as pd
csv_data = pd.read_csv("new_output.csv", sep = ",")
csv_data.to_json("test.json", orient = "records")