# Loads all the logs from the above time range
import pandas as pd
bulk = slio.load_a_list_of_logs(logs)
      
temp = bulk.iloc[0].copy()
temp["speed"] = 3.0
temp = pd.DataFrame(temp).T
bulk = temp.append(bulk).reset_index(drop=True)

temp = bulk.iloc[0].copy()
temp["speed"] = 0.5
temp = pd.DataFrame(temp).T
bulk = temp.append(bulk).reset_index(drop=True)
bulk["speed"] = bulk["speed"].values.astype(np.float64)
