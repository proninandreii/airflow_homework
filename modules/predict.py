import glob
import json
import os
import pandas as pd
import dill

def predict():
    model_files = glob.glob('data/models/*.pkl')
    if not model_files:
        print("No model files found in 'data/models/'")
        return

    latest_model_file = max(model_files, key=os.path.getctime)

    with open(latest_model_file, 'rb') as model_file:
        model = dill.load(model_file)

    test_data_folder = "data/test"
    predictions = []
    for file in os.listdir(test_data_folder):
        if file.endswith(".json"):

            with open(os.path.join(test_data_folder, file), "rb") as json_file:
                test_data = json.load(json_file)

            test_data = pd.DataFrame(test_data, index=[0])
            result = model.predict(test_data)
            predictions.extend(result)

    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
    predictions_df.to_csv("data/predictions/predictions.csv", index=False)

if __name__ == '__main__':
    predict()
