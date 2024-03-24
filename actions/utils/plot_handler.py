import base64

import matplotlib.figure
import matplotlib.pyplot as plt
import datetime
import seaborn
import pickle
import numpy as np
from actions.utils import globals
from sklearn import linear_model, ensemble
import json
import requests
import pandas as pd
import gzip
from scipy.stats import ttest_ind, t, wilcoxon

class PlotHandler:
    def __init__(self, save_plot=False):
        self._date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._plot_name = None
        self._save = save_plot
        self.json_file_path = "actions/utils/plot_args.json"
        self.json_data_path = "actions/utils/data.json"
        #self.website_url = "http://localhost:3000/rasa-webhook"
        self.website_url = "https://dashboards.create.aau.dk/rasa-webhook"
        self.data = pd.read_csv("actions/utils/dataREanonymized_long.csv")

    def change_arg(self, arg, value):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)

        config['visualization'][arg] = value

        with open(self.json_file_path, 'w') as json_file:
            json.dump(config, json_file, indent=2)
            print(json.dumps(config, indent=2))

    def send_args(self):
        with open(self.json_file_path, 'r') as file:
            json_data = json.load(file)

        compressed_content = gzip.compress(json.dumps(json_data).encode("utf-8"))
        compressed_content_decoded = base64.b64encode(compressed_content).decode("utf-8")

        payload = {"file_type": "args", "file_content": compressed_content_decoded}

        response = requests.post(self.website_url, json=payload)
        return response

    def edit_data(self):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)
        variable = config['visualization']['variable']

        # Filter the DataFrame to keep only the specified variables
        filtered_dataframe = self.data[self.data['variable'].isin([variable])]
        filtered_dataframe = filtered_dataframe[filtered_dataframe['site_id'].isin(["Vitality"])]

        #if filtered_dataframe['ATTRIBUTE_TYPE'].iloc[0] == 'Quantitative' or filtered_dataframe['ATTRIBUTE_TYPE'].iloc[0] == 'Categorical_binary':
        filtered_dataframe['Value'] = filtered_dataframe['Value'].astype(float).dropna()

        aggregated_dataframe = filtered_dataframe.groupby(['YQ', 'site_id'])['Value'].median().reset_index()

        json_data = aggregated_dataframe.to_json(orient='records')

        with open(self.json_data_path, 'w') as json_file:
            json_file.write(json_data)

        with open(self.json_data_path, 'r') as json_file:
            config = json.load(json_file)

        compressed_content = gzip.compress(json.dumps(config).encode("utf-8"))
        compressed_content_decoded = base64.b64encode(compressed_content).decode("utf-8")

        payload = {"file_type": "data", "file_content": compressed_content_decoded}

        response = requests.post(self.website_url, json=payload)

        return response


    def compare_to_past(self):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)
        var = config['visualization']['variable']

        variable_data = self.data[self.data['variable'] == var]

        variable_data['value'] = pd.to_numeric(variable_data['Value'], errors='coerce')

        variable_data = variable_data.dropna(subset=['value'])

        # Check if there is data for "2022 Q2", if not, compare "2022 Q1" to "2021 Q2"
        if '2022 Q2' in variable_data['YQ'].values:
            q2_data = variable_data[variable_data['YQ'] == '2022 Q2']['Value']
            q1_data = variable_data[variable_data['YQ'] == '2022 Q1']['Value']
        else:
            q2_data = variable_data[variable_data['YQ'] == '2022 Q1']['Value']
            q1_data = variable_data[variable_data['YQ'] == '2021 Q4']['Value']

        q2_data = pd.to_numeric(q2_data)
        q1_data = pd.to_numeric(q1_data)

        min_len = min(len(q1_data), len(q2_data))
        if len(q1_data) > len(q2_data):
            q1_data = q1_data.sample(n=min_len, random_state=42)
        elif len(q2_data) > len(q1_data):
            q2_data = q2_data.sample(n=min_len, random_state=42)

        # Perform Wilcoxon signed-rank test
        _, p_value = wilcoxon(q2_data, q1_data)

        # Calculate Cohen's d
        mean_diff = q2_data.mean() - q1_data.mean()
        pooled_std = np.sqrt((q1_data.var() + q2_data.var()) / 2)
        cohens_d = mean_diff / pooled_std

        print("Group 2 mean: ", q2_data.mean())
        print("Group 1 mean: ", q1_data.mean())
        print("Group 2 std: ", q2_data.std())
        print("Group 1 std: ", q1_data.std())


        # Flag to indicate if we did not have data from 2022 Q2
        no_2022_q2_data = '2022 Q2' not in variable_data['YQ'].values

        return p_value, cohens_d, no_2022_q2_data