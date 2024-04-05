import base64
import datetime
import numpy as np
import json
import requests
import pandas as pd
import gzip
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import shap

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

    def edit_data(self, show_nat_value):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)
        variable = config['visualization']['variable']

        # Filter the DataFrame to keep only the specified variables
        filtered_dataframe = self.data[self.data['variable'].isin([variable])]
        #filtered_dataframe = filtered_dataframe[filtered_dataframe['site_id'].isin(["Vitality"])]
        filtered_dataframe['Value'] = filtered_dataframe['Value'].astype(float).dropna()

        filtered_dataframe_site = filtered_dataframe[filtered_dataframe['site_id'].isin(["Vitality"])]

        if filtered_dataframe['ATTRIBUTE_TYPE'].iloc[0] == 'Quantitative' or filtered_dataframe['ATTRIBUTE_TYPE'].iloc[0] == 'Categorical':
            aggregated_dataframe = filtered_dataframe_site.groupby(['YQ', 'site_id'])['Value'].median().reset_index()
        else:
            aggregated_dataframe = filtered_dataframe_site.groupby(['YQ', 'site_id'])['Value'].mean().reset_index()

        if show_nat_value:
            filtered_dataframe_country = filtered_dataframe[~filtered_dataframe['site_id'].isin(["Vitality"])]

            if filtered_dataframe['ATTRIBUTE_TYPE'].iloc[0] == 'Quantitative' or filtered_dataframe['ATTRIBUTE_TYPE'].iloc[0] == 'Categorical':
                nat_df = filtered_dataframe_country.groupby(['YQ'])['Value'].median().reset_index()
            else:
                nat_df = filtered_dataframe_country.groupby(['YQ'])['Value'].mean().reset_index()
            nat_df = nat_df[['YQ', 'Value']]

            aggregated_dataframe = pd.merge(aggregated_dataframe, nat_df, on='YQ')
            aggregated_dataframe.rename(columns={'Value_y': 'nat_value'}, inplace=True)
            aggregated_dataframe.rename(columns={'Value_x': 'Value'}, inplace=True)

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


        # Flag to indicate if we did not have data from 2022 Q2
        no_2022_q2_data = '2022 Q2' not in variable_data['YQ'].values

        return p_value, cohens_d, no_2022_q2_data

    def find_predictors(self, local):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)

        data = self.data

        # Specify the target variable you want to predict
        target_variable = config['visualization']['variable']

        # Extract all unique predictor variable names from the "variable" column
        predictor_variables = data['variable'].unique()

        # Remove the target variable from the predictor variables list
        predictor_variables = [var for var in predictor_variables if var != target_variable]

        # Define the order of categories in the "TAB" column
        category_order = ['PC', 'Bleeding', 'Imaging', 'Treatment', 'PO', 'Discharge']

        # Map category names to their positions in the order of occurrence
        category_positions = {category: index for index, category in enumerate(category_order)}

        # Get the category of the target variable
        target_category = data.loc[data['variable'] == target_variable, 'TAB'].iloc[0]

        if local:
            data = data[data['site_id'].isin(["Vitality"])]
        # Filter predictor variables based on the category order
        predictor_variables_filtered = []
        predictor_variables_filtered_names = []
        for var in predictor_variables:
            var_category = data.loc[data['variable'] == var, 'TAB'].iloc[0]
            if category_positions[var_category] < category_positions[target_category]:
                predictor_variables_filtered.append(var)
                predictor_variables_filtered_names.append(data.loc[data['variable'] == var, 'INDICATOR'].iloc[0])

        # If the target variable is from the "PC" category, print a message and exit
        if target_category == 'PC':
            error = "The target variable is a patient characteristic, it cannot be predicted."
            return error, None

        # Convert non-numeric values in 'Value' column to NaN
        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')

        # Pivot the data to wide format based on the 'variable' column
        data_wide = data.pivot_table(index=['YQ', 'subject_id'], columns='variable', values='Value').reset_index()

        # Fill missing values with 0 if needed
        data_wide.fillna(0, inplace=True)

        # Convert non-numeric variables to numeric if needed
        for var in data_wide.columns:
            if var != 'YQ' and var not in predictor_variables_filtered:
                try:
                    if not pd.api.types.is_numeric_dtype(data_wide[var]):
                        data_wide[var] = pd.to_numeric(data_wide[var], errors='coerce')
                except KeyError:
                    print(f"Skipping variable {var} due to KeyError")

        filtered_indices = [i for i, var in enumerate(predictor_variables_filtered) if var in data_wide.columns]
        predictor_variables_filtered = [predictor_variables_filtered[i] for i in filtered_indices]

        # Remove corresponding indices from predictor_variables_filtered_names
        predictor_variables_filtered_names = [predictor_variables_filtered_names[i] for i in filtered_indices]

        # Split the data into training and testing sets
        X = data_wide[predictor_variables_filtered]
        y = data_wide[target_variable]

        # In long format, we don't need to split the data; each record is independent
        # So, we can directly use all data for training
        X_train, X_test, y_train, y_test = X, X, y, y  # Just for consistency
        # Build the Gradient Boosting Regressor model
        gbr = GradientBoostingRegressor()
        gbr.fit(X_train, y_train)

        # Predict on the testing set (we're using the same data for training and testing in this case)
        y_pred = gbr.predict(X_test)

        # Calculate accuracy (RMSE in this case)
        accuracy = mean_squared_error(y_test, y_pred, squared=False)
        print(f'Root Mean Squared Error: {accuracy}')

        feature_importances = gbr.feature_importances_
        # Sort feature importances
        sorted_indices = feature_importances.argsort()[::-1]
        # Print top 10 important features
        feature_weights = {}
        for i in sorted_indices[:5]:
            feature_weights[predictor_variables_filtered[i]] = feature_importances[i]

        explainer = shap.TreeExplainer(gbr)

        shap_values = explainer.shap_values(X_test)

        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Get indices of features sorted by importance
        sorted_indices = np.argsort(mean_shap_values)[::-1][:10]

        shap_values = {}
        for i in sorted_indices[:10]:
            shap_values[predictor_variables_filtered_names[i]] = mean_shap_values[i]

        # Return error (if any) and feature weights
        return None, {'Root Mean Squared Error': accuracy, 'Feature Importances': feature_weights, 'Shap Values': shap_values}

    def DNT_guideline(self):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)

        data = self.data
        show_nat = config['visualization']['show_nat_val']

        df_filtered = data[data['site_id'].isin(["Vitality"])]

        # Filter the DataFrame to keep only 'door_to_needle' variable
        filtered_df = df_filtered[df_filtered['variable'] == 'door_to_needle'].copy()

        filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['Value'])

        # Calculate the percentage of patients with door to needle time below 60 minutes in each quarter
        filtered_df['door_to_needle_below_60'] = filtered_df['Value'] < 45
        percentage_below_60 = filtered_df.groupby('YQ')['door_to_needle_below_60'].mean() * 100

        percentage_below_60_df = percentage_below_60.reset_index()
        percentage_below_60_df.columns = ['YQ', 'Value']

        if show_nat == True:
            df_filtered_nat = data[~data['site_id'].isin(["Vitality"])]
            filtered_df_nat = df_filtered_nat[df_filtered_nat['variable'] == 'door_to_needle'].copy()

            filtered_df_nat['Value'] = pd.to_numeric(filtered_df_nat['Value'], errors='coerce')
            filtered_df_nat = filtered_df_nat.dropna(subset=['Value'])

            # Calculate the percentage of patients with door to needle time below 60 minutes in each quarter
            filtered_df_nat['door_to_needle_below_60'] = filtered_df_nat['Value'] < 45
            percentage_below_60_nat = filtered_df_nat.groupby('YQ')['door_to_needle_below_60'].mean() * 100

            percentage_below_60_df_nat = percentage_below_60_nat.reset_index()
            percentage_below_60_df_nat.columns = ['YQ', 'Value']
            percentage_below_60_df['nat_value'] = percentage_below_60_df_nat['Value']
            # print(percentage_below_60_nat

        json_data = percentage_below_60_df.to_json(orient='records')

        with open(self.json_data_path, 'w') as json_file:
            json_file.write(json_data)

        with open(self.json_data_path, 'r') as json_file:
            config = json.load(json_file)

        compressed_content = gzip.compress(json.dumps(config).encode("utf-8"))
        compressed_content_decoded = base64.b64encode(compressed_content).decode("utf-8")

        payload = {"file_type": "data", "file_content": compressed_content_decoded}

        response = requests.post(self.website_url, json=payload)

        print(response.status_code)

        return response.json()

    def dysphagia_guideline(self):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)

        data = self.data
        show_nat = config['visualization']['show_nat_val']

        df_filtered = data[data['site_id'].isin(["Vitality"])]
        # Filter the DataFrame to keep only 'dysphagia_screening_type' variable
        filtered_df = df_filtered[df_filtered['variable'] == 'dysphagia_screening_type'].copy()

        # Count non-null values for each quarter
        non_null_counts = filtered_df.groupby('YQ')['Value'].count()

        # Count total number of patients for each quarter
        total_patients = df_filtered.groupby('YQ')['subject_id'].nunique()

        # Calculate percentage of patients with non-null values for each quarter
        percentage_filled_in = (non_null_counts / total_patients) * 100

        # Convert percentage_filled_in Series to a DataFrame
        percentage_filled_in_df = percentage_filled_in.reset_index()
        percentage_filled_in_df.columns = ['YQ', 'Value']

        if show_nat == True:
            df_filtered_nat = data[~data['site_id'].isin(["Vitality"])]
            # Filter the DataFrame to keep only 'dysphagia_screening_type' variable
            filtered_df_nat = df_filtered_nat[df_filtered_nat['variable'] == 'dysphagia_screening_type'].copy()

            # Count non-null values for each quarter
            non_null_counts_nat = filtered_df_nat.groupby('YQ')['Value'].count()

            # Count total number of patients for each quarter
            total_patients_nat = df_filtered_nat.groupby('YQ')['subject_id'].nunique()

            # Calculate percentage of patients with non-null values for each quarter
            percentage_filled_in_nat = (non_null_counts_nat / total_patients_nat) * 100

            # Convert percentage_filled_in Series to a DataFrame
            percentage_filled_in_df_nat = percentage_filled_in_nat.reset_index()
            percentage_filled_in_df_nat.columns = ['YQ', 'Value']
            percentage_filled_in_df['nat_value'] = percentage_filled_in_df_nat['Value']

        json_data = percentage_filled_in_df.to_json(orient='records')

        with open(self.json_data_path, 'w') as json_file:
            json_file.write(json_data)

        with open(self.json_data_path, 'r') as json_file:
            config = json.load(json_file)

        compressed_content = gzip.compress(json.dumps(config).encode("utf-8"))
        compressed_content_decoded = base64.b64encode(compressed_content).decode("utf-8")

        payload = {"file_type": "data", "file_content": compressed_content_decoded}

        response = requests.post(self.website_url, json=payload)

        print(response.status_code)

        return response.json()

    def guideline_anticoags(self):
        with open(self.json_file_path, 'r') as json_file:
            config = json.load(json_file)

        data = self.data
        show_nat = config['visualization']['show_nat_val']

        df_filtered = data[data['site_id'].isin(["Vitality"])]

        # Filter the DataFrame to keep only the variables of interest
        filtered_df = df_filtered[
            df_filtered['variable'].isin(['discharge_warfarin', 'discharge_heparin', 'discharge_dabigatran',
                                          'discharge_rivaroxaban', 'discharge_apixaban', 'discharge_edoxaban',
                                          'discharge_cilostazol',
                                          'discharge_clopidrogel',
                                          'discharge_ticagrelor',
                                          'discharge_ticlopidine',
                                          'discharge_prasugrel',
                                          'discharge_dipyridamol',
                                          ])].copy()

        # Convert 'Value' column to numeric
        filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')

        # Replace NaN values with 0
        filtered_df['Value'].fillna(0, inplace=True)

        # Mark patients with 1 in any of the variables
        filtered_df['any_discharge_medication'] = (filtered_df['Value'] == 1).astype(int)

        # Count patients with 1 in any discharge medication for each quarter
        count_any_discharge_medication = filtered_df.groupby('YQ')['any_discharge_medication'].sum()

        # Count total number of patients for each quarter
        total_patients = df_filtered.groupby('YQ')['subject_id'].nunique()

        # Calculate percentage of patients with 1 in any discharge medication for each quarter
        percentage_any_discharge_medication = (count_any_discharge_medication / total_patients) * 100

        # Convert percentage_any_discharge_medication Series to a DataFrame
        percentage_any_discharge_medication_df = percentage_any_discharge_medication.reset_index()
        percentage_any_discharge_medication_df.columns = ['YQ', 'Value']

        if show_nat == True:
            df_filtered_nat = data[~data['site_id'].isin(["Vitality"])]

            # Filter the DataFrame to keep only the variables of interest
            filtered_df_nat = df_filtered_nat[
                df_filtered_nat['variable'].isin(['discharge_warfarin', 'discharge_heparin', 'discharge_dabigatran',
                                                  'discharge_rivaroxaban', 'discharge_apixaban', 'discharge_edoxaban',
                                                  'discharge_cilostazol',
                                                  'discharge_clopidrogel',
                                                  'discharge_ticagrelor',
                                                  'discharge_ticlopidine',
                                                  'discharge_prasugrel',
                                                  'discharge_dipyridamol',
                                                  ])].copy()

            # Convert 'Value' column to numeric
            filtered_df_nat['Value'] = pd.to_numeric(filtered_df_nat['Value'], errors='coerce')

            # Replace NaN values with 0
            filtered_df_nat['Value'].fillna(0, inplace=True)

            # Mark patients with 1 in any of the variables
            filtered_df_nat['any_discharge_medication'] = (filtered_df_nat['Value'] == 1).astype(int)

            # Count patients with 1 in any discharge medication for each quarter
            count_any_discharge_medication_nat = filtered_df_nat.groupby('YQ')['any_discharge_medication'].sum()

            # Count total number of patients for each quarter
            total_patients_nat = df_filtered_nat.groupby('YQ')['subject_id'].nunique()

            # Calculate percentage of patients with 1 in any discharge medication for each quarter
            percentage_any_discharge_medication_nat = (count_any_discharge_medication_nat / total_patients_nat) * 100

            # Convert percentage_any_discharge_medication Series to a DataFrame
            percentage_any_discharge_medication_df_nat = percentage_any_discharge_medication_nat.reset_index()
            percentage_any_discharge_medication_df_nat.columns = ['YQ', 'Value']
            percentage_any_discharge_medication_df['nat_value'] = percentage_any_discharge_medication_df_nat['Value']

            json_data = percentage_any_discharge_medication_df.to_json(orient='records')

            with open(self.json_data_path, 'w') as json_file:
                json_file.write(json_data)

            with open(self.json_data_path, 'r') as json_file:
                config = json.load(json_file)

            compressed_content = gzip.compress(json.dumps(config).encode("utf-8"))
            compressed_content_decoded = base64.b64encode(compressed_content).decode("utf-8")

            payload = {"file_type": "data", "file_content": compressed_content_decoded}

            response = requests.post(self.website_url, json=payload)

            print(response.status_code)

            return response.json()