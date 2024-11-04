import os
import logging
import tarfile
import re
from typing import Any, Dict, List
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
import pandas as pd
from google.cloud import bigquery
import joblib
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

## This file creates a custom predictor from our model -- in this case, a random forest model
## that predicts the number of chargers for a given FSA and time range. The model is loaded in,
## and the predict method is used to make predictions.

class RFStationPredictor(Predictor):
    def __init__(self) -> None:
        pass

    def load(self, artifacts_uri: str) -> None:
        """Loads the preprocessor and model artifacts."""
        logger.info(f"Downloading artifacts from {artifacts_uri}")
        prediction_utils.download_model_artifacts(artifacts_uri)
        logger.info("Artifacts successfully downloaded!")

        os.makedirs("./model", exist_ok=True)
        with tarfile.open("model.tar.gz", "r:gz") as tar:
            tar.extractall(path="./model")

        logger.info("Model successfully loaded!")

    def predict(self, instances: Dict[str, List[str]]) -> Dict[str, Any]:

        # check if the input data is valid
        instances = instances['instances'][0]
        fsas = instances["FSA"]
        time_start = instances["time_start"]
        time_end = instances["time_end"]
        
        assert len(fsas) != 0, "fsa is empty"
        assert all(isinstance(item, str) for item in fsas), "fsa is not a list of strings"
        assert all(re.match('[A-Za-z][0-9]+[A-Za-z]', item) for item in fsas), "fsa is not in the correct format"
        assert len(time_start) != 0, "time_start is empty"
        assert len(time_end) != 0, "time_end is empty"
        assert len(time_start) == 1, "only one time start is allowed"
        assert len(time_end) == 1, "only one time end is allowed"
        assert all(re.match('[A-Za-z]+\s\d\d\d\d', item) for item in time_start), "time_start is not in the correct format (Month YYYY)"
        assert all(re.match('[A-Za-z]+\s\d\d\d\d', item) for item in time_end), "time_end is not in the correct format (Month YYYY)"

        # check if the time range is valid
        datetime_start = pd.to_datetime(time_start[0])
        datetime_end = pd.to_datetime(time_end[0])
        assert datetime_start < datetime_end, "time_start is not before time_end"

        # create a dataframe with the time range and FSAs
        date_range = pd.date_range(start=datetime_start, end=datetime_end, freq='MS')
        month_year_range = pd.DataFrame({'year': date_range.year, 'month': date_range.month})

        input_df = pd.DataFrame([(fsa, year, month) for fsa in fsas for year, month in zip(month_year_range['year'], month_year_range['month'])], 
                            columns=['FSA', 'year', 'month'])
                
        ## encode FSA
        encoder = joblib.load('./model_code/fsa_encoder.joblib')
        input_df['FSA_encoded'] = encoder.transform(input_df['FSA'])
        

        output_df = input_df.copy()

        # reformat columns
        input_df['Year_relative'] = input_df['year'] - 2018
        input_df['Year_sqrt'] = input_df['year'].apply(lambda x: np.sqrt(x))
        input_df['Year_squared'] = input_df['year'].apply(lambda x: x**2)
        input_df.drop('year', axis=1, inplace=True)
        input_df['Month_sin'] = np.sin(2 * np.pi * input_df['month']/12)
        input_df['Month_cos'] = np.cos(2 * np.pi * input_df['month']/12)

        input_df['Quarter'] = input_df['month'] // 3 + 1
        input_df['Quarter_sin'] = np.sin(2 * np.pi * input_df['Quarter']/4)
        input_df['Quarter_cos'] = np.cos(2 * np.pi * input_df['Quarter']/4)

        input_df.drop('Quarter', axis=1, inplace=True)
        input_df.drop('month', axis=1, inplace=True)
        
        # read features from the feature store
        client = bigquery.Client()
        fsa_list = ', '.join([f"'{fsa}'" for fsa in fsas])
        query = f"""
        SELECT * FROM `kiwi-external-pilot.features_rf_charger_pred_v1.features`
        WHERE FSA IN ({fsa_list})
        """
        features_df = client.query_and_wait(query).to_dataframe()
        features_df = features_df.drop("int64_field_0", axis=1)
        features_df = features_df.drop_duplicates()
                

        # merge features with input_df
        merged_input_df = input_df.merge(features_df, on=['FSA'], how='inner').drop_duplicates()
        merged_input_df.drop('FSA', axis=1, inplace=True)        

        order =  [
             'FSA_encoded',
         'urban',
         'avg_household_size',
         'avg_income',
         'total_priv_dwelling_households',
            'Year_sqrt', 'Year_squared',
         'total_house_households',
         'total_condos',
         'total_apartment_households',
         'frac_employed',
         'frac_unemployed',
         'population',
         'private_dwellings',
         'Month_sin', 
        'Month_cos', 
        'Quarter_sin', 
        'Quarter_cos']
        merged_input_df = merged_input_df[order]
        
        

        # make predictions
        loaded_model = joblib.load('./model_code/station_density_regressor_v1.joblib')
        pred_df = loaded_model.predict(merged_input_df)
        
        output_df['predictions'] = pred_df
        # decode FSA, return predictions
        output_df['FSA'] = encoder.inverse_transform(output_df['FSA_encoded'])
        output_df = output_df.drop(columns=['FSA_encoded'])

        # reformat dates and create dictionary of predictions
        output_df['date'] = pd.to_datetime(output_df['year'].astype(str) + '-' + output_df['month'].astype(str).str.zfill(2) + '-01')
        
        fsa_pred_dict = output_df.groupby('FSA').apply(
           lambda x: dict(zip(x['date'].dt.strftime('%Y-%m'), x['predictions'].astype(int).transform(lambda x: np.maximum.accumulate(x))))
        ).to_dict()
        
        return fsa_pred_dict
