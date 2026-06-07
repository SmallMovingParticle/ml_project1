import sys 
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer ##use to do pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import dill


@dataclass
class DataTransformationConfig:
    PreProcessor_obj_file_path= os.path.join('artifacts', "prepocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            numerical_columns =["writing_score", "reading_score"]

            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course",]
        
            num_pipeline = Pipeline(
            steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scalar", StandardScaler())
        ]
        )

            cat_pipeline=Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_ecoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ]


        )
            logging.info("categorical columns done")
            logging.info("numerical columns scaling done")
            
            
            
            preprocessor= ColumnTransformer([
                ("num_pipeline" , num_pipeline , numerical_columns),
                ("cat_pipeline", cat_pipeline , categorical_columns)]
            )


            return preprocessor




        
        except CustomException as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation (self , train_path , test_path):

        try:
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading of train and test data is complete")
            logging.info("obtaining processing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name= "math_score"
            numerical_column = ["writing_score" , "reading_score"]

            # 1. Separating features and target for Training Data (Already correct)
            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis=1)
            target_feature_train_df = train_df[target_column_name] # <- Create this to use in train_arr below

            # 2. FIX: Properly separate features and target for Testing Data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1) # <- ADD THIS LINE
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessing object on training dataframe and testing dataframe")

            # 3. Transform the feature dataframes
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) # <- FIX: Pass input_feature_test_df here

            # 4. FIX: Recombine arrays using the correct respective targets
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df) # <- FIX: Combine train data with train targets
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)   # <- Combine test data with test targets
            ]

            logging.info("saved the preprocessing object")

            save_object(
                file_path = self.data_tranformation_config.PreProcessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.PreProcessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            
