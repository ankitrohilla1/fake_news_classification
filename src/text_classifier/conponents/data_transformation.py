import os
from text_classifier.logging import logger
from transformers import BertTokenizer
import tensorflow as tf
from text_classifier.entity import DataTransformationConfig
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

    def transform_data(self):
        fake_df = pd.read_csv(os.path.join(self.config.data_path, 'Fake.csv'), encoding='UTF-8')
        true_df = pd.read_csv(os.path.join(self.config.data_path, 'True.csv'), encoding='UTF-8')
        fake_df['label'] = 0
        true_df['label'] = 1
        df = pd.concat([fake_df, true_df]).reset_index()
        df.drop_duplicates(inplace=True)
        df['text'] = df['text'] + " " + df['title']
        df.drop(columns=['title', 'subject', 'date'])
        df = df.sample(frac=1).reset_index(drop=True)
        X_train, X_test, Y_train, Y_test = train_test_split(df['text'], df['label'], stratify = df['label'], test_size = 0.25, random_state =42)
        return (X_train, X_test, Y_train, Y_test)
    
    def convert_examples_to_features(self, X):  
        X = self.tokenizer(
            text = list(X),
            add_special_tokens = True,
            max_length = 120,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True
            )
        return X

    def convert(self):
        X_train, X_test, Y_train, Y_test = self.transform_data()
        train_encoding = self.convert_examples_to_features(X_train)
        test_encoding = self.convert_examples_to_features(X_test)
        train_encodings = {
            'X_train': train_encoding,
            'y_train': Y_train
        }
        test_encodings = {
            'X_test': test_encoding,
            'y_test': Y_test
        }

        # Save the dictionary to a file
        with open(os.path.join(self.config.root_dir,"train_encodings.pkl"), 'wb') as f:
            pickle.dump(train_encodings, f)
        with open(os.path.join(self.config.root_dir,"test_encodings.pkl"), 'wb') as f:
            pickle.dump(test_encodings, f)


