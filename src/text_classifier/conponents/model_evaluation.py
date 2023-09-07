from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from text_classifier.entity import ModelEvaluationConfig




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def check_model_performance(self):
        pass


        

