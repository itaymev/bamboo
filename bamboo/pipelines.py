# bamboo/pipelines.py
import pandas as pd
from bamboo.utils import log

from bamboo.bamboo import Bamboo

@log
def save_pipeline(self, filepath):
    """
    Save the cleaning pipeline steps for reuse.
    """
    import joblib
    joblib.dump(self, filepath)

@staticmethod
@log
def load_pipeline(filepath):
    """
    Load a previously saved cleaning pipeline.
    """
    import joblib
    return joblib.load(filepath)
