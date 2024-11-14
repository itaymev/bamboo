import json
import pandas as pd
from bamboo.bamboo import Bamboo
from bamboo.utils import log

class BambooPipeline:
    def __init__(self):
        self.pipeline_steps = []

    def add_step(self, method_name: str, **kwargs):
        """
        Add a step to the pipeline. Each step corresponds to a method of the Bamboo class
        and its associated arguments.

        Parameters:
        - method_name: str
            The name of the method to add to the pipeline.
        - kwargs: dict
            Keyword arguments to be passed to the method.
        """
        self.pipeline_steps.append({
            'method_name': method_name,
            'arguments': kwargs
        })

    def execute_pipeline(self, bamboo: Bamboo) -> Bamboo:
        """
        Execute the pipeline of chained methods on a Bamboo instance.

        Parameters:
        - bamboo: Bamboo
            The Bamboo instance on which the pipeline will be executed.

        Returns:
        - Bamboo: The Bamboo instance with all pipeline steps applied.
        """
        for step in self.pipeline_steps:
            method = getattr(bamboo, step['method_name'])
            method(**step['arguments'])
        return bamboo

    def save_pipeline(self, filepath: str):
        """
        Save the pipeline to a JSON file.

        Parameters:
        - filepath: str
            The path to save the pipeline JSON file.
        """
        with open(filepath, 'w') as file:
            json.dump(self.pipeline_steps, file, indent=4)

    @staticmethod
    def load_pipeline(filepath: str) -> 'BambooPipeline':
        """
        Load a pipeline from a JSON file.

        Parameters:
        - filepath: str
            The path to load the pipeline from.

        Returns:
        - DataCleaningPipeline: An instance of DataCleaningPipeline with the loaded steps.
        """
        pipeline = BambooPipeline()
        with open(filepath, 'r') as file:
            pipeline.pipeline_steps = json.load(file)
        return pipeline
