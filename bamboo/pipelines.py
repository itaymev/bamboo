import pandas as pd

class Bamboo:
    """
    A class for handling and cleaning datasets in various formats (Pandas DataFrame, CSV, Excel, JSON).
    Provides pipelines for cleaning operations and data transformations.
    """

    def __init__(self, data):
        """
        Initialize the Bamboo class with a dataset.

        Parameters:
        - data: str or pd.DataFrame
            The data to be cleaned. Can be a file path (CSV, Excel, JSON) or a Pandas DataFrame.
        """
        self.data = self._load_data(data)

    def _load_data(self, data):
        """
        Private method to load data based on the input type.

        Parameters:
        - data: str or pd.DataFrame
            The data to load (CSV, Excel, JSON, or DataFrame).

        Returns:
        - pd.DataFrame
            A Pandas DataFrame containing the loaded data.
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith(('.xls', '.xlsx')):
                return pd.read_excel(data)
            elif data.endswith('.json'):
                return pd.read_json(data)
            else:
                raise ValueError("Unsupported file format!")
        else:
            raise ValueError("Unsupported data type! Must be a Pandas DataFrame or a valid file path.")

    def preview_data(self, rows=5):
        """
        Preview the first few rows of the dataset.

        Parameters:
        - rows: int, optional (default=5)
            The number of rows to display.

        Returns:
        - pd.DataFrame
            A Pandas DataFrame containing the previewed rows.
        """
        return self.data.head(rows)

    def get_data(self):
        """
        Get the current state of the data.

        Returns:
        - pd.DataFrame
            The current state of the dataset.
        """
        return self.data


    def set_data(self, new_data):
        """
        Replace the current dataset with new data.

        Parameters:
        - new_data: pd.DataFrame
            The new data to replace the current dataset.

        Returns:
        - Bamboo: The Bamboo instance with the updated dataset.
        """
        self.save_state()  # Automatically save the current state before making changes
        if isinstance(new_data, pd.DataFrame):
            self.data = new_data
            self.log_changes("Replaced dataset with new data.")
        else:
            raise ValueError("Data must be a Pandas DataFrame.")
        
        return self  # Enables method chaining

        
    def reset_data(self):
        """
        Reset the dataset to its original state.

        Returns:
        - Bamboo: The Bamboo instance with the reset dataset.
        """
        if hasattr(self, '_history') and self._history:
            self.data = self._history[0]  # Revert to the first saved state
            self._history = []  # Clear the history
            self.log_changes("Reset data to its original state.")
        else:
            raise ValueError("No original state to reset to!")
        
        return self  # Enables method chaining


        
    def export_data(self, output_path, format='csv'):
        """
        Export the cleaned data to a specified format (CSV, Excel, JSON).

        Parameters:
        - output_path: str
            The file path where the cleaned data will be saved.
        - format: str, optional (default='csv')
            The format in which to save the data. Choose from 'csv', 'excel', or 'json'.
        
        Returns:
        - Bamboo: The Bamboo instance with the exported data.
        """
        if self.data.empty:
            raise ValueError("Cannot export an empty dataset.")
        
        # Proceed with exporting logic
        if format == 'csv':
            self.data.to_csv(output_path, index=False)
        elif format == 'excel':
            self.data.to_excel(output_path, index=False)
        elif format == 'json':
            self.data.to_json(output_path, orient='records')
        else:
            raise ValueError("Unsupported export format! Choose from 'csv', 'excel', or 'json'.")
        
        return self  # Enables method chaining

    
    def save_state(self):
        """
        Save the current state of the dataset to enable undo functionality.
        """
        if not hasattr(self, '_history'):
            self._history = []
        self._history.append(self.data.copy())

        return self # Enables method chaining

    def undo(self, steps=1):
        """
        Undo the last 'n' cleaning steps and revert to the previous state of the data.

        Parameters:
        - steps: int, optional (default=1)
            The number of steps to undo.

        Returns:
        - pd.DataFrame: The dataset in its previous state.
        """
        if hasattr(self, '_history') and len(self._history) >= steps:
            for _ in range(steps):
                self.data = self._history.pop()
            self.log_changes(f"Reverted {steps} step(s) to previous state of data.")
        else:
            raise ValueError(f"Cannot undo {steps} step(s). Not enough history.")
        
        return self  # Enables method chaining

        
    def save_pipeline(self, filepath):
        """
        Save the cleaning pipeline steps for reuse.

        Parameters:
        - filepath: str
            Path where the pipeline steps will be saved.
        """
        import joblib
        joblib.dump(self, filepath)

    @staticmethod
    def load_pipeline(filepath):
        """
        Load a previously saved cleaning pipeline.

        Parameters:
        - filepath: str
            Path from which the pipeline steps will be loaded.

        Returns:
        - Bamboo: A Bamboo instance with loaded pipeline steps.
        """
        import joblib
        return joblib.load(filepath)

        
    def log_changes(self, message):
        """
        Log changes to the dataset for audit purposes.

        Parameters:
        - message: str
            A message describing the change made to the data.
        """
        if not hasattr(self, '_change_log'):
            self._change_log = []
        
        if not self._change_log or self._change_log[-1] != message:
            self._change_log.append(message)

    def show_change_log(self):
        """
        Display the log of changes made to the dataset during cleaning.

        Returns:
        - str: A formatted log of changes made to the dataset
        """
        if hasattr(self, '_change_log') and self._change_log:
            log = "\n".join([f"{i+1}. {change}" for i, change in enumerate(self._change_log)])
            return log
        else:
            return "No changes logged."
