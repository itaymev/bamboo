# bamboo/bamboo.py
import pandas as pd
from bamboo.utils import log

class Bamboo:
    """
    A class for handling and cleaning datasets in various formats (Pandas DataFrame, CSV, Excel, JSON).
    Provides pipelines for cleaning operations and data transformations.
    """

    @log
    def __init__(self, data):
        """
        Initialize the Bamboo class with a dataset.
        """
        self.data = self._load_data(data)

    @log
    def _load_data(self, data):
        """
        Private method to load data based on the input type.
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            try:
                if data.endswith('.csv'):
                    return pd.read_csv(data)
                elif data.endswith(('.xls', '.xlsx')):
                    return pd.read_excel(data)
                elif data.endswith('.json'):
                    return pd.read_json(data)
                else:
                    raise ValueError("Unsupported file format!")
            except Exception as e:
                raise ValueError(f"Failed to load file: {str(e)}")
        else:
            raise ValueError("Unsupported data type! Must be a Pandas DataFrame or a valid file path.")

    @log
    def preview_data(self, rows=5):
        """
        Preview the first few rows of the dataset.
        """
        return self.data.head(rows)

    @log
    def get_data(self):
        """
        Get the current state of the data.
        """
        return self.data

    @log
    def set_data(self, new_data):
        """
        Replace the current dataset with new data.
        """
        self.save_state()
        if isinstance(new_data, pd.DataFrame):
            self.data = new_data
            self.log_changes("Replaced dataset with new data.")
        else:
            raise ValueError("Data must be a Pandas DataFrame.")
        return self

    @log
    def reset_data(self):
        """
        Reset the dataset to its original state.
        """
        if hasattr(self, '_history') and self._history:
            self.data = self._history[0]
            self._history = []
            self.log_changes("Reset data to its original state.")
        else:
            raise ValueError("No original state to reset to!")
        return self

    @log
    def export_data(self, output_path, format='csv'):
        """
        Export the cleaned data to a specified format (CSV, Excel, JSON).
        """
        if self.data.empty:
            raise ValueError("Cannot export an empty dataset.")

        if format == 'csv':
            self.data.to_csv(output_path, index=False)
        elif format == 'excel':
            self.data.to_excel(output_path, index=False)
        elif format == 'json':
            self.data.to_json(output_path, orient='records')
        else:
            raise ValueError("Unsupported export format! Choose from 'csv', 'excel', or 'json'.")
        return self

    @log
    def save_state(self):
        """
        Save the current state of the dataset to enable undo functionality.
        """
        if not hasattr(self, '_history'):
            self._history = []
        self._history.append(self.data.copy())
        return self

    @log
    def undo(self, steps=1):
        """
        Undo the last 'n' cleaning steps and revert to the previous state of the data.
        """
        if hasattr(self, '_history') and len(self._history) >= steps:
            for _ in range(steps):
                self.data = self._history.pop()
            self.log_changes(f"Reverted {steps} step(s) to previous state of data.")
        else:
            raise ValueError(f"Cannot undo {steps} step(s). Not enough history.")
        return self
    
    @log
    def log_changes(self, message):
        """
        Log changes to the dataset for audit purposes.
        """
        if not hasattr(self, '_change_log'):
            self._change_log = []
        if not self._change_log or self._change_log[-1] != message:
            self._change_log.append(message)

    @log
    def show_change_log(self):
        """
        Display the log of changes made to the dataset during cleaning.
        """
        if hasattr(self, '_change_log') and self._change_log:
            log = "\n".join([f"{i+1}. {change}" for i, change in enumerate(self._change_log)])
        else:
            log = "No changes logged."
        return log