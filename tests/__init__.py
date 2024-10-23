# tests/__init__.py

# Import all test modules to ensure test discovery works
import unittest
from .test_pipelines import *
from .test_imputation import *
from .test_outliers import *
from .test_dtype_validation import *
from .test_duplicates import *
from .test_formatting import *
from .test_categorical import *
from .test_validation import *
from .test_dates import *
from .test_profiling import *
