import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from convstruct.api import *
from convstruct.api.core import Core
from convstruct.evaluator import Evaluator, initBatch
from convstruct.api.ops import *
from convstruct.api.util import *
