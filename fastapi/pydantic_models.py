from typing import List

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field


class SegmentationReport(BaseModel):
    filename: str
