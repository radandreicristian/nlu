import os
import pathlib
from typing import Any

import pandas as pd

from util.result.abstract_result_writer import AbstractResultWriter
from util.tools import get_timestamp


class LocalResultWriter(AbstractResultWriter):

    def __init__(self, splits: int, output_path: str):
        self.columns = [''] + list([str(i) for i in range(splits)]) + ['mean', 'dev']
        self.df = pd.DataFrame(columns=self.columns)
        if not os.path.isdir(output_path):
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        self.output_path = os.path.join(output_path, "quantitative_analysis_" + get_timestamp() + ".csv")

    def get_destination(self) -> Any:
        return self.df

    def append_row(self, values: list, index: int) -> None:
        row = pd.Series(dict(zip(self.columns, values)), name=index)
        self.df.append(row)

    def save(self) -> None:
        self.df.to_csv(self.output_path, encoding="utf-8")
