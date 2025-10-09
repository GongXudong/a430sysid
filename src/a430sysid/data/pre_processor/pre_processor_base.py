from abc import ABC, abstractmethod

import pandas as pd


# 插值、滤波、增加新列等预处理基类
class PreProcessorBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class PreProcessorList(PreProcessorBase):
    def __init__(self):
        self.pre_processor_list: list[PreProcessorBase] = []

    def add(self, pre_processor: PreProcessorBase):
        self.pre_processor_list.append(pre_processor)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for pre_processor in self.pre_processor_list:
            df = pre_processor.process(df)

        return df

    def __str__(self):
        return (
            f"PreProcessorList: contains {len(self.pre_processor_list)} pre_processors, \n"
            + "\n".join(["    " + str(tmp) for tmp in self.pre_processor_list])
        )
