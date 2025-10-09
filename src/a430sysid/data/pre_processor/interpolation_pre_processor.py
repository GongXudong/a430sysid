import pandas as pd

from a430sysid.data.pre_processor.pre_processor_base import PreProcessorBase


class InterpolationPreProcessor(PreProcessorBase):
    def __init__(
        self,
        method: str = "linear",
        limit_direction: str = "forward",
        interpolation_columns: list[str] = [],
    ):
        """
        Args:
            method: 插值方法 ('linear', 'cubic', 'spline', 'polynomial')
            limit_direction: 限制方向 ('forward', 'backward', 'both')
        """
        self.method = method
        self.limit_direction = limit_direction
        self.interpolation_columns = interpolation_columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用插值方法填充缺失值

        参数:
        df: 包含轨迹数据的DataFrame
        method: 插值方法 ('linear', 'cubic', 'spline', 'polynomial')
        limit_direction: 限制方向 ('forward', 'backward', 'both')
        """

        assert set(self.interpolation_columns) <= set(df.columns)

        df_interpolated = df.copy()

        # 为每个列单独进行插值
        for column in self.interpolation_columns:
            if df[column].isnull().sum() > 0:
                print(f"\n正在处理 {column} 列的缺失值...")

                # 使用pandas的interpolate方法
                df_interpolated[column] = df[column].interpolate(
                    method=self.method,
                    limit_direction=self.limit_direction,
                    order=(
                        3 if self.method in ["cubic", "spline", "polynomial"] else None
                    ),
                )

                # 如果还有缺失值（比如在开头或结尾），使用最近的有效值填充
                if df_interpolated[column].isnull().sum() > 0:
                    df_interpolated[column] = (
                        df_interpolated[column]
                        .fillna(method="bfill")
                        .fillna(method="ffill")
                    )

        # 验证填充结果
        remaining_missing = (
            df_interpolated[self.interpolation_columns].isnull().sum().sum()
        )
        if remaining_missing == 0:
            print("\n✓ 所有缺失值已成功填充!")
        else:
            print(f"\n⚠ 仍有 {remaining_missing} 个缺失值未填充")

        return df_interpolated

    def detect_missing_values(self, df: pd.DataFrame):
        """检测数据中的缺失值"""
        print("=== 缺失值检测结果 ===")
        print(f"数据集总行数: {len(df)}")
        print("\n各列缺失值统计:")
        missing_stats = df.isnull().sum()
        print(missing_stats)

        print(f"\n总缺失值数量: {df.isnull().sum().sum()}")
        print(f"缺失值比例: {df.isnull().sum().sum() / df.size * 100:.2f}%")

        # 检查连续缺失的情况
        for column in df.columns:
            missing_runs = (df[column].isnull() != df[column].isnull().shift()).cumsum()
            missing_groups = (
                df[column]
                .isnull()
                .groupby(missing_runs)
                .apply(lambda x: len(x) if x.iloc[0] else 0)
            )
            consecutive_missing = missing_groups[missing_groups > 0]
            if len(consecutive_missing) > 0:
                print(f"\n{column}列连续缺失情况:")
                print(f"  连续缺失段数: {len(consecutive_missing)}")
                print(f"  最大连续缺失长度: {consecutive_missing.max()}")

        return missing_stats

    def __str__(self):
        return f"InterpolationPreProcessor: interpolate {self.interpolation_columns} with method = {self.method}, limit_direction = {self.limit_direction}."
