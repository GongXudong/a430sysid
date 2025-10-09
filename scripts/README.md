# scripts

## process_trajectory.py

飞行轨迹预处理脚本。

根据配置文件预处理飞行轨迹，配置文件位于 *configs/data/pre_processing_config.yaml*.

配置文件说明：

```yaml
pre_processors: dict  # 预处理器配置
data_dir: str  # 轨迹数据存放目录
to_replaced_path_str: str
new_path_str: str  # 处理好的轨迹数据的存放位置，具体而言，使用{new_path_str}替换掉{data_dir}中的{to_replaced_path_str}
```

运行以下命令进行预处理：

```python
uv run src/a430sysid/data/scripts/process_trajectory.py 
```
