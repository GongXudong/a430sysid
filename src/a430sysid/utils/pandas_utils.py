import numpy as np


def contains_strings(arr):
    """
    检查numpy数组中是否包含字符串元素

    参数:
        arr: 待检查的numpy数组

    返回:
        bool: 如果数组中存在字符串则返回True，否则返回False
    """
    # 检查数组的dtype是否为字符串类型
    if np.issubdtype(arr.dtype, np.str_):
        return True

    # 对于object类型的数组，需要逐个检查元素
    if arr.dtype == np.dtype("O"):
        # 使用vectorize创建向量化函数，检查每个元素是否为字符串
        is_string = np.vectorize(lambda x: isinstance(x, str))
        return np.any(is_string(arr))

    # 其他数据类型（如数值型）不包含字符串
    return False


def find_strings(arr):
    """
    检查numpy数组中字符串元素的数量和位置

    参数:
        arr: 待检查的numpy数组

    返回:
        dict: 包含以下键的字典:
            - 'contains_strings': 布尔值，表示是否包含字符串
            - 'count': 字符串元素的数量
            - 'positions': 字符串元素的坐标元组列表
            - 'values': 字符串元素的值列表
    """
    result = {"contains_strings": False, "count": 0, "positions": [], "values": []}

    # 对于字符串类型的数组，所有元素都是字符串
    if np.issubdtype(arr.dtype, np.str_):
        result["contains_strings"] = True
        result["count"] = arr.size
        # 生成所有元素的坐标
        result["positions"] = [tuple(indices) for indices in np.ndindex(arr.shape)]
        result["values"] = arr.flatten().tolist()
        return result

    # 对于object类型的数组，需要逐个检查元素
    if arr.dtype == np.dtype("O"):
        # 创建一个函数检查元素是否为字符串
        def is_string(x):
            return isinstance(x, str)

        # 向量化检查并获取掩码
        mask = np.vectorize(is_string)(arr)

        if np.any(mask):
            result["contains_strings"] = True
            result["count"] = np.sum(mask)
            # 获取所有True值的坐标
            result["positions"] = [tuple(indices) for indices in np.argwhere(mask)]
            # 获取对应的值
            result["values"] = arr[mask].tolist()
        return result

    # 其他数据类型不包含字符串
    return result


# 测试示例
if __name__ == "__main__":
    # 测试用例
    test_arrays = [
        np.array([1, 2, 3, 4]),
        np.array(["a", "b", "c"]),
        np.array([1, "two", 3.0, True, "five"]),
        np.array([[1, "apple"], ["banana", 3.14], [True, "cherry"]]),  # 二维数组
        np.array([1.5, 2.0, 3.7]),
        np.array([True, False, True]),
        np.array([1, 2, 3], dtype=object),
        np.array([]),
    ]

    for i, arr in enumerate(test_arrays):
        print(f"=== 测试用例 {i+1} ===")
        print("数组内容:", arr)
        print("数组形状:", arr.shape)

        info = find_strings(arr)
        print(f"是否包含字符串: {info['contains_strings']}")
        print(f"字符串数量: {info['count']}")

        if info["count"] > 0:
            print("字符串位置及值:")
            for pos, val in zip(info["positions"], info["values"]):
                print(f"  位置 {pos}: 值 '{val}'")
        print()

# 测试示例
# if __name__ == "__main__":
#     # 测试用例
#     test_arrays = [
#         np.array([1, 2, 3, 4]),                # 整数数组
#         np.array(['a', 'b', 'c']),             # 字符串数组
#         np.array([1, 'two', 3.0, True]),       # 混合类型的object数组
#         np.array([1.5, 2.0, 3.7]),             # 浮点数数组
#         np.array([True, False, True]),         # 布尔数组
#         np.array([1, 2, 3], dtype=object),     # object类型的数值数组
#         np.array([])                           # 空数组
#     ]

#     for i, arr in enumerate(test_arrays):
#         result = contains_strings(arr)
#         print(f"测试用例 {i+1}: {arr} → {'包含字符串' if result else '不包含字符串'}")
