import unittest

import numpy as np
import faiss


class FaissGpuTestCase(unittest.TestCase):
    """验证 faiss GPU 环境是否可用的基础单元测试。"""

    @classmethod
    def setUpClass(cls):
        # 生成一组可重复的向量，方便在 CPU/GPU 上做一致性比对
        cls.dim = 32
        rng = np.random.default_rng(42)
        cls.database = rng.random((128, cls.dim), dtype=np.float32)
        cls.query = rng.random((8, cls.dim), dtype=np.float32)
        cls.cpu_index = faiss.IndexFlatL2(cls.dim)
        cls.cpu_index.add(cls.database)

    def test_gpu_device_available(self):
        """确保至少存在一个可供 faiss 使用的 GPU。"""
        gpu_count = faiss.get_num_gpus()
        self.assertGreater(gpu_count, 0, "未检测到可用 GPU，faiss GPU 环境异常")

    def test_gpu_search_consistency(self):
        """在 GPU 上运行检索并验证结果与 CPU Index 一致。"""
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(self.dim))
        gpu_index.add(self.database)
        cpu_distances, cpu_indices = self.cpu_index.search(self.query, 4)
        gpu_distances, gpu_indices = gpu_index.search(self.query, 4)
        self.assertTrue(
            np.allclose(cpu_distances, gpu_distances, atol=1e-5),
            "GPU 搜索与 CPU 距离结果不一致",
        )
        self.assertTrue(
            np.array_equal(cpu_indices, gpu_indices),
            "GPU 搜索与 CPU 最近邻索引不一致",
        )


if __name__ == "__main__":
    unittest.main()
