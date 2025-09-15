import hnswlib
import numpy as np
import os
import pickle

class SimpleVectorDB:
    def __init__(self, dim=1024, max_elements=10000, persist_path=None):
        """
        dim: 向量维度
        max_elements: 初始最大元素数量（会自动扩容）
        persist_path: 持久化文件路径，可选
        """
        self.dim = dim
        self.max_elements = max_elements
        self.persist_path = persist_path
        self.id_map = set()
        self.text_store = {}  # id -> 原始文本

        self.index = hnswlib.Index(space='cosine', dim=dim)

        if persist_path and os.path.exists(persist_path + ".index"):
            # 加载已有索引
            self.index.load_index(persist_path + ".index", max_elements=max_elements)
            # 尝试加载文本数据
            if os.path.exists(persist_path + "_texts.pkl"):
                with open(persist_path + "_texts.pkl", "rb") as f:
                    self.text_store = pickle.load(f)
            self.id_map = set(self.text_store.keys())
        else:
            self.index.init_index(max_elements=max_elements, ef_construction=100, M=16)

        self.index.set_ef(50)  # 查询精度

        # 用于生成自增 id
        self.next_id = max(self.id_map, default=0) + 1

    def add(self, vector, text=None, id=None):
        """
        vector: embedding 向量
        text: 可选，原始文本/数据
        id: 可选，手动指定 id，否则自动生成
        """
        vector = np.array(vector, dtype='float32')

        # 检查容量，必要时扩容（翻倍）
        if len(self.id_map) >= self.index.get_max_elements():
            new_size = int(self.index.get_max_elements() * 1.5) + 1
            self.index.resize_index(new_size)

        if id is None:
            id = self.next_id
            self.next_id += 1

        self.index.add_items(vector.reshape(1, -1), np.array([id]))
        self.id_map.add(id)

        if text is not None:
            self.text_store[id] = text

        self._maybe_persist()
        return id

    def remove(self, id):
        if id in self.id_map:
            self.index.mark_deleted(id)
            self.id_map.remove(id)
            if id in self.text_store:
                del self.text_store[id]
            self._maybe_persist()

    def query(self, vector, k=5, return_text=True):
        vector = np.array(vector, dtype='float32')

        # 修正 k，防止超过已有数量
        available = len(self.id_map)
        if available == 0:
            return []
        k = min(k, available)

        labels, distances = self.index.knn_query(vector.reshape(1, -1), k=k)
        labels_list = labels[0].tolist()
        distances_list = distances[0].tolist()

        if return_text:
            results = [(self.text_store.get(l, None), d) for l, d in zip(labels_list, distances_list)]
        else:
            results = list(zip(labels_list, distances_list))
        return results

    def _maybe_persist(self):
        if self.persist_path:
            self.index.save_index(self.persist_path + ".index")
            with open(self.persist_path + "_texts.pkl", "wb") as f:
                pickle.dump(self.text_store, f)
