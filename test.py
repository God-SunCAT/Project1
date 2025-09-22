from sklearn.cluster import KMeans
from module.LlamaRequest import llm_embedding
# 示例文本数据
documents = [
    "猫喜欢吃鱼",
    "狗是人类的朋友",
    "我喜欢养猫",
    "狗很忠诚",
    "鱼在水里游",
    "小猫很可爱",
    "小狗会摇尾巴"
]

# 1. 文本转向量（TF-IDF）
X = []
for x in documents:
    X.append(llm_embedding(x))

# 2. KMeans 聚类
num_clusters = 3  # 你可以根据需要调整聚类数量
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# 3. 输出结果
print("每条文本的聚类结果：")
for doc, label in zip(documents, kmeans.labels_):
    print(f"【类别 {label}】 {doc}")
