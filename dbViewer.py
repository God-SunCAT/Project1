import streamlit as st
import numpy as np
import pickle
import os
from module.VectorDB import SimpleVectorDB  # 假设你把上面类放在 simple_vector_db.py 里
from module.LlamaRequest import llm_embedding
# ========================
# 初始化或加载数据库
# ========================
# streamlit run dbViewer.py
persist_path = "./db/SelfModeling_VectorDB"  # 你的持久化路径
# persist_path = "./db/Memory_VectorDB"  # 你的持久化路径
db = SimpleVectorDB(dim=1024, persist_path=persist_path)

st.title("🧩 SimpleVectorDB 可视化查看器")

# ========================
# 添加向量
# ========================
st.header("添加数据")
text_input = st.text_area("输入要存储的文本")
if st.button("添加随机向量+文本"):
    if text_input.strip():
        vector = np.random.rand(db.dim).astype("float32")
        db.add(vector, {"content": text_input})
        st.success("✅ 已添加数据！")
    else:
        st.warning("请输入文本内容！")

# ========================
# 查询
# ========================
st.header("向量查询")
query_text = st.text_area("输入查询文本")
top_k = st.slider("返回数量 k", 1, 10, 5)

if st.button("查询"):
    if query_text.strip():
        query_vector = llm_embedding(query_text) # 这里可替换为 embedding
        results = db.query(query_vector, k=top_k)
        st.subheader("🔍 查询结果")
        for i, (data, dist) in enumerate(results):
            st.write(f"Rank {i+1} | 距离: {dist:.4f}")
            st.json(data)
    else:
        st.warning("请输入查询内容！")

# ========================
# 浏览数据库
# ========================
st.header("数据库内容浏览")
if len(db.data_store) == 0:
    st.info("数据库为空")
else:
    for idx, (id, data) in enumerate(db.data_store.items()):
        with st.expander(f"ID: {id}"):
            st.json(data)
            if st.button(f"删除 ID {id}", key=f"del_{id}"):
                db.remove(id)
                st.warning(f"已删除 ID {id}")
                st.experimental_rerun()
