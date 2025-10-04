import streamlit as st
import numpy as np
from module.VectorDB import SimpleVectorDB, queryByWeight  # 假设你把上面类放在 module/VectorDB.py 里
from module.LlamaRequest import llm_embedding
import os
# streamlit run dbViewer.py
# ========================
# 数据库路径选择
# ========================

db_options = {
    "SelfModeling_VectorDB": "./db/SelfModeling_VectorDB",
    "DetailMemory_VectorDB": "./db/DetailMemory_VectorDB",
    "CompressionMemory_VectorDB": "./db/CompressionMemory_VectorDB"
}

st.title("🧩 SimpleVectorDB 可视化查看器")

db_choice = st.selectbox("选择要操作的数据库", list(db_options.keys()))
persist_path = db_options[db_choice]

# 初始化数据库
db = SimpleVectorDB(dim=1024, persist_path=persist_path)

st.success(f"当前数据库：{db_choice} ({persist_path})")

# ========================
# 添加向量
# ========================
# st.header("添加数据")
# text_input = st.text_area("输入要存储的文本")
# if st.button("添加向量+文本"):
#     if text_input.strip():
#         vector = llm_embedding(text_input)
#         db.add(vector, {"content": text_input})
#         st.success("✅ 已添加数据！")
#     else:
#         st.warning("请输入文本内容！")

# ========================
# 查询
# ========================
st.header("向量查询")
query_text = st.text_area("输入查询文本")
top_k = st.slider("返回数量 k", 1, 10, 5)

if st.button("查询"):
    if query_text.strip():
        query_vector = llm_embedding(query_text)  # 这里可替换为 embedding
        # results = db.query(query_vector, k=top_k)
        results = queryByWeight(db, query_vector, top_k)
        st.subheader("🔍 查询结果")
        for i, (data, id, dist) in enumerate(results):
            st.write(f"Rank {i+1} | ID: {id} | 权重: {dist:.4f}")
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
            if st.button(f"删除 ID {id}", key=f"del_{id}_{db_choice}"):
                db.remove(id)
                st.warning(f"已删除 ID {id}")
                st.experimental_rerun()
