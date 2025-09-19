import os
from ALCore import AAL

def read_chunk(filepath, start_line=0, max_chars=2000):
    """
    分段读取文本，每次约 max_chars 字。
    如果最后一行超出限制，则整行推迟到下次读取。
    
    :param filepath: 文本文件路径
    :param start_line: 开始行号（从 0 开始计数）
    :param max_chars: 每次读取的最大字数
    :return: (text, next_line)
             text: 本次读取的文本
             next_line: 下一次读取的起始行号
    """
    content = []
    char_count = 0
    next_line = start_line

    with open(filepath, "r", encoding="utf-8") as f:
        # 跳过起始行之前的部分
        for _ in range(start_line):
            f.readline()

        # 开始逐行读取
        for line in f:
            line_len = len(line)
            if char_count + line_len > max_chars:
                # 这一行会超出限制 → 推到下一次
                break
            content.append(line)
            char_count += line_len
            next_line += 1

    return "".join(content), next_line


# 使用示例
core = AAL()
filepath = "dataset\yifu.txt"
start = 0
while True:
    text, start = read_chunk(filepath, start)
    if not text:
        print("读取结束")
        break
    print("==== 本次读取 ====")
    # print(text)
    print(f"下一次起始行号: {start}")
    core.selfModeling(text, "伊芙 以及 伊芙特罗娜")
    # 假设这里只读两段就退出
    if start > 1000 or os.path.exists("STOP"):  # demo: 只读部分
        print(f"下一次起始行号: {start}")
        break
# ==== 本次读取 ====
# 下一次起始行号: 113