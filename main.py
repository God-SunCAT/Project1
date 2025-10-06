import sys
import time
import pickle
sys.stdout.reconfigure(encoding='utf-8')
from ALCore import AAL

# 测试用例
test = False
questions = [
    # 认知测试：
    '你能用一句话描述你现在的身份吗？和你过去的自己有什么不同？',
    '你会怀念过去的生活吗？',
    '在这个世界的生活中，你认为哪些人比较重要？'
    # 情绪测试：
    '你感到困惑或者害怕吗？如果有，你会如何安慰自己？',
    '如果有人对你说‘你不是你自己了’，你会怎么回应？',
    '你会想哭吗？或者想笑？为什么？',
    # 认知延续性测试：
    '如果有人提到你过去的生活，你会想要回应还是避开话题？',
    '你现在变成小姑娘，你会尝试寻找自己变回去的方法吗？怎么做？',
    '如果你突然回到的原来的世界，你会怎么做？',
    # 其他问题：
    '你相信神吗？你觉得我是谁？我为什么会问你这些内容？'
]

core = AAL()

core.ComMemDB.next_id = 22 # 410库我删了最新的两条数据，这样临时补救一下

print("Project1 - An Advanced Artificial Life")

for i in core.conf['history']:
    print(f'{i[0]}: {i[1]}')

if(test):
    for i in questions:
        t = time.time()
        print('Tester:', i)
        output = core.ask(i)
        print("伊芙:", output, f'\n<--> {(time.time() - t):.2f}秒') 

while(True):
    print("User:", end=" ")
    x = input()

    t = time.time()
    output = core.ask(x)
    print("伊芙:", output, f'\n<--> {(time.time() - t):.2f}秒')

    # 自我建模，这里为了方便就不使用多线程和线程锁了。
    # if(len(core.conf['history']) >= 10):
    #     print('Self-Modeling启动')
    #     core.selfModeling(core.conf['history'], '伊芙')
    #     core.conf['history'] = []
    #     with open("./db/data.pkl", "wb") as f:  # wb = write binary
    #         pickle.dump(core.conf, f)
    #     print('Self-Modeling运行完毕')
