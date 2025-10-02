import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

from ALCore import AAL

core = AAL()
print("Project1 - An Advanced Artificial Life")

history = core.conf['history']
for i in history:
    print(f'{i[0]}: {i[1]}')

while(True):
    print("User:", end=" ")
    x = input()

    t = time.time()
    output = core.ask(x)
    print("伊芙:", output, f'\n<--> {(time.time() - t):.2f}秒')

    # 自我建模，这里为了方便就不使用多线程和线程锁了。
    if(len(core.conf['history']) >= 30):
        core.selfModeling(core.conf['history'])
