import sys
sys.stdout.reconfigure(encoding='utf-8')

from ALCore import AAL

core = AAL()
print("Project1 - An Advanced Artificial Life")
while(True):
    print("User:", end=" ")
    x = input()
    print("伊芙:", core.ask(x))
