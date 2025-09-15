import sys
sys.stdout.reconfigure(encoding='utf-8')

from ALCore import AAL

core = AAL()
print("Project1 - An Advanced Artificial Life")
while(True):
    print("User:", end=" ")
    x = input()
    print("Alice:", core.ask(x))
