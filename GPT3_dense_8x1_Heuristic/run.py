import os
import threading

pars = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]


for par in pars:
    os.system('python Topo_GPT3.py --par='+str(par))

