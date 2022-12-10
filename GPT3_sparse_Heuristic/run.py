import os
import threading

num_chip = [8, 4, 2, 1]
pars = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

for i in num_chip:    
    for par in pars:
        os.system('python Topo_GPT3.py --num_chip '+str(i)+' --par '+str(par))

