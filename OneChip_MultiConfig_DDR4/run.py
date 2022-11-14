import os
import threading


def run_resnet18(par):
    os.system('python Topo_resnet18.py --par='+str(par)+' > resnet18_par'+str(par)+'.txt')
    
threads = []
pars = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]


for par in pars:
    thread = threading.Thread(target=run_resnet18, args=(par,))    
    threads.append(thread)
            
    
for i in range(len(threads)):
    threads[i].start()
   
for i in range(len(threads)):
    threads[i].join()
