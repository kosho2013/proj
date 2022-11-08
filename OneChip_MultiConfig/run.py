import os
import threading


def run_pixelfly_block32(par):
    os.system('python Topo_pixelfly_block32.py --par='+str(par)+' > pixelfly_block32_par'+str(par)+'.txt')
        
def run_resnet18(par):
    os.system('python Topo_resnet18.py --par='+str(par)+' > resnet18_par'+str(par)+'.txt')
    
threads = []
pars = [1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


for par in pars:
    thread = threading.Thread(target=run_pixelfly_block32, args=(par,))    
    threads.append(thread)
    
    thread = threading.Thread(target=run_resnet18, args=(par,))    
    threads.append(thread)
            
    
for i in range(len(threads)):
    threads[i].start()
   
for i in range(len(threads)):
    threads[i].join()
