import os
import threading

def run(i, j):
    os.system('python GPT3.py --num_chip '+str(i)+' --C '+str(j)+' > '+str(i)+'x'+str(8/i)+'_C'+str(j)+'.txt')


# num_chip = [8, 4, 2, 1]
num_chip = [1]
C = [4, 5, 6]



threads = []
for i in num_chip:
    for j in C:
        thread = threading.Thread(target=run, args=(i, j))
        threads.append(thread)
        
        
        
for i in range(len(threads)):
    threads[i].start()
   
for i in range(len(threads)):
    threads[i].join()


