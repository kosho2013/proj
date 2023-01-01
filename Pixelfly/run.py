import os
import threading


def run_one_config(workload, datatype, operation):
    os.system('python main.py --workload=' + workload + ' --datatype=' + datatype + ' --operation=' + operation + ' > ' + workload + '_' + datatype + '_' + operation + '.txt')
        

threads = []
workloads = ['pixelfly_block8', 'pixelfly_block16', 'pixelfly_block32']
datatypes = ['BF16']
operations = ['centralized', 'data', 'model', 'hybrid']

for workload in workloads:
    for datatype in datatypes:
        for operation in operations:
            thread = threading.Thread(target=run_one_config, args=(workload, datatype, operation))
            threads.append(thread)
            
    
for i in range(len(threads)):
    threads[i].start()
   
for i in range(len(threads)):
    threads[i].join()
