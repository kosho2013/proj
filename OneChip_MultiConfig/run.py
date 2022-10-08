import os
import threading


def run_one_config(workload, datatype, operation):
    os.system('python main.py --workload=' + workload + ' --datatype=' + datatype + ' --operation=' + operation + ' > ' + workload + '_' + datatype + '_' + operation + '.txt')
        

threads = []
workloads = ['resnet18']
datatypes = ['BF16']
operations = ['centralized']

for workload in workloads:
    for datatype in datatypes:
        for operation in operations:
            thread = threading.Thread(target=run_one_config, args=(workload, datatype, operation))
            threads.append(thread)
            
    
for i in range(len(threads)):
    threads[i].start()
   
for i in range(len(threads)):
    threads[i].join()
