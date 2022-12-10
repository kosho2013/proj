pars = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

for par in pars:
    f = open('resnet18_par'+str(par)+'.txt')
    lines = f.readlines()
    f.close()
    
    print(par)
    for line in lines:
        if line.startswith('#'):
            print('# of configs', line.split()[3])
        
        if line.startswith('throughput'):
            print('thru', line.split()[1])
            
        if line.startswith('OI'):
            print('OI', line.split()[1])
            
    print()
    print()
    print()