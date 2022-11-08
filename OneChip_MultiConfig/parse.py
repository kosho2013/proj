
pars = [1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


for app in ['pixelfly_block32', 'resnet18']:
    for par in pars:
        f = open(app+'_par'+str(par)+'.txt', 'r')
        lines = f.readlines()
        
        
        
        curr_ba = -1
        curr_oi = -1
        curr_config = -1
        final = [-1, -1, -1, -1] # thru, batch, OI, config
        
        for line in lines:
            tmp = line.split()
            if len(tmp) > 0:
                if tmp[0] == 'batch':
                    curr_ba = int(tmp[1])
                elif tmp[0] == 'OI':
                    curr_oi = float(tmp[1])
                elif tmp[0] == '#':
                    curr_config = int(tmp[3])
                elif tmp[0] == 'throughput' and float(tmp[1]) > final[0]:    
                    final[0] = float(tmp[1])
                    final[1] = curr_ba
                    final[2] = curr_oi
                    final[3] = curr_config
                    
                    
        if final == [-1, 1]:
            print('Not valid!!')
        else:
            print(app)
            print('batch', final[1])
            print('par', par)
            print('OI', final[2])
            print('# of config', final[3])
            print('throughput', final[0])
        
        
        
        print()
        print()
        print()
        
        
        f.close()
        
        