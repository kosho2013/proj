# packing


def GPU(side, batch, block, datatype):
    w = side * block * 2 * datatype # bytes
    inp = side * batch * datatype # bytes
    flop = 6 * side * block * 2 * batch
    b = w + inp + w + inp + inp
    oi = flop / b
    print('flop', flop / batch)
    print('oi', oi)


def RDU(side, batch, block, datatype):
    inp = side * batch * datatype # bytes
    flop = 6 * side * block * 2 * batch
    b = inp
    oi = flop / b
    print('oi', oi)



side = 4096

for batch in [32, 4096]:
    for scheme in ['packing', 'random']:
        for block in [8, 16, 32]:
            if scheme == 'packing' and (block == 16 or block == 32):
                datatype = 2
                print('batch', batch, 'packing', 'block', block, 'datatype', datatype)
                GPU(side, batch, block, datatype)
            else:
                datatype = 4
                print('batch', batch, 'random', 'block', block, 'datatype', datatype)
                GPU(side, batch, block, datatype)
                
            print()
            print()
            print()


for block in [8, 16, 32]:
    print('RDU, block', block)
    RDU(side, 32, block, 2)
    
    print()
    print()
    print()