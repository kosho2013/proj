import gurobipy as gp
import yaml
import pandas as pd
import pydot
import math
import copy
import argparse
import pprint
import sys
import numpy as np


class tbuffer:
    def __init__(self, name, tenor_size):
        self.name = name
        self.tenor_size = tenor_size
        self.downstream_dict = {}
        self.num = 0
        
    def update_depth(self, depth, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name] = [depth, 1, 0] # d, part, num 
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[key][2] = math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
            
            self.num += math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
    
    def update_buffer_partitioning(self, partition, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name][1] = partition
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[downstream_buffer_name][2] = math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
            
            self.num += math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
            
            
class tcompute:
    def __init__(self, name, m, k, n, lanes_par, stages_par, topo_num): 
        self.name = name
        self.m = m
        self.k = k
        self.n = n
        
        self.lanes_par = lanes_par
        self.stages_par = stages_par
        self.lanes = LANES * self.lanes_par
        self.stages = STAGES * self.stages_par
        self.num = self.stages_par * self.lanes_par
        
        self.topo_num = topo_num
        
        if self.k == -1:
            self.cycles = math.ceil(self.m / self.lanes) * self.n
            self.compute = 'SIMD'
        else:
            if self.n == 1:
                self.cycles = math.ceil(self.m / self.lanes) * self.k       
                self.compute = 'SIMD'     
            else:
                self.cycles = math.ceil(self.m / self.lanes) * self.k * math.ceil(self.n / self.stages)
                self.compute = 'Systolic'
                
    
    def update_compute_stitching(self, lanes_par, stages_par):
        self.lanes_par = lanes_par
        self.stages_par = stages_par
        self.lanes = LANES * self.lanes_par
        self.stages = STAGES * self.stages_par
        self.num = self.stages_par * self.lanes_par
        
        if self.k == -1:
            self.cycles = math.ceil(self.m / self.lanes) * self.n
            self.compute = 'SIMD'
        else:
            if self.n == 1:
                self.cycles = math.ceil(self.m / self.lanes) * self.k       
                self.compute = 'SIMD'     
            else:
                self.cycles = math.ceil(self.m / self.lanes) * self.k * math.ceil(self.n / self.stages)
                self.compute = 'Systolic'


    def update_topo_num(self, topo_num):
        self.topo_num = topo_num




def add_node(node_dict, node):
    if isinstance(node, tbuffer):
        label = 'name: '+str(node.name)+'\n'+'tenor_size: '+str(node.tenor_size)+'\n'+'num: '+str(node.num)+'\n'
        
        for key, [d, part, num] in node.downstream_dict.items():            
            label += key+', d: '+str(d)+', part: '+str(part)+', num: '+str(num)+'\n'
        
        pydot_node = pydot.Node(node.name, style="filled", fillcolor="green", label=label)
        node_dict[node.name] = [node, pydot_node]
        
    elif isinstance(node, tcompute):
        label = 'name: '+str(node.name)+'\n'+'m: '+str(node.m)+', k: '+str(node.k)+', n: '+str(node.n)+'\n'+'lanes: '+str(node.lanes)+'\n'+'stages: '+str(node.stages)+'\n'+'num: '+str(node.num)+'\n'+'cycles: '+str(node.cycles)+'\n'+str(node.compute)+'\n'+'topo_num: '+str(node.topo_num)+'\n'
        
        pydot_node = pydot.Node(node.name, style="filled", fillcolor="red", label=label)
        node_dict[node.name] = [node, pydot_node]
        
    else:
        raise Exception('Wrong node type!')


def add_edge(edge_dict, node1, node2, label):
    if label == 'lanes' or label == 'stages' or label == ' ':
        if node1 in edge_dict.keys():
            edge_dict[node1].append([node2, label])
        else:
            edge_dict[node1] = [[node2, label]]
    else:
        raise Exception('Wrong label type!')

        
        
def bfs(node_dict, edge_dict, reverse_edge_dict):
    indegree_map = {}
    for _, [node, _] in node_dict.items():
        indegree_map[node.name] = 0
    
    for _, [node, _] in node_dict.items():
        start = node.name
        if start in edge_dict.keys():
            for end, _ in edge_dict[start]:
                if end in indegree_map.keys():
                    indegree_map[end] += 1
                else:
                    indegree_map[end] = 1

    cnt = 0
    curr_tmp = set()
    next_tmp = set()
    
    for node in indegree_map.keys():
        if indegree_map[node] == 0:
            curr_tmp.add(node)
            
    while len(curr_tmp) > 0:
        flag = False
        
        for node in curr_tmp:
            if indegree_map[node] == 0:
                if node in edge_dict.keys():
                    for end, _ in edge_dict[node]:
                        next_tmp.add(end)
                        indegree_map[end] -= 1
            
            if isinstance(node_dict[node][0], tcompute):
                flag = True
                node_dict[node][0].update_topo_num(cnt)
                add_node(node_dict, node_dict[node][0])
        
        if flag:
            cnt += 1
        curr_tmp = copy.deepcopy(next_tmp)
        next_tmp = set()
        
    

    for _, [node, _] in node_dict.items():
        if isinstance(node, tbuffer):
            if node.name in edge_dict.keys():
                if node.name in reverse_edge_dict:
                    a = node_dict[reverse_edge_dict[node.name][0][0]][0].topo_num
                    
                    for end, _ in edge_dict[node.name]:
                        b = node_dict[end][0].topo_num
                        node.update_depth(b-a+1, end)
                        add_node(node_dict, node)
                else:
                    for end, _ in edge_dict[node.name]:
                        node.update_depth(1, end)
                        add_node(node_dict, node)
            else:
                node.update_depth(1, ' ')
                add_node(node_dict, node)
                    
            
            
        
    







def count(node_dict):
    pcu = 0
    pmu = 0
    for key in node_dict.keys():
        if isinstance(node_dict[key][0], tcompute):
            pcu += node_dict[key][0].num
        elif isinstance(node_dict[key][0], tbuffer):
            pmu += node_dict[key][0].num
    return pcu, pmu
    
    
def check_layer_num(node):
    for i in range(len(node.name)):
        if node.name[i].isdigit():
            return int(node.name[i:])
    raise Exception('Wrong node name, no layer number!')
    
    
    
def plot(graph, name, node_dict, edge_dict):
    # node
    for _, [_, pydot_node] in node_dict.items():
        graph.add_node(pydot_node)
    
    # edge
    for node1 in edge_dict.keys():
        for node2, label in edge_dict[node1]:
            graph.add_edge(pydot.Edge(node_dict[node1][1], node_dict[node2][1], label=label))
                
    graph.write_png('./workload/'+name+'.png')          
                
                
                
                
                
                
                
                
def get_cycles(node_dict, total_layer):   
    cycles = -1
    for _, [node, _] in node_dict.items():
        if isinstance(node, tcompute):
            cycles = max(cycles, node.cycles)
    return cycles
        
        
        
        
def convert(tmp, ba):
    if isinstance(tmp, str):
        value = 1
        nums = tmp.split('x')
        for i in nums:
            if i == 'ba':
                value *= ba
            else:
                value *= int(i)
        return value
    else:
        return tmp
        
        
        
    
if __name__ == '__main__':


    
    workload = 'pixelfly_block32'
    datatype = 'BF16'
    word = 2
    
    
        
        
    
    PMU = 640
    PCU = 640
    CAPACITY = 524288 # B
    LANES = 64 # B
    LANES = int(LANES / word) # number of data
    STAGES = 6 # number of data
    FREQ = 1.25 # GHz
    DRAM_BW = 2039.0 # GB/s
    
    
    
    
    
    
    
    
    
    content_workload = pd.ExcelFile('./workload/'+workload+'.xlsx').parse('Sheet1')
    
    
    
    
    
    
    
    





    batch = [32]
    
    for ba in batch:
        print('batch', ba, '***********************')
        graph = pydot.Dot(graph_type='digraph')
        edge_dict = {}
        reverse_edge_dict = {}
        node_dict = {}
        layer_dict = {}
        total_layer = 0
        
        
        
        for layer in content_workload.iterrows():
            total_layer += 1
            layer_dict[total_layer] = layer
            
        

        flop = 0
        # forward loop
        for i in range(1, total_layer+1):
            layer = layer_dict[i]
            
            layer_num = int(layer[1]['layer_num'])
            layer_type = str(layer[1]['layer_type'])
            sparsity = str(layer[1]['sparsity'])
            m = layer[1]['m']
            k = layer[1]['k']
            n = layer[1]['n']
            from_dram = str(layer[1]['from_dram'])
            
            
            m = convert(m, ba)
            k = convert(k, ba)
            n = convert(n, ba)
            
            
            
            if layer_type == 'gemm':
                if from_dram == 'yes':
                    w_tbuffer = tbuffer('w'+str(layer_num), m*k*word)  
                    add_node(node_dict, w_tbuffer)
                    
                    
                    if sparsity == 'pixelfly':
                        in_tbuffer = tbuffer('in'+str(layer_num), m*n*word)
                    elif sparsity == 'dense':
                        in_tbuffer = tbuffer('in'+str(layer_num), k*n*word)
                    else:
                        raise Exception('Wrong sparsity!')
                    
                    add_node(node_dict, in_tbuffer)
                    
                    
                    
                    f_compute = tcompute('forward'+str(layer_num), m, k, n, 1, 1, -1)
                    add_node(node_dict, f_compute)
                    add_edge(edge_dict, w_tbuffer.name, f_compute.name, 'lanes')
                    add_edge(edge_dict, in_tbuffer.name, f_compute.name, 'stages')
                    
                    
                    out_tbuffer = tbuffer('in'+str(layer_num+1), m*n*word)
                    add_node(node_dict, out_tbuffer)
                    add_edge(edge_dict, f_compute.name, out_tbuffer.name, 'lanes')
                
                elif from_dram == 'no':
                    w_tbuffer = tbuffer('w'+str(layer_num), m*k*word)  
                    add_node(node_dict, w_tbuffer)
                    
                    
                    
                    
                    f_compute = tcompute('forward'+str(layer_num), m, k, n, 1, 1, -1)
                    add_node(node_dict, f_compute)
                    add_edge(edge_dict, w_tbuffer.name, f_compute.name, 'lanes')
                    add_edge(edge_dict, 'in'+str(layer_num), f_compute.name, 'stages')
                    
                    
                    
                    out_tbuffer = tbuffer('in'+str(layer_num+1), m*n*word)
                    add_node(node_dict, out_tbuffer)
                    add_edge(edge_dict, f_compute.name, out_tbuffer.name, 'lanes')
                    
                else:
                    raise Exception('Wrong from_dram!')
                
                flop += m * k * n * 9
                
            elif layer_type == 'loss':
                f_compute = tcompute('loss'+str(layer_num), m, k, n, 1, 1, -1) 
                add_node(node_dict, f_compute)
                add_edge(edge_dict, 'in'+str(layer_num), f_compute.name, 'lanes')
                
                dataGradient_tbuffer = tbuffer('dataGradient'+str(layer_num), node_dict['in'+str(layer_num)][0].tenor_size)
                    
                add_node(node_dict, dataGradient_tbuffer)
                add_edge(edge_dict, f_compute.name, dataGradient_tbuffer.name, 'lanes')
                    
                flop += m * n
                
            else:
                raise Exception('Wrong layer_type!')
                
                
        
        
        
        # backward loop
        for i in range(total_layer, 0, -1):
            layer = layer_dict[i]
            
            layer_num = int(layer[1]['layer_num'])
            layer_type = str(layer[1]['layer_type'])
            sparsity = str(layer[1]['sparsity'])
            m = layer[1]['m']
            k = layer[1]['k']
            n = layer[1]['n']
            from_dram = str(layer[1]['from_dram'])
            
            m = convert(m, ba)
            k = convert(k, ba)
            n = convert(n, ba)
            
            
            
            if layer_type == 'gemm':         
                if sparsity == 'pixelfly':
                    dg_compute = tcompute('backpropDataGradient'+str(layer_num), m, k, n, 1, 1, -1) 
                elif sparsity == 'dense':
                    dg_compute = tcompute('backpropDataGradient'+str(layer_num), k, m, n, 1, 1, -1) 
                else:
                    raise Exception('Wrong sparsity!') 
                add_node(node_dict, dg_compute)
                add_edge(edge_dict, 'w'+str(layer_num), dg_compute.name, 'lanes')
                add_edge(edge_dict, 'dataGradient'+str(layer_num+1), dg_compute.name, 'stages')
                
                
                if sparsity == 'pixelfly':
                    dg_buffer = tbuffer('dataGradient'+str(layer_num), m*n*word)
                elif sparsity == 'dense':
                    dg_buffer = tbuffer('dataGradient'+str(layer_num), k*n*word)
                else:
                    raise Exception('Wrong sparsity!') 
                add_node(node_dict, dg_buffer)
                add_edge(edge_dict, dg_compute.name, dg_buffer.name, 'lanes')
                
                
                wg_compute = tcompute('backpropWeightGradient'+str(layer_num), m, n, k, 1, 1, -1) 
                add_node(node_dict, wg_compute)
                add_edge(edge_dict, 'in'+str(layer_num), wg_compute.name, 'stages')
                add_edge(edge_dict, 'dataGradient'+str(layer_num+1), wg_compute.name, 'lanes')
                
                
                wg_tbuffer = tbuffer('weightGradient'+str(layer_num), m*k*word)
                add_node(node_dict, wg_tbuffer)
                add_edge(edge_dict, wg_compute.name, wg_tbuffer.name, 'lanes')
            
                    
                wu_compute = tcompute('weightUpdate'+str(layer_num), m, -1, k, 1, 1, -1)
                add_node(node_dict, wu_compute)
                add_edge(edge_dict, 'weightGradient'+str(layer_num), wu_compute.name, 'lanes') 
            elif layer_type == 'loss':
                continue
            else:
                raise Exception('Wrong layer_type!')
            
            
        

        # create reverse edge_dict
        for node1 in edge_dict.keys():
            for node2, label in edge_dict[node1]:
                if node2 in reverse_edge_dict.keys():
                    reverse_edge_dict[node2].append([node1, label])
                else:
                    reverse_edge_dict[node2] = [[node1, label]]

  
        bfs(node_dict, edge_dict, reverse_edge_dict)
        
        
        
        plot(graph, 'aaa', node_dict, edge_dict)
        
        
        
                
        Nb = 0
        Nb_cin = []
        Nb_cout = []
        Nb_dim = []
        TSb = []
        
        Nc = 0
        mkn = {}
        Nc_name = []
        M = []
        K = []
        N = []
        
        Nd = 0
        Nd_cout = []
        Nd_dim = []
        TSd = []
        
        for key in node_dict.keys():
            node = node_dict[key][0]
            if isinstance(node, tcompute):
                if node.topo_num in mkn.keys():
                    mkn[node.topo_num].append((node.name, node.m, node.k, node.n))
                else:
                    mkn[node.topo_num] = [(node.name, node.m, node.k, node.n)]
                Nc += 1
                
            elif isinstance(node, tbuffer):
                if node.name in reverse_edge_dict.keys():
                    for key, _ in node.downstream_dict.items():
                        if key != ' ':
                            Nb += 1
                            Nb_cin.append(reverse_edge_dict[node.name][0][0])
                            Nb_cout.append(key)
                            
                            for next_node, dim in edge_dict[node.name]:
                                if next_node == key:
                                    Nb_dim.append(dim)
                            
                            TSb.append(node.tenor_size)
                        
                else:
                    for key, _ in node.downstream_dict.items():
                        Nd += 1
                        Nd_cout.append(key)
                        
                        for next_node, dim in edge_dict[node.name]:
                            if next_node == key:
                                Nd_dim.append(dim)
                        
                        TSd.append(node.tenor_size)
    
        
        for i in range(Nc):
            if i in mkn.keys():
                for value in mkn[i]:
                    Nc_name.append(value[0])
                    M.append(value[1])
                    K.append(value[2])
                    N.append(value[3])
        
        



        
        
        Nc_dict = {}
        for i in range(len(Nc_name)):
            Nc_dict[Nc_name[i]] = i
        
        
        
        PCU_lim = PCU
        PMU_lim = PMU
        Cap = CAPACITY
        VecWidth = LANES
        StageWidth = STAGES
        DRAM_BW = DRAM_BW
        Freq = FREQ
        Input = node_dict['in1'][0].tenor_size
        C = 5
        
        
        print('PCU_lim', PCU_lim)
        print('PMU_lim', PMU_lim)
        print('Cap', Cap)
        print('VecWidth', VecWidth)
        print('StageWidth', StageWidth)
        print('Freq', Freq)
        print('DRAM_BW', DRAM_BW)
        print('Input', Input)
        print('C', C)
        print('Nb', Nb)
        print('Nb_cin', Nb_cin)
        print('Nb_cout', Nb_cout)
        print('Nb_dim', Nb_dim)
        print('TSb', TSb)
        print('Nc', Nc)
        print('Nc_name', Nc_name)
        print('M', M)
        print('K', K)
        print('N', N)
        print('Nd', Nd)
        print('Nd_cout', Nd_cout)
        print('Nd_dim', Nd_dim)
        print('TSd', TSd)
        
        
        
        FLOP = flop
        
        
        
        
        
        



        
        
        
        

        model = gp.Model()
        model.params.NonConvex = 2
        
        
        
        
        Config = model.addMVar(Nc, name='Config', vtype=gp.GRB.INTEGER, lb=0)
        Ac = model.addMVar((Nc, C), name='Ac', vtype=gp.GRB.BINARY)
        Par_lane = model.addMVar(Nc, name='Par_lane', vtype=gp.GRB.INTEGER, lb=1)
        Par_stage = model.addMVar(Nc, name='Par_stage', vtype=gp.GRB.INTEGER, lb=1)
        Par_total = model.addMVar(Nc, name='Par_total', vtype=gp.GRB.INTEGER, lb=1)
        # Ab1 = model.addVars(Nb, C, name='Ab1', vtype=gp.GRB.BINARY)
        # Ab2 = model.addMVar((Nb, C), name='Ab2', vtype=gp.GRB.BINARY)
        # Ab3 = model.addMVar((Nb, C), name='Ab3', vtype=gp.GRB.BINARY)


        
        t1 = np.ones((C))
        for i in range(Nc):
            model.addConstr(Ac[i, :] @ t1 == 1)
            
        
        t2 = np.zeros((C))
        for i in range(C):
            t2[i] = i
        for i in range(Nc):
            model.addConstr(Ac[i, :] @ t2 == Config[i])

    
        
        for i in range(Nb):
            cin_idx = Nc_dict[Nb_cin[i]]
            cout_idx = Nc_dict[Nb_cout[i]]
        
            model.addConstr(Config[cin_idx] <= Config[cout_idx])
        
        
        
        for i in range(Nc):
            model.addConstr(Par_lane[i] @ Par_stage[i] == Par_total[i])
        for i in range(C):
            model.addConstr(Par_total @ Ac[:, i] <= PCU_lim)
        
        
        
        
        # for i in range(Nb):
            # cin_idx = Nc_dict[Nb_cin[i]]
            # cout_idx = Nc_dict[Nb_cout[i]]
            
            # for j in range(C):
                # model.addConstr((Par_lane[i].tolist()[0] > 3) >> (Par_stage[i].tolist()[0] == 3))
                
        
        
        
        
        
        model.setObjective(Config[-1], gp.GRB.MAXIMIZE)
        
        model.optimize()


        print('Obj:', model.objVal)
        for v in model.getVars():
            print(v.varName, v.x)
            













