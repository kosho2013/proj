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
        self.downstream_dict[downstream_buffer_name] = [depth, 1, 0, 1] # d: depth, part, num, d: 1 
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[key][2] = math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
            
            self.num += self.downstream_dict[key][2]
    
    def update_buffer_partitioning(self, partition, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name][1] = partition
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[downstream_buffer_name][2] = math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
            
            self.num += self.downstream_dict[key][2]
            
    def update_depth_to_one(self, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name][0] = self.downstream_dict[downstream_buffer_name][3]
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[key][2] = math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
            
            self.num += self.downstream_dict[key][2]
        
            
            
class tcompute:
    def __init__(self, name, m, k, n, lanes_par, stages_par, topo_num, allreduce): 
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
        
        self.allreduce = allreduce
        
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
        
        for key, [d, part, num, d_real] in node.downstream_dict.items():            
            label += key+', d: '+str(d)+', part: '+str(part)+', num: '+str(num)+', d: '+str(d_real)+'\n'
        
        pydot_node = pydot.Node(node.name, style="filled", fillcolor="green", label=label)
        node_dict[node.name] = [node, pydot_node]
        
    elif isinstance(node, tcompute):
        label = 'name: '+str(node.name)+'\n'+'m: '+str(node.m)+', k: '+str(node.k)+', n: '+str(node.n)+'\n'+'lanes: '+str(node.lanes)+'\n'+'stages: '+str(node.stages)+'\n'+'num: '+str(node.num)+'\n'+'cycles: '+str(node.cycles)+'\n'+str(node.compute)+'\n'+'topo_num: '+str(node.topo_num)+'\n'+'allreduce: '+str(node.allreduce)+'\n'
        
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

    
    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--num_chip', type=str)
    parser.add_argument('--C', type=str)
    args = parser.parse_args()
    
    
    
    num_chip = int(args.num_chip)
    C = int(args.C)
    
    

    datatype = 'BF16'
    word = 2
    
    
        
        
    
    PMU = 640
    PCU = 640
    CAPACITY = 524288 # B
    LANES = 64 # B
    LANES = int(LANES / word) # number of data
    STAGES = 6 # number of data
    FREQ = 1.25 # GHz
    DRAM_BW = 128.0 # GB/s
    PCIE_BW = 25.0 # GB/s
    
    
    
    
    
    
    

    
    d_model = 12288
    d_head = 128
    d_hidden = 49152
    batch = 2048
    
    n_head = 96
    
    





    graph = pydot.Dot(graph_type='digraph')
    edge_dict = {}
    reverse_edge_dict = {}
    node_dict = {}
    
    
        
    


    in1_tbuffer = tbuffer('in1', d_model*batch*word)
    add_node(node_dict, in1_tbuffer)
    
    
    
    
    # layer 1
    w1_Q_tbuffer = tbuffer('w1_Q', n_head/num_chip*d_head*d_model*word)  
    add_node(node_dict, w1_Q_tbuffer)

    f_tcompute = tcompute('forward1_Q', n_head/num_chip*d_head, d_model, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in2_Q_tbuffer = tbuffer('in2_Q', n_head/num_chip*d_head*batch*word)
    add_node(node_dict, in2_Q_tbuffer)
    
    add_edge(edge_dict, in1_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w1_Q_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in2_Q_tbuffer.name, 'lanes')
    
    
    
    
    
    
    w2_K_tbuffer = tbuffer('w1_K', n_head/num_chip*d_head*d_model*word)  
    add_node(node_dict, w2_K_tbuffer)

    f_tcompute = tcompute('forward1_K', n_head/num_chip*d_head, d_model, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in2_K_tbuffer = tbuffer('in2_K', n_head/num_chip*d_head*batch*word)
    add_node(node_dict, in2_K_tbuffer)
    
    add_edge(edge_dict, in1_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w2_K_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in2_K_tbuffer.name, 'lanes')
    
    
    
    
    
    
    w3_V_tbuffer = tbuffer('w1_V', n_head/num_chip*d_head*d_model*word)  
    add_node(node_dict, w3_V_tbuffer)

    f_tcompute = tcompute('forward1_V', n_head/num_chip*d_head, d_model, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in3_V_tbuffer = tbuffer('in4_V', n_head/num_chip*d_head*batch*word)
    add_node(node_dict, in3_V_tbuffer)
    
    add_edge(edge_dict, in1_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w3_V_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in3_V_tbuffer.name, 'lanes')
    
    
    
    
    sparse = 64
    
    
    
    # layer 2
    f_tcompute = tcompute('forward2', n_head/num_chip*batch, d_head, sparse, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in3_tbuffer = tbuffer('in3', n_head/num_chip*batch*sparse*word)
    add_node(node_dict, in3_tbuffer)
    
    add_edge(edge_dict, in2_Q_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, in2_K_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, f_tcompute.name, in3_tbuffer.name, 'lanes')
    
    
    
    
    
    
    
    
    # layer 3
    f_tcompute = tcompute('forward3', n_head/num_chip*batch, sparse, d_head, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in4_tbuffer = tbuffer('in4', n_head/num_chip*d_head*batch*word)
    add_node(node_dict, in4_tbuffer)

    add_edge(edge_dict, in3_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, in3_V_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, f_tcompute.name, in4_tbuffer.name, 'lanes')
        
    
    
    
    
    
    
    # layer 4
    w4_tbuffer = tbuffer('w4', d_model/num_chip*d_model*word)  
    add_node(node_dict, w4_tbuffer)
    
    f_tcompute = tcompute('forward4', d_model, d_model/num_chip, batch, 1, 1, -1, True)
    add_node(node_dict, f_tcompute)
    
    in5_tbuffer = tbuffer('in5', d_model*batch*word)
    add_node(node_dict, in5_tbuffer)
    
    add_edge(edge_dict, in4_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w4_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in5_tbuffer.name, 'lanes')
    
    
    
    tile_partition_layer_5 = 8/num_chip
    
    # layer 5
    w5_tbuffer = tbuffer('w5', d_hidden/num_chip*d_model*word/tile_partition_layer_5)  
    add_node(node_dict, w5_tbuffer)
    
    f_tcompute = tcompute('forward5', d_hidden/num_chip, d_model, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in6_tbuffer = tbuffer('in6', d_hidden/num_chip*batch*word)
    add_node(node_dict, in6_tbuffer)
    
    add_edge(edge_dict, in5_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w5_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in6_tbuffer.name, 'lanes')
    
    
    
    
    
    
    
    tile_partition_layer_6 = 8/num_chip
    
    # layer 6
    w6_tbuffer = tbuffer('w6', d_hidden/num_chip*d_model*word/tile_partition_layer_6)  
    add_node(node_dict, w6_tbuffer)
    
    f_tcompute = tcompute('forward6', d_model, d_hidden/num_chip, batch, 1, 1, -1, True)
    add_node(node_dict, f_tcompute)
    
    in7_tbuffer = tbuffer('in7', d_model*batch*word)
    add_node(node_dict, in7_tbuffer)
    
    add_edge(edge_dict, in6_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w6_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in7_tbuffer.name, 'lanes')
    
    
    
        
        
     
        
        
        
        
        
        
        
    
    
    
    
    
    
    
            
   
    
    
    # create reverse edge_dict
    for node1 in edge_dict.keys():
        for node2, label in edge_dict[node1]:
            if node2 in reverse_edge_dict.keys():
                reverse_edge_dict[node2].append([node1, label])
            else:
                reverse_edge_dict[node2] = [[node1, label]]



    bfs(node_dict, edge_dict, reverse_edge_dict)



    
    
    
    
    
            
    Nb = 0
    Nb_cin = []
    Nb_cout = []
    Nb_dim = []
    TSb = []
    D = []
    
    Nc = 0
    mkn = {}
    Nc_name = []
    M = []
    K = []
    N = []
    AllReduce = []
    
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
            
            if node.allreduce == True:
                AllReduce.append(node_dict[edge_dict[node.name][0][0]][0].tenor_size)
            else:
                AllReduce.append(0)
                
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
                        D.append(node.downstream_dict[key][0])
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
    
    
    print('PCU_lim', PCU_lim)
    print('PMU_lim', PMU_lim)
    print('Cap', Cap)
    print('VecWidth', VecWidth)
    print('StageWidth', StageWidth)
    print('Freq', Freq)
    print('DRAM_BW', DRAM_BW)
    print('Nb', Nb)
    print('Nb_cin', Nb_cin)
    print('Nb_cout', Nb_cout)
    print('Nb_dim', Nb_dim)
    print('TSb', TSb)
    print('D', D)
    print('Nc', Nc)
    print('Nc_name', Nc_name)
    print('M', M)
    print('K', K)
    print('N', N)
    print('AllReduce', AllReduce)
    print('Nd', Nd)
    print('Nd_cout', Nd_cout)
    print('Nd_dim', Nd_dim)
    print('TSd', TSd)
    
    
    print()
    print()
    print()
    print()
    print()
    
    
    
    
    TSb = np.array(TSb)
    AllReduce = np.array(AllReduce)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    Flop = 0
    for i in range(len(M)):
        m = M[i]
        k = K[i]
        n = N[i]
        
        if k == -1:
            Flop += m * n
        else:
            Flop += 2 * m * k * n

    print('FLOP', Flop)
        
        
        
        
    
    if num_chip == 8:
        Intermediate = 0
    else:   
        Intermediate = d_model*batch*word/num_chip
    
    
    
    model = gp.Model()
    model.params.NonConvex = 2
    model.Params.Threads = 40
    
    
    
    
    Config = model.addMVar(Nc, name='Config', vtype=gp.GRB.INTEGER, lb=0)
    Ab1 = model.addMVar((Nb, C), name='Ab1', vtype=gp.GRB.BINARY) # on-chip
    Ab2 = model.addMVar((Nb, C), name='Ab2', vtype=gp.GRB.BINARY) # to/from DRAM
    Ac = model.addMVar((Nc, C), name='Ac', vtype=gp.GRB.BINARY)
    Ad = model.addMVar((Nd, C), name='Ad', vtype=gp.GRB.BINARY)
    Par_lane = model.addMVar(Nc, name='Par_lane', vtype=gp.GRB.INTEGER, lb=1)
    Par_stage = model.addMVar(Nc, name='Par_stage', vtype=gp.GRB.INTEGER, lb=1)
    Par_total = model.addMVar(Nc, name='Par_total', vtype=gp.GRB.INTEGER, lb=1)
    
    num_PMU_per_buffer1 = model.addMVar(Nb, name='num_PMU_per_buffer1', vtype=gp.GRB.INTEGER, lb=1)
    num_PMU_per_buffer2 = model.addMVar(Nb, name='num_PMU_per_buffer2', vtype=gp.GRB.INTEGER, lb=1)
    num_PMU_per_DRAMbuffer = model.addMVar(Nd, name='num_PMU_per_DRAMbuffer', vtype=gp.GRB.INTEGER, lb=1)
    
    Partb = model.addMVar(Nb, name='Partb', vtype=gp.GRB.INTEGER, lb=1)
    Partd = model.addMVar(Nd, name='Partd', vtype=gp.GRB.INTEGER, lb=1)
    
    Cycle = model.addMVar(Nc, name='Cycle', vtype=gp.GRB.INTEGER, lb=0)
    DRAM_Latency = model.addMVar(C, name='DRAM_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
    PCIE_Latency = model.addMVar(C, name='PCIE_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
    Compute_Latency = model.addMVar(C, name='Compute_Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
    Latency = model.addMVar(C, name='Latency', vtype=gp.GRB.CONTINUOUS, lb=0)
    
    
    


    
    
    
    # compute assignment   
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
    
    
    t3 = np.ones((Nc))
    for i in range(C):
        model.addConstr(t3 @ Ac[:, i] >= 1)
    
    
    
    
    
    
    
    # PCU limits
    for i in range(Nc):
        model.addConstr(Par_lane[i] @ Par_stage[i] == Par_total[i])
        
        
    for i in range(C):
        model.addConstr(Par_total @ Ac[:, i] <= PCU_lim)
    
    
    
    
    
    
    
    
    # buffer assignment
    for i in range(Nb):
        cin_idx = Nc_dict[Nb_cin[i]]
        cout_idx = Nc_dict[Nb_cout[i]]
        
        for j in range(C):
            t1 = model.addVar(vtype=gp.GRB.BINARY)
            t2 = model.addVar(vtype=gp.GRB.BINARY)
            t3 = model.addVar(vtype=gp.GRB.BINARY)
            t4 = model.addVar(vtype=gp.GRB.BINARY)
            model.addConstr(t1 == gp.and_(Ac[cin_idx, j].tolist()[0], Ac[cout_idx, j].tolist()[0]))
            model.addConstr(t2 == gp.or_(Ac[cin_idx, j].tolist()[0], Ac[cout_idx, j].tolist()[0]))
            model.addConstr(t3 == 1 - t1)
            model.addConstr(t4 == gp.and_(t3, t2))
            
            
            model.addConstr((t1 == 1) >> (Ab1[i, j].tolist()[0] == 1))
            model.addConstr((t1 == 0) >> (Ab1[i, j].tolist()[0] == 0))
            
            
            model.addConstr((t4 == 1) >> (Ab2[i, j].tolist()[0] == 1))
            model.addConstr((t4 == 0) >> (Ab2[i, j].tolist()[0] == 0))
    
    
    for i in range(Nd):
        cout_idx = Nc_dict[Nd_cout[i]]
        
        for j in range(C):
            model.addConstr((Ac[cout_idx, j].tolist()[0] == 1) >> (Ad[i, j].tolist()[0] == 1))
            model.addConstr((Ac[cout_idx, j].tolist()[0] == 0) >> (Ad[i, j].tolist()[0] == 0))
    
    
    
    
    
    
    # buffer partition
    for i in range(Nb):
        cout_idx = Nc_dict[Nb_cout[i]]
        
        if Nb_dim[i] == 'lanes':
            model.addConstr(Partb[i] == Par_lane[cout_idx])
        elif Nb_dim[i] == 'stages':
            model.addConstr(Partb[i] * VecWidth >= StageWidth * Par_stage[cout_idx])
    

    for i in range(Nd):
        cout_idx = Nc_dict[Nd_cout[i]]
        
        if Nd_dim[i] == 'lanes':
            model.addConstr(Partd[i] == Par_lane[cout_idx])
        elif Nd_dim[i] == 'stages':
            model.addConstr(Partd[i] * VecWidth >= StageWidth * Par_stage[cout_idx])
            
            
            
            
            
            
            
    
            
            
            
    
    # PMU limits
    tmpb1 = model.addMVar(Nb, name='tmpb1', vtype=gp.GRB.INTEGER, lb=1)
    tmpb2 = model.addMVar(Nb, name='tmpb2', vtype=gp.GRB.INTEGER, lb=1)
        
    for i in range(Nb):
        model.addConstr(tmpb1[i] @ Partb[i] * Cap >= TSb[i] * D[i])
        model.addConstr(num_PMU_per_buffer1[i] >= tmpb1[i] @ Partb[i])
        
        model.addConstr(tmpb2[i] @ Partb[i] * Cap >= TSb[i])
        model.addConstr(num_PMU_per_buffer2[i] >= tmpb2[i] @ Partb[i])
    
    
    
    tmpd = model.addMVar(Nd, name='tmpd', vtype=gp.GRB.INTEGER, lb=1)
    
    for i in range(Nd):
        model.addConstr(tmpd[i] @ Partd[i] * Cap >= TSd[i])
        model.addConstr(num_PMU_per_DRAMbuffer[i] >= tmpd[i] @ Partd[i])
        
    
    for i in range(C):
        model.addConstr(num_PMU_per_buffer1 @ Ab1[:, i] 
                        + num_PMU_per_buffer2 @ Ab2[:, i]
                        + num_PMU_per_DRAMbuffer @ Ad[:, i] <= PMU_lim)
                        
                        
        
    
    
    
    # performance
    for i in range(Nc):
        m = M[i]
        k = K[i]
        n = N[i]
    
        if k == -1:
            model.addConstr(Par_stage[i] == 1)
        
            t1 = model.addMVar(1, vtype=gp.GRB.INTEGER, lb=1)
            model.addConstr(t1[0] @ Par_lane[i] * VecWidth >= m)
            model.addConstr(Cycle[i] >= t1 * n)
            
        elif n == 1:
            model.addConstr(Par_stage[i] == 1)

            t1 = model.addMVar(1, vtype=gp.GRB.INTEGER, lb=1)
            model.addConstr(t1[0] @ Par_lane[i] * VecWidth >= m)
            model.addConstr(Cycle[i] >= t1 * k)
        
        else:
            t1 = model.addMVar(1, vtype=gp.GRB.INTEGER, lb=1)
            t2 = model.addMVar(1, vtype=gp.GRB.INTEGER, lb=1)
            model.addConstr(t1[0] @ Par_lane[i] * VecWidth >= m)
            model.addConstr(t2[0] @ Par_stage[i] * StageWidth >= n)
            model.addConstr(Cycle[i] >= t1[0] @ t2[0] * k)
            
    
    
    for i in range(C):
        t1 = model.addMVar(Nc, vtype=gp.GRB.INTEGER, lb=0)
        t2 = model.addMVar(1, vtype=gp.GRB.CONTINUOUS, lb=0)
        
        for j in range(Nc):
            model.addConstr(t1[j] == Cycle[j] @ Ac[j, i])
            
        model.addConstr(t2[0].tolist()[0] == gp.max_(t1[j].tolist()[0] for j in range(Nc)))
        model.addConstr(Compute_Latency[i] == t2[0] / Freq)
        model.addConstr(DRAM_Latency[i] == (TSb @ Ab2[:, i]) / DRAM_BW)
        model.addConstr(PCIE_Latency[i] == (AllReduce @ Ac[:, i]) / PCIE_BW)
        model.addConstr(Latency[i].tolist()[0] == gp.max_(Compute_Latency[i].tolist()[0], DRAM_Latency[i].tolist()[0], PCIE_Latency[i].tolist()[0]))

    
    
    
    
    model.setObjective(np.ones((C)) @ Latency + Intermediate / PCIE_BW, gp.GRB.MINIMIZE)
    model.optimize()
    


    
    for v in model.getVars():
        print(v.varName, v.x)
    
    print('Latency:', model.objVal)
    
    
    
    
    