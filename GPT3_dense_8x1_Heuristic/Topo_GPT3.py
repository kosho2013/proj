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

    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--par', type=str)
    args = parser.parse_args()
    
    
    
    par = int(args.par)
    

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
    num_chip = 8





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
    
    
    
    
    
    
    # layer 2
    f_tcompute = tcompute('forward2', n_head/num_chip*batch, d_head, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in3_tbuffer = tbuffer('in3', n_head/num_chip*batch*batch*word)
    add_node(node_dict, in3_tbuffer)
    
    add_edge(edge_dict, in2_Q_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, in2_K_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, f_tcompute.name, in3_tbuffer.name, 'lanes')
    
    
    
    
    
    
    
    
    # layer 3
    f_tcompute = tcompute('forward3', n_head/num_chip*batch, batch, d_head, 1, 1, -1, False)
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
    
    
    
    
    
    # layer 5
    w5_tbuffer = tbuffer('w5', d_hidden/num_chip*d_model*word)  
    add_node(node_dict, w5_tbuffer)
    
    f_tcompute = tcompute('forward5', d_hidden/num_chip, d_model, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in6_tbuffer = tbuffer('in6', d_hidden/num_chip*batch*word)
    add_node(node_dict, in6_tbuffer)
    
    add_edge(edge_dict, in5_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w5_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in6_tbuffer.name, 'lanes')
    
    
    
    
    
    
    
    # layer 6
    w6_tbuffer = tbuffer('w6', d_hidden/num_chip*d_model*word)  
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
        
        
        
        
    
    
    
    
    configs = []
    curr_config = []
    PCU_used = 0
    PMU_used = 0
    
    
    
    
    for i in range(0, len(Nc_name)):
        node = Nc_name[i]
        
        
        additional_PCU = 0
        additional_PMU = 0
        
        
        if par == 1:
            lanes_par = 1
            stages_par = 1
        else:                    
            lanes_par = min(math.ceil(node_dict[node][0].m / LANES) + 1, par)
            stages_par = min(math.ceil(node_dict[node][0].n / STAGES) + 1, par)
            
            
        
        
        node_dict[node][0].update_compute_stitching(lanes_par, stages_par)
        add_node(node_dict, node_dict[node][0])
        additional_PCU = node_dict[node][0].num
        
        for upstream_buffer, label in reverse_edge_dict[node]:
            if label == 'lanes':
                incoming_buffer_node = node_dict[upstream_buffer][0]
                par_factor = math.ceil(LANES * lanes_par / LANES)
                incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                add_node(node_dict, incoming_buffer_node)
                additional_PMU += incoming_buffer_node.downstream_dict[node][2]
                
            elif label == 'stages':
                incoming_buffer_node = node_dict[upstream_buffer][0]
                par_factor = math.ceil(STAGES * stages_par / LANES)
                incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                add_node(node_dict, incoming_buffer_node)
                additional_PMU += incoming_buffer_node.downstream_dict[node][2]
        
        
        if PCU_used + additional_PCU <= PCU_lim and PMU_used + additional_PMU <= PMU_lim:
            curr_config.append(node)
            PCU_used += additional_PCU
            PMU_used += additional_PMU
        else:
            # start new config, depth = 1
            PCU_used = 0
            PMU_used = 0
            additional_PCU = 0
            additional_PMU = 0
            
            
            if par == 1:
                lanes_par = 1
                stages_par = 1
            else:                    
                lanes_par = min(math.ceil(node_dict[node][0].m / LANES) + 1, par)
                stages_par = min(math.ceil(node_dict[node][0].n / STAGES) + 1, par)
            
            
            
            node_dict[node][0].update_compute_stitching(lanes_par, stages_par)
            add_node(node_dict, node_dict[node][0])
            additional_PCU = node_dict[node][0].num
            
            for upstream_buffer, label in reverse_edge_dict[node]:
                if label == 'lanes':
                    incoming_buffer_node = node_dict[upstream_buffer][0]
                    par_factor = math.ceil(LANES * lanes_par / LANES)
                    incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                    incoming_buffer_node.update_depth_to_one(node)
                    add_node(node_dict, incoming_buffer_node)
                    additional_PMU += incoming_buffer_node.downstream_dict[node][2]
                    
                elif label == 'stages':
                    incoming_buffer_node = node_dict[upstream_buffer][0]
                    par_factor = math.ceil(STAGES * stages_par / LANES)
                    incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                    incoming_buffer_node.update_depth_to_one(node)
                    add_node(node_dict, incoming_buffer_node)
                    additional_PMU += incoming_buffer_node.downstream_dict[node][2]
                
                
            if PCU_used + additional_PCU <= PCU_lim and PMU_used + additional_PMU <= PMU_lim:    
                configs.append(curr_config)
                curr_config = [node]
                PCU_used = additional_PCU
                PMU_used = additional_PMU
            else:
                print(node)
                print()
                print()
                print()
                raise Exception('wrong!!!!!!!!!!!!!!!!!!')
        
    
    
    
    plot(graph, 'GPT3_'+str(par), node_dict, edge_dict)
    
    
    
    if num_chip == 8:
        Intermediate = 0
    else:   
        Intermediate = d_model*batch*word/num_chip
    
    
    
    
    
    
    

                
    configs.append(curr_config)
    curr_config = []
    
    
    config_dict = {}
    for i in range(len(configs)):
        for node in configs[i]:
            config_dict[node] = i
    
    
    
    cycles = []
    allreduce_per_config = []
    for config in configs:
        cycle = -1
        tmp = 0
        
        for node in config:
            cycle = max(cycle, node_dict[node][0].cycles)
            tmp += AllReduce[Nc_dict[node]]
            
        cycles.append(cycle)
        allreduce_per_config.append(tmp)
    
    
    from_DRAM = [0 for i in range(len(configs))]
    to_DRAM = [0 for i in range(len(configs))]
    
    for i in range(len(Nb_cin)):
        if config_dict[Nb_cin[i]] != config_dict[Nb_cout[i]]:
            from_DRAM[config_dict[Nb_cout[i]]] += TSb[i]
            to_DRAM[config_dict[Nb_cin[i]]] += TSb[i]
            
            
    
    
    for i in range(len(from_DRAM)):
        cycles[i] = max(cycles[i] / Freq, (from_DRAM[i] + to_DRAM[i]) / DRAM_BW, allreduce_per_config[i] / PCIE_BW)
    
    
    
    print(allreduce_per_config)
    
    Bm = sum(from_DRAM + to_DRAM)
    Bn = sum(AllReduce) + Intermediate
    Latency = sum(cycles)
    
    
    
    
    Throughput = Flop / Latency
    Compute = 2 * PCU * LANES * STAGES * FREQ
    OI = Flop / Bm
    MI = Bm / Bn
    NI = Flop / Bn
    Hardware_OI = Compute / DRAM_BW
    Hardware_MI = DRAM_BW / PCIE_BW
    Hardware_NI = Compute / PCIE_BW
    
        
        
        
        
    print('OI', OI)
    print('MI', MI)
    print('NI', NI)
    print('Latency', Latency)
    print('Throughput', Throughput)
    print('sample/s/layer', 1e9 / Latency * 8/num_chip)
    
    
    
    
    
    if MI < Hardware_MI and OI * MI < Hardware_OI * Hardware_MI:
        print('Network Bound, Peak', OI * MI * PCIE_BW)
    elif MI > Hardware_MI and OI < Hardware_OI:
        print('Memory Bound, Peak', OI * DRAM_BW)
    elif OI > Hardware_OI and OI * MI > Hardware_OI * Hardware_MI:
        print('Compute Bound, Peak', Compute)
        
        
    
    
    print('Hardware_OI', Hardware_OI)
    print('Hardware_MI', Hardware_MI)
    print('Hardware_NI', Hardware_NI)
    
    
    print()
    print()
    print()
    
    
    
    for i in range(len(configs)):
        computes = []
        f = open('HLT/par'+str(par)+'/config'+str(i)+'.txt', 'w')  
        
        for compute in configs[i]:
            computes.append((compute, node_dict[compute][0].topo_num))
        computes.sort(key = lambda x: x[1])
        
        
        
        num_PCU = 0
        num_PMU = 0
        
        for compute, _ in computes:
            num_lane = int(node_dict[compute][0].lanes/VecWidth)
            num_stage = int(node_dict[compute][0].stages/StageWidth)
            num_PCU += num_lane * num_stage
            
            m = int(node_dict[compute][0].m)
            k = int(node_dict[compute][0].k)
            n = int(node_dict[compute][0].n)
            
            f.write('compute '+node_dict[compute][0].name+' m '+str(m)+' k '+str(k)+' n '+str(n)+' lanes '+str(num_lane)+' stages '+str(num_stage)+'\n')

            for buffer, connection in reverse_edge_dict[compute]:
                tensor_size = int(node_dict[buffer][0].tenor_size)
                depth = 1
                partition = int(node_dict[buffer][0].downstream_dict[compute][1])
                num = int(math.ceil(((depth * tensor_size) / partition) / CAPACITY) * partition)
                num_PMU += num
                    
                f.write('buffer '+node_dict[buffer][0].name+' connection '+connection+' num '+str(num)+'\n')
                
            
         
        f.write('PCU '+str(num_PCU)+'\n')
        f.write('PMU '+str(num_PMU)+'\n')
        f.close()
    
    