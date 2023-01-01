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
        self.downstream_dict[downstream_buffer_name] = [depth, 1, 0, 0] # d: depth, part, num (non-tile), num (tile) 
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[key][2] = math.ceil(((d * self.tenor_size) / part) / Cap) * part
            if self.name.startswith('w'):
                self.downstream_dict[key][3] = math.ceil(((d * self.tenor_size) / part) / Cap) * part
            else:
                self.downstream_dict[key][3] = math.ceil(((d * self.tenor_size / Tile_factor) / part) / Cap) * part
                
            self.num += self.downstream_dict[key][2]
    
    def update_buffer_partitioning(self, partition, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name][1] = partition
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[downstream_buffer_name][2] = math.ceil(((d * self.tenor_size) / part) / Cap) * part
            if self.name.startswith('w'):
                self.downstream_dict[key][3] = math.ceil(((d * self.tenor_size) / part) / Cap) * part
            else:
                self.downstream_dict[key][3] = math.ceil(((d * self.tenor_size / Tile_factor) / part) / Cap) * part
                
            self.num += self.downstream_dict[key][2]
            
    def update_depth_to_one(self, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name][0] = 1
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.downstream_dict[key][2] = math.ceil(((d * self.tenor_size) / part) / Cap) * part
            if self.name.startswith('w'):
                self.downstream_dict[key][3] = math.ceil(((d * self.tenor_size) / part) / Cap) * part
            else:
                self.downstream_dict[key][3] = math.ceil(((d * self.tenor_size / Tile_factor) / part) / Cap) * part
            
            self.num += self.downstream_dict[key][2]
        
            
            
class tcompute:
    def __init__(self, name, m, k, n, lanes_par, stages_par, topo_num, allreduce): 
        self.name = name
        self.m = m
        self.k = k
        self.n = n
        
        self.lanes_par = lanes_par
        self.stages_par = stages_par
        self.lanes = VecWidth * self.lanes_par
        self.stages = StageWidth * self.stages_par
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
        self.lanes = VecWidth * self.lanes_par
        self.stages = StageWidth * self.stages_par
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

    # system parameters   
    datatype = 'BF16'
    word = 2 
 
    PMU_lim = 640
    PCU_lim = 640
    Cap = 524288 # B
    VecWidth = 64 # B
    VecWidth = int(VecWidth / word) # number of data
    StageWidth = 6 # number of data
    Freq = 1.25 # GHz
    DRAM_BW = 100.0 # GB/s
    PCIE_BW = 25.0 # GB/s
    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--num_chip', type=str)
    parser.add_argument('--par', type=str)
    args = parser.parse_args()    
    num_chip = int(args.num_chip)
    par = int(args.par)
    
    Tile_factor = 16
    
    
    
    # construct dataflow graph
    d_model = 12288
    d_head = 128
    d_hidden = 49152
    batch = 2048
    
    n_head = 96
    n_layer = 96
    pipeline_stage = int(8 / num_chip)
    





    graph = pydot.Dot(graph_type='digraph')
    edge_dict = {}
    reverse_edge_dict = {}
    node_dict = {}
    
    
        
    
    
    
    
    
    
    # LN1
    inLN1_tbuffer = tbuffer('inLN1', d_model*batch*word)
    add_node(node_dict, inLN1_tbuffer)
    
    f_tcompute = tcompute('LN1', n_head/num_chip*d_model, -1, batch*2, 1, 1, -1, False) # softmax has two passes
    add_node(node_dict, f_tcompute)
    
    in1_tbuffer = tbuffer('in1', d_model*batch*word)
    add_node(node_dict, in1_tbuffer)
    
    add_edge(edge_dict, inLN1_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in1_tbuffer.name, 'lanes')
    
    
    
    
    # forward 1
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
    
    
    
    
    sparse = 32 # for the 2048 dim
    
    
    
    # forward 2
    f_tcompute = tcompute('forward2', n_head/num_chip*batch, d_head, sparse*2, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    inSoftmax_tbuffer = tbuffer('inSoftmax', n_head/num_chip*batch*sparse*2*word)
    add_node(node_dict, inSoftmax_tbuffer)
    
    add_edge(edge_dict, in2_Q_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, in2_K_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, f_tcompute.name, inSoftmax_tbuffer.name, 'lanes')
    
    
    
    
    
    # softmax
    f_tcompute = tcompute('softmax', n_head/num_chip*batch, -1, sparse*2*2, 1, 1, -1, False) # softmax has two passes
    add_node(node_dict, f_tcompute)
    
    in3_tbuffer = tbuffer('in3', n_head/num_chip*batch*sparse*2*word) # attention
    add_node(node_dict, in3_tbuffer)
    
    add_edge(edge_dict, inSoftmax_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in3_tbuffer.name, 'lanes')
    
    
    
    
    
    
    # forward 3
    f_tcompute = tcompute('forward3', n_head/num_chip*batch, sparse*2, d_head, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in4_tbuffer = tbuffer('in4', n_head/num_chip*d_head*batch*word)
    add_node(node_dict, in4_tbuffer)

    add_edge(edge_dict, in3_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, in3_V_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, f_tcompute.name, in4_tbuffer.name, 'lanes')
        
    
    
    
    
    
    
    # forward 4
    w4_tbuffer = tbuffer('w4', d_model/num_chip*d_model*word)  
    add_node(node_dict, w4_tbuffer)
    
    if num_chip == 1:
        f_tcompute = tcompute('forward4', d_model, d_model/num_chip, batch, 1, 1, -1, False)
    else:
        f_tcompute = tcompute('forward4', d_model, d_model/num_chip, batch, 1, 1, -1, True)
    add_node(node_dict, f_tcompute)
    
    inLN2_tbuffer = tbuffer('inLN2', d_model*batch*word)
    add_node(node_dict, inLN2_tbuffer)
    
    add_edge(edge_dict, in4_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w4_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, inLN2_tbuffer.name, 'lanes')
    
    
    
    
    
    # LN2
    f_tcompute = tcompute('LN2', d_model, -1, batch*2, 1, 1, -1, False) # softmax has two passes
    add_node(node_dict, f_tcompute)
    
    in5_tbuffer = tbuffer('in5', d_model*batch*word)
    add_node(node_dict, in5_tbuffer)
    
    add_edge(edge_dict, inLN2_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in5_tbuffer.name, 'lanes')
    
    
    
    sparse_big = 384
    sparse_small = 96

    
    # forward 5
    w5_tbuffer = tbuffer('w5', d_hidden/num_chip*sparse_small*2*word)  
    add_node(node_dict, w5_tbuffer)
    
    f_tcompute = tcompute('forward5', d_hidden/num_chip, sparse_small*2, batch, 1, 1, -1, False)
    add_node(node_dict, f_tcompute)
    
    in6_tbuffer = tbuffer('in6', d_hidden/num_chip*batch*word)
    add_node(node_dict, in6_tbuffer)
    
    add_edge(edge_dict, in5_tbuffer.name, f_tcompute.name, 'stages')
    add_edge(edge_dict, w5_tbuffer.name, f_tcompute.name, 'lanes')
    add_edge(edge_dict, f_tcompute.name, in6_tbuffer.name, 'lanes')
    
    
    
    
    
    
    
    
    # forward 6
    w6_tbuffer = tbuffer('w6', d_model*sparse_big*2/num_chip*word)  
    add_node(node_dict, w6_tbuffer)
    
    if num_chip == 1:
        f_tcompute = tcompute('forward6', d_model, sparse_big*2/num_chip, batch, 1, 1, -1, False)
    else:
        f_tcompute = tcompute('forward6', d_model, sparse_big*2/num_chip, batch, 1, 1, -1, True)
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



    
    
    
    
    
    
    # workload parameters      
    Nb = 0
    Nb_cin = []
    Nb_cout = []
    Nb_dim = []
    TSb_notile = []
    TSb_tile = []
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
                        
                        TSb_notile.append(node.tenor_size)
                        TSb_tile.append(node.tenor_size/Tile_factor)
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
        
        
            
    
        
        
        
        
        
    
    
    # Model
    PCU_lim = math.ceil(PCU_lim * 0.8)
    
    
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
            lanes_par = min(math.ceil(node_dict[node][0].m / VecWidth) + 1, par)
            stages_par = min(math.ceil(node_dict[node][0].n / StageWidth) + 1, par)
            
            
        
        
        node_dict[node][0].update_compute_stitching(lanes_par, stages_par)
        add_node(node_dict, node_dict[node][0])
        additional_PCU = node_dict[node][0].num
        
        for upstream_buffer, label in reverse_edge_dict[node]:
            if label == 'lanes':
                incoming_buffer_node = node_dict[upstream_buffer][0]
                par_factor = math.ceil(VecWidth * lanes_par / VecWidth)
                incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                add_node(node_dict, incoming_buffer_node)
                additional_PMU += incoming_buffer_node.downstream_dict[node][3]
                
            elif label == 'stages':
                incoming_buffer_node = node_dict[upstream_buffer][0]
                par_factor = math.ceil(StageWidth * stages_par / VecWidth)
                incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                add_node(node_dict, incoming_buffer_node)
                additional_PMU += incoming_buffer_node.downstream_dict[node][3]
        
        
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
                lanes_par = min(math.ceil(node_dict[node][0].m / VecWidth) + 1, par)
                stages_par = min(math.ceil(node_dict[node][0].n / StageWidth) + 1, par)
            
            
            
            node_dict[node][0].update_compute_stitching(lanes_par, stages_par)
            add_node(node_dict, node_dict[node][0])
            additional_PCU = node_dict[node][0].num
            
            for upstream_buffer, label in reverse_edge_dict[node]:
                if label == 'lanes':
                    incoming_buffer_node = node_dict[upstream_buffer][0]
                    par_factor = math.ceil(VecWidth * lanes_par / VecWidth)
                    incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                    incoming_buffer_node.update_depth_to_one(node)
                    add_node(node_dict, incoming_buffer_node)
                    additional_PMU += incoming_buffer_node.downstream_dict[node][3]
                    
                elif label == 'stages':
                    incoming_buffer_node = node_dict[upstream_buffer][0]
                    par_factor = math.ceil(StageWidth * stages_par / VecWidth)
                    incoming_buffer_node.update_buffer_partitioning(par_factor, node)
                    incoming_buffer_node.update_depth_to_one(node)
                    add_node(node_dict, incoming_buffer_node)
                    additional_PMU += incoming_buffer_node.downstream_dict[node][3]
                
                
            if PCU_used + additional_PCU <= PCU_lim and PMU_used + additional_PMU <= PMU_lim:    
                configs.append(curr_config)
                curr_config = [node]
                PCU_used = additional_PCU
                PMU_used = additional_PMU
            else:
                raise Exception('wrong!!!!!!!!!!!!!!!!!!')
        
    
    
    
    
    
    
    
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
            from_DRAM[config_dict[Nb_cout[i]]] += TSb_notile[i]
            to_DRAM[config_dict[Nb_cin[i]]] += TSb_notile[i]
            
            
    
    
    for i in range(len(from_DRAM)):
        cycles[i] = max(cycles[i] / Freq, (from_DRAM[i] + to_DRAM[i]) / DRAM_BW, allreduce_per_config[i] / PCIE_BW)
    
    

    
    Bm = sum(from_DRAM + to_DRAM)
    if Bm == 0:
        Bm = TSd[0]
        
    Bn = sum(AllReduce) + Intermediate
    Latency = sum(cycles)
    
    
    Flop = 0
    for i in range(len(M)):
        m = M[i]
        k = K[i]
        n = N[i]
        
        if k == -1:
            Flop += m * n
        else:
            Flop += 2 * m * k * n
            
    
    Throughput = Flop / Latency
    Compute = 2 * PCU_lim * VecWidth * StageWidth * Freq
    OI = Flop / Bm
    MI = Bm / Bn
    NI = Flop / Bn
    Hardware_OI = Compute / DRAM_BW
    Hardware_MI = DRAM_BW / PCIE_BW
    Hardware_NI = Compute / PCIE_BW
    
        
        
        
    print('num_chip:', num_chip, 'par:', par)    
    print('# of configs', len(configs))
    print('configs:', configs)
    print('OI', OI)
    print('MI', MI)
    print('NI', NI)
    print('Latency', Latency)
    print('Throughput', Throughput)
    print('sample/s/layer', 1e9 / Latency * pipeline_stage)
    
    
    
    
    
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