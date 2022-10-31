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
        self.downstream_dict[downstream_buffer_name] = [depth, 1]
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
            self.num += math.ceil(((d * self.tenor_size) / part) / CAPACITY) * part
    
    def update_buffer_partitioning(self, partition, downstream_buffer_name):
        self.downstream_dict[downstream_buffer_name][1] = partition
        
        self.num = 0
        for key in self.downstream_dict.keys():
            d = self.downstream_dict[key][0]
            part = self.downstream_dict[key][1]
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
        
        for key, [d, part] in node.downstream_dict.items():            
            label += key+', d: '+str(d)+', part: '+str(part)+'\n'
        
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
    for i in range(1, total_layer+1):
        for _, [node, _] in node_dict.items():
                if isinstance(node, tcompute):
                    if check_layer_num(node) == i:
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


    
    workload = 'resnet18'
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
    
    
    
    
    
    
    




    batch = [2]
    
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
            
        

        
        # forward loop
        for i in range(1, total_layer+1):
            layer = layer_dict[i]
  
            layer_num = int(layer[1]['layer_num'])
            layer_type = str(layer[1]['layer_type'])
            sparsity = str(layer[1]['sparsity'])            
            m = layer[1]['m']
            k = layer[1]['k']
            n = layer[1]['n']
            input_size = layer[1]['input_size']
            output_size = layer[1]['output_size']
            from_dram = str(layer[1]['from_dram']) 
            
            m = convert(m, ba)
            k = convert(k, ba)
            n = convert(n, ba)
            input_size = convert(input_size, ba)
            output_size = convert(output_size, ba)
            
            
            if layer_type == 'gemm' or layer_type == 'conv':
                if from_dram == 'yes':
                    w_tbuffer = tbuffer('w'+str(layer_num), m*k*word)  
                    add_node(node_dict, w_tbuffer)
                    
                    
                    in_tbuffer = tbuffer('in'+str(layer_num), input_size*word)
                    add_node(node_dict, in_tbuffer)
                    
                    
                    
                    f_compute = tcompute('forward'+str(layer_num)+'_'+layer_type, m, k, n, 1, 1, -1)
                    add_node(node_dict, f_compute)
                    add_edge(edge_dict, w_tbuffer.name, f_compute.name, 'lanes')
                    add_edge(edge_dict, in_tbuffer.name, f_compute.name, 'stages')
                    
                    
                    out_tbuffer = tbuffer('in'+str(layer_num+1), output_size*word)
                    add_node(node_dict, out_tbuffer)
                    add_edge(edge_dict, f_compute.name, out_tbuffer.name, 'lanes')
                
                elif from_dram == 'no':
                    w_tbuffer = tbuffer('w'+str(layer_num), m*k*word)  
                    add_node(node_dict, w_tbuffer)
                    
                    
                    
                    
                    f_compute = tcompute('forward'+str(layer_num)+'_'+layer_type, m, k, n, 1, 1, -1)
                    add_node(node_dict, f_compute)
                    add_edge(edge_dict, w_tbuffer.name, f_compute.name, 'lanes')
                    add_edge(edge_dict, 'in'+str(layer_num), f_compute.name, 'stages')
                    
                    
                    
                    out_tbuffer = tbuffer('in'+str(layer_num+1), output_size*word)
                    add_node(node_dict, out_tbuffer)
                    add_edge(edge_dict, f_compute.name, out_tbuffer.name, 'lanes')
                    
                else:
                    raise Exception('Wrong from_dram!')
            elif layer_type == 'pooling' or layer_type == 'batchnorm' or layer_type == 'add' or layer_type == 'softmax':
                f_compute = tcompute('forward'+str(layer_num)+'_'+layer_type, m, k, n, 1, 1, -1)
                add_node(node_dict, f_compute)
                add_edge(edge_dict, 'in'+str(layer_num), f_compute.name, 'lanes')
                
                if layer_type == 'add':
                    add_edge(edge_dict, 'in'+str(layer_num-4), f_compute.name, 'lanes')
                
                
                
                out_tbuffer = tbuffer('in'+str(layer_num+1), output_size*word)
                add_node(node_dict, out_tbuffer)
                add_edge(edge_dict, f_compute.name, out_tbuffer.name, 'lanes')
                
            
            elif layer_type == 'loss':
                f_compute = tcompute('forward'+str(layer_num)+'_'+layer_type, m, k, n, 1, 1, -1) 
                add_node(node_dict, f_compute)
                add_edge(edge_dict, 'in'+str(layer_num), f_compute.name, 'lanes')
                
                dataGradient_tbuffer = tbuffer('dataGradient'+str(layer_num), output_size*word)
                    
                add_node(node_dict, dataGradient_tbuffer)
                add_edge(edge_dict, f_compute.name, dataGradient_tbuffer.name, 'lanes')
                    
                
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
            input_size = layer[1]['input_size']
            output_size = layer[1]['output_size']
            from_dram = str(layer[1]['from_dram']) 
            
            m = convert(m, ba)
            k = convert(k, ba)
            n = convert(n, ba)
            input_size = convert(input_size, ba)
            output_size = convert(output_size, ba)
            
            
            if layer_type == 'gemm' or layer_type == 'conv':
                dg_compute = tcompute('backpropDataGradient'+str(layer_num)+'_'+layer_type, k, m, n, 1, 1, -1) 
                    
                add_node(node_dict, dg_compute)
                add_edge(edge_dict, 'w'+str(layer_num), dg_compute.name, 'lanes')
                add_edge(edge_dict, 'dataGradient'+str(layer_num+1), dg_compute.name, 'stages')
                
                if layer_num in [4, 9, 14, 19, 24, 29, 34, 39]:
                    dg_buffer = tbuffer('dataGradient'+str(layer_num)+'_tmp', input_size*word)
                else:
                    dg_buffer = tbuffer('dataGradient'+str(layer_num), input_size*word)
                add_node(node_dict, dg_buffer)
                add_edge(edge_dict, dg_compute.name, dg_buffer.name, 'lanes')
                
                
                wg_compute = tcompute('backpropWeightGradient'+str(layer_num)+'_'+layer_type, m, n, k, 1, 1, -1) 
                add_node(node_dict, wg_compute)
                add_edge(edge_dict, 'in'+str(layer_num), wg_compute.name, 'stages')
                add_edge(edge_dict, 'dataGradient'+str(layer_num+1), wg_compute.name, 'lanes')
                
                
                wg_tbuffer = tbuffer('weightGradient'+str(layer_num), m*k*word)
                add_node(node_dict, wg_tbuffer)
                add_edge(edge_dict, wg_compute.name, wg_tbuffer.name, 'lanes')
            
                    
                wu_compute = tcompute('weightUpdate'+str(layer_num)+'_'+layer_type, m, -1, k, 1, 1, -1)
                add_node(node_dict, wu_compute)
                add_edge(edge_dict, 'weightGradient'+str(layer_num), wu_compute.name, 'lanes')
                
                if layer_num in [4, 9, 14, 19, 24, 29, 34, 39]:
                    tmp_compute = tcompute('backpropDataGradient'+str(layer_num)+'_add', input_size, k, n, 1, 1, -1)
                    add_node(node_dict, tmp_compute)
                    add_edge(edge_dict, dg_buffer.name, tmp_compute.name, 'lanes')
                    add_edge(edge_dict, 'dataGradient'+str(layer_num+5), tmp_compute.name, 'lanes')
                    
                    dg_buffer_final = tbuffer('dataGradient'+str(layer_num), input_size*word)
                    add_node(node_dict, dg_buffer_final)
                    add_edge(edge_dict, tmp_compute.name, dg_buffer_final.name, 'lanes')
                
            elif layer_type == 'pooling' or layer_type == 'batchnorm' or layer_type == 'softmax':
                dg_compute = tcompute('backpropDataGradient'+str(layer_num)+'_'+layer_type, m, k, n, 1, 1, -1) 
                add_node(node_dict, dg_compute)
                if layer_num in [7, 12, 17, 22, 27, 32, 37, 42]:
                    add_edge(edge_dict, 'dataGradient'+str(layer_num+2), dg_compute.name, 'lanes')
                else:
                    add_edge(edge_dict, 'dataGradient'+str(layer_num+1), dg_compute.name, 'lanes')
                

                dg_buffer = tbuffer('dataGradient'+str(layer_num), input_size*word)
                add_node(node_dict, dg_buffer)
                add_edge(edge_dict, dg_compute.name, dg_buffer.name, 'lanes')
            
            elif layer_type == 'loss' or layer_type == 'add':
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


