import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GAE, GATv2Conv, GraphSAGE, GENConv, GMMConv, \
    GravNetConv, MessagePassing, global_max_pool, global_add_pool, GAT, GINConv, GINEConv, GraphNorm, SAGEConv, RGATConv,CuGraphGATConv,\
    TopKPooling,TransformerConv,GatedGraphConv,GraphNorm,HypergraphConv
from torch.nn.functional import sigmoid
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch_geometric.utils import add_self_loops,dropout_edge,dropout_node
from tqdm import tqdm
from torch.nn import Conv1d
from torch.nn import init
from torch_geometric.nn import NNConv
from torch_geometric.nn import aggr
from torch_geometric.transforms import TwoHop 
#以此为基准进行修改
#最好的模型   
# 加上边的卷积   
# transform中分别对节点和边做注意力   
# 边的卷积加入顶点信息
#去掉边的残差
# *********************************************

#去掉节点的残差,
#1.完全规范，
#3.去掉上面layernorm
#4.seed改为1

#去掉节点的残差试试
#NENN标准
#加点额外注意力

d_model=512
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v =64  # K(=Q), V的维度 
n_heads = 8     # Multi-Head Attention设置为8

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax,add_self_loops,dropout_node,dropout_edge
from typing import Optional, Tuple, Union

from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
# m>n：升维，将特征进行各种类型的特征组合，提高模型分辨能力
# m<n：降维，去除区分度低的组合特征
#先降维再升维，是跟着自编码器学的
#先升维再降维，是跟着transform学的

from typing import Optional
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

# from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


class HypergraphConv(MessagePassing):
   
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False)
            self.lin2 = Linear(in_channels, heads * out_channels, bias=False)
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False)
            self.lin2 = Linear(in_channels,out_channels, bias=False)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)



    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        
        num_nodes = x.size(0)
        #print(num_nodes)#3,顶点数目

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1
        #print(num_edges)#4，边的数目
        
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x).view(-1, self.heads, self.out_channels)
        hyperedge_attr = self.lin2(hyperedge_attr).view(-1, self.heads, self.out_channels)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            # print(hyperedge_attr)
            hyperedge_attr = self.lin(hyperedge_attr)#边的特征
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            # print(hyperedge_attr)
            x_i = x[hyperedge_index[0]]#顶点特征
            x_j = hyperedge_attr[hyperedge_index[1]]#边的特征
            # print(x_j)
            
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)#顶点与边相连算注意力
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)#对所有连接到同一条边的顶点进行softmax 
                #print(alpha)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)#对所有连接到同一个顶点的边进行softmax
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0
        
        #print(x)#传入顶点特征值
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        #print(out)#顶点的特征传播到边，这里是边的特征，没有使用原始边的特征
        #print(1/0)

        out=out+hyperedge_attr

        #print(hyperedge_index.flip([0]))#下面去上面，上面到下面
        #print(out)#传入顶点转化后的边的特征值
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))
        #print(out)#边的特征值再回到顶点
        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        #第一次传播x为三个顶点的特征值
        #print(x_j)#顶点x的特征值，不过分裂成与边相对应的
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        #print(out)
        return out







class TransformerConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None


        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        #forward1
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        #message
        # self.lin_edge1 = Linear(edge_dim, heads * out_channels)
        self.lin_edge2 = Linear(edge_dim, heads * out_channels)
        self.lin_edge3 = Linear(edge_dim, heads * out_channels)

        
        self.con_head=nn.Linear(self.heads * self.out_channels,out_channels, bias=False)
       
        # self.aggr_call3=aggr.SumAggregation()
       

        #forward2
        # self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
        # if self.beta:
        #     self.lin_beta = Linear(3 * out_channels,out_channels, bias=False)
        # else:
        #     self.lin_beta = self.register_parameter('lin_beta', None)


        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()

        
        # self.lin_edge1.reset_parameters()
        self.lin_edge2.reset_parameters()
        self.lin_edge3.reset_parameters()

        # self.con_key.reset_parameters()
        # self.norm_key.reset_parameters()
        # self.con_mess.reset_parameters()
        self.con_head.reset_parameters()
        # self.fc_mess[0].reset_parameters()
        # self.fc_mess[2].reset_parameters()
        # self.norm_mess.reset_parameters()


        # self.aggr_call1.reset_parameters()
        ## self.aggr_call2.reset_parameters()
        # self.aggr_call3.reset_parameters()
        ## self.atten.reset_parameters()
        # self.norm_aggr.reset_parameters()
        # self.aggr_line.reset_parameters()




        # self.rnn.reset_parameters()
        # self.lin_skip.reset_parameters()
        # if self.beta:
        #     self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
       

        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)


        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        alpha = self._alpha
        self._alpha = None
        
        # x_r = self.lin_skip(x)
        # beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
        # beta = beta.sigmoid()

        # out = beta * x_r + (1 - beta) * out
  

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # edge_attr1 = self.lin_edge1(edge_attr).view(-1, self.heads,self.out_channels)
        edge_attr2 = self.lin_edge2(edge_attr).view(-1, self.heads,self.out_channels)
        edge_attr3 = self.lin_edge3(edge_attr).view(-1, self.heads,self.out_channels)

        key_j=key_j

        alpha1 = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha1 = softmax(alpha1, index, ptr, size_i)
        self._alpha = alpha1
        
        out1=value_j
        out1 = out1 * alpha1.view(-1, self.heads, 1)


        alpha2 = (query_i * edge_attr2).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha2 = softmax(alpha2, index, ptr, size_i)
        
        out2=edge_attr3
        out2 = out2 * alpha2.view(-1, self.heads, 1)


        out=out1+out2
        # out=out1

        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            out=self.con_head(out)
        else:
            out = out.mean(dim=1)


        return out
    # def aggregate(self, inputs, index, ptr, dim_size):

        
       
    #     out=self.aggr_call(inputs, index, ptr=ptr, dim_size=dim_size)
    #     return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')



class GraphTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=11, dropout=0.3, **kwargs):
        super(GraphTransformerBlock, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = TransformerConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim,beta=True,dropout=0,concat=True,)
        self.BatchNorm = nn.BatchNorm1d(out_channels)
        self.conv2=HypergraphConv(in_channels, out_channels)
        self.BatchNorm2 = nn.BatchNorm1d(out_channels)

    def DHT(self, edge_index, batch, add_loops=False):

        num_edge = edge_index.size(1)
        device = edge_index.device

        ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
        edge_to_node_index = torch.arange(0,num_edge,1, device=device).repeat_interleave(2).view(1,-1)
        hyperedge_index = edge_index.T.reshape(1,-1)
        hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long() 

        ### Transform batch of nodes to batch of edges
        edge_batch = hyperedge_index[1,:].reshape(-1,2)[:,0]
        edge_batch = torch.index_select(batch, 0, edge_batch)

        return hyperedge_index, edge_batch
   
    def forward(self, x, edge_index, edge_attr,batch):
        

        x_gat = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x_gat =F.gelu(self.BatchNorm(x_gat))
        
        #把无向图重复的边去掉
        edge_index=edge_index[...,:edge_index.size()[-1] // 2]
        edge_attr = edge_attr[:edge_attr.size()[0] // 2, ...]
        hyperedge_index, edge_batch = self.DHT(edge_index, batch)

        edge_attr2=self.conv2(edge_attr, hyperedge_index,hyperedge_attr=x)
        #把无向图重复的边去掉再加上
        edge_index_reversed = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, edge_index_reversed], dim=1)
        edge_attr=edge_attr.repeat(2, 1)
        edge_attr2=edge_attr2.repeat(2, 1)


        # edge_attr=edge_attr2+edge_attr
        edge_attr=edge_attr2
        edge_attr=F.gelu(self.BatchNorm2(edge_attr))
    
        return  x_gat,edge_attr









class MyNet(nn.Module):
    def __init__(self, emb_dim=256, feat_dim=256, edge_dim=256,temp_dim=256, heads=8, drop_ratio=0, pool='add'):
        super(MyNet, self).__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        # self.layerNorm_graph = nn.LayerNorm(emb_dim*2)
        self.layerNorm_out= nn.LayerNorm(emb_dim*2)



        self.in_node = nn.Linear(46, emb_dim)#54
        self.in_edge = nn.Linear(21, emb_dim)#10



      
        self.conv1 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv2 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv3 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv4 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv5 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        self.conv6 = GraphTransformerBlock(emb_dim, emb_dim, heads=heads, edge_dim=edge_dim)
        


        


        self.graphline = nn.Linear(emb_dim*7, emb_dim*2)
        
       
        self.trans_graph = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim*4),
            nn.GELU(),
            nn.Linear(emb_dim*4, emb_dim*2),
            nn.GELU(),
        )
        self.trans_add = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim*4),
            nn.GELU(),
            nn.Linear(emb_dim*4, emb_dim*2),
        )
        self.trans_out = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim*4),
            nn.GELU(),
            nn.Linear(emb_dim*4, emb_dim*2),
            nn.GELU(),
        )
        # self.trans_weight = nn.Sequential(
        #     nn.Linear(emb_dim*2, 1 ),
        #     nn.Sigmoid(),
            
        # )


        self.out_lin = nn.Sequential(
            nn.Linear(emb_dim*2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim//2),
            nn.GELU(),
            nn.Linear(emb_dim//2,emb_dim//4),
            nn.GELU(),
            nn.Linear(emb_dim//4, emb_dim//8),
            nn.GELU(),
            nn.Linear(emb_dim//8,1),
        )


        # self.dropout= nn.Dropout(self.drop_ratio)
    

        
    def forward(self, data):


        x = data.x
        
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        #先减后加，变成重复的都放在后面
        edge_index = edge_index[..., ::2]
        edge_attr = edge_attr[::2, ...]
        edge_index_reversed = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, edge_index_reversed], dim=1)
        edge_attr=edge_attr.repeat(2, 1)
        

        
        
        h=self.in_node(x)
        h = F.gelu(h)

        edge_attr=self.in_edge(edge_attr)
        edge_attr=F.gelu(edge_attr)

       

        h1,edge_attr=self.conv1(h, edge_index, edge_attr,batch)
        h2,edge_attr= self.conv2(h1, edge_index, edge_attr,batch)
        h3,edge_attr = self.conv3(h2, edge_index, edge_attr,batch)
        h4,edge_attr= self.conv4(h3, edge_index, edge_attr,batch)
        h5,edge_attr =self.conv5(h4, edge_index, edge_attr,batch)
        h6,edge_attr =self.conv6(h5, edge_index, edge_attr,batch)
       

        hhh1=torch.cat([h,h1,h2,h3,h4,h5,h6],dim=-1)
        # hhh1=self.dropout(hhh1)
        hhh1=self.graphline(hhh1)
        # hhh1=self.layerNorm_graph(hhh1)
        concat=F.gelu(hhh1)

        concat=self.trans_graph(concat)

        # weight=self.trans_weight(concat)
        # concat=weight*concat

        add_x=global_add_pool(concat,batch)
        score=torch.sigmoid(self.trans_add(add_x))
        result=torch.mul(score, add_x)
        result=self.layerNorm_out(result)

        result=self.trans_out(result)
        out = self.out_lin(result)


        return out.squeeze()


