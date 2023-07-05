import torch
from torch import nn
import torch.nn.functional as F
import os
import h5py, math


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1. / math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert (loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x))  # b x c x k
        x = self.relu(x)
        return x


class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H * W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)  # 1, 32, 3600
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(
                sigma[node_id, :])  # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        # 1 512 60 60
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        sigma = torch.sigmoid(self.sigma)  # 32 512
        soft_assign = self.gen_soft_assign(x, sigma)  # 1 32 3600, # B x C x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)  # 1 32 512
        for node_id in range(self.node_num):
            aa = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # 1 3600 512
            bb = self.anchor[node_id, :]  # 512
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(
                sigma[node_id, :])  # + eps) 1,3600 512
            c = soft_assign[:, node_id, :]  # 1 3600
            cc = soft_assign[:, node_id, :].unsqueeze(2)  # 1 3600 1 # 1 3600 1
            d = residual.mul(soft_assign[:, node_id, :].unsqueeze(2))  # 1 3600 512
            dd = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1)  # 1 512
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (
                    soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2)  # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1)  # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign


class MutualModule0(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule0, self).__init__()
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

    # graph0: edge, graph1/2: region, assign:edge
    def forward(self, x1_graph1, x1_graph2, x1_graph3, x2_graph1, x2_graph2, x2_graph3, x1_assign, x2_assign):
        m1, m2 = self.corr_matrix(x1_graph1, x1_graph2, x1_graph3, x2_graph1, x2_graph2, x2_graph3)  # 1 512 32
        x1_graph2 = x1_graph2 + m1  # v'_e
        x2_graph2 = x2_graph2 + m2

        x1_graph2 = self.gcn(x1_graph2)
        x1_graph2 = x1_graph2.bmm(x1_assign)  # 1 512 3600 # reprojection
        x1_graph2 = self.conv(x1_graph2.unsqueeze(3)).squeeze(3)  # 1 512 3600

        x2_graph2 = self.gcn(x2_graph2)
        x2_graph2 = x2_graph2.bmm(x2_assign)  # 1 512 3600 # reprojection
        x2_graph2 = self.conv(x2_graph2.unsqueeze(3)).squeeze(3)  # 1 512 3600

        return x1_graph2, x2_graph2

    def corr_matrix(self, x1_graph1, x1_graph2, x1_graph3, x2_graph1, x2_graph2, x2_graph3):
        x_graph = torch.cat((x1_graph3, x2_graph3), 1)

        # edge:1 512 32, region1:1 512 32, region2:1 512 32
        assign1 = x1_graph2.permute(0, 2, 1).contiguous().bmm(x_graph)
        assign1 = F.softmax(assign1, dim=-1)

        assign2 = x2_graph2.permute(0, 2, 1).contiguous().bmm(x_graph)
        assign2 = F.softmax(assign2, dim=-1)

        m1 = assign1.bmm(x1_graph1.permute(0, 2, 1).contiguous())
        m1 = m1.permute(0, 2, 1).contiguous()

        m2 = assign2.bmm(x2_graph1.permute(0, 2, 1).contiguous())
        m2 = m2.permute(0, 2, 1).contiguous()
        return m1, m2


class MutualNet(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=512, num_clusters=8, dropout=0.1):
        super(MutualNet, self).__init__()

        self.dim = dim

        self.x1_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.x2_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.x1_conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.x1_conv1[0].reset_params()

        self.x1_conv2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.x1_conv2[0].reset_params()

        self.x1_conv3 = nn.Sequential(BasicConv2d(self.dim, self.dim // 2, BatchNorm, kernel_size=1, padding=0))
        self.x1_conv3[0].reset_params()

        self.x2_conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.x2_conv1[0].reset_params()

        self.x2_conv2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.x2_conv2[0].reset_params()

        self.x2_conv3 = nn.Sequential(BasicConv2d(self.dim, self.dim // 2, BatchNorm, kernel_size=1, padding=0))
        self.x2_conv3[0].reset_params()

        self.r2e = MutualModule0(self.dim, BatchNorm, dropout)

    def forward(self, x1, x2):
        x1_graph, x1_assign = self.x1_proj0(x1)
        x2_graph, x2_assign = self.x2_proj0(x2)

        x1_graph1 = self.x1_conv1(x1_graph.unsqueeze(3)).squeeze(3)
        x1_graph2 = self.x1_conv2(x1_graph.unsqueeze(3)).squeeze(3)
        x1_graph3 = self.x1_conv3(x1_graph.unsqueeze(3)).squeeze(3)

        x2_graph1 = self.x2_conv1(x2_graph.unsqueeze(3)).squeeze(3)
        x2_graph2 = self.x2_conv2(x2_graph.unsqueeze(3)).squeeze(3)
        x2_graph3 = self.x2_conv3(x2_graph.unsqueeze(3)).squeeze(3)

        x1_g, x2_g = self.r2e(x1_graph1, x1_graph2, x1_graph3, x2_graph1, x2_graph2, x2_graph3, x1_assign, x2_assign)
        x1 = x1 + x1_g.view(x1.size()).contiguous()
        x2 = x2 + x2_g.view(x1.size()).contiguous()

        return x1, x2
