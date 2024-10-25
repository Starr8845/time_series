from layers.mtgnn_layer import *

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_nodes = args.enc_in
        self.in_dim = args.in_dim
        self.pred_len = args.pred_len
        self.gnn = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=self.num_nodes, device='cuda',
                         predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=args.subgraph_size, node_dim=32,
                         dilation_exponential=1, conv_channels=16, residual_channels=16, skip_channels=64,
                         end_channels=128, seq_length=args.seq_len, in_dim=self.in_dim, out_dim=self.pred_len * self.in_dim,
                         layers=3, propalpha=0.05,
                         tanhalpha=3, layer_norm_affline=True)
        self.hid_size = 128
        self.memory = nn.Parameter(torch.FloatTensor(size=(args.mem_num, 128)))  
        nn.init.xavier_normal_(self.memory, gain=nn.init.calculate_gain("relu"))
        # self.projector = nn.Linear(end_channels, end_channels)
        self.final_linear = nn.Linear(256, self.pred_len)
        # self.final_linear = nn.Linear(128, self.pred_len)

    def memory_enhance(self, repres, get_attention=False):
        attention = torch.einsum("nd,fd->nf", repres, self.memory)
        m = nn.Softmax(dim=1)
        attention = m(attention)
        output = torch.einsum("nf,fd->nd", attention, self.memory)
        # 对应loss constraints的计算
        values, indices = attention.topk(2, dim=1, largest=True, sorted=True)
        largest = self.memory[indices[:,0].squeeze()]
        second_largest = self.memory[indices[:,1].squeeze()]
        distance1 = torch.linalg.vector_norm(repres.detach()-largest, dim=1, ord=2)/(self.hid_size)
        distance2 = torch.linalg.vector_norm(repres.detach()-second_largest, dim=1, ord=2)/(self.hid_size)
        loss_constraint = torch.mean(distance1)
        temp = (distance1-distance2+1e-3)<0
        if torch.sum(temp)>0:
            loss2 = torch.mean((distance1-distance2+1e-3)[temp])
            loss_constraint+=loss2
        loss_constraint += torch.linalg.matrix_norm(self.memory)
        repres = torch.cat((output, repres), dim=1)
        return repres, loss_constraint, None
        same_proto_mask = ((indices[:,0].reshape(-1,1) - indices[:,0].reshape(1,-1))==0)
        if get_attention:
            return repres, loss_constraint, same_proto_mask, attention
        return repres, loss_constraint, same_proto_mask

    def get_repre(self, x):
        # x: [Batch, Input length, Channel]
        batch_size, input_length, variable_num = x.shape[0], x.shape[1], x.shape[2]
        B, L, N = x.shape
        x = x.reshape(B, L, self.num_nodes, self.in_dim).permute(0, 3, 2, 1)
        repres = self.gnn.get_repre(x)
        repres = repres.reshape(-1, repres.shape[2])
        repres, loss_constraint, same_proto_mask = self.memory_enhance(repres, False)
        repres = repres.view(batch_size, variable_num, -1)
        return repres, loss_constraint, same_proto_mask
        
    def predict(self, repres_enhanced):
        result = self.final_linear(repres_enhanced).permute(0,2,1) # [Batch, Output length, Channel]
        return result

    def forward(self, x):
        repres, _, _ = self.get_repre(x)
        result = self.predict(repres)
        return result
    

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = nn.Parameter(torch.arange(self.num_nodes), requires_grad=False)

    def get_repre(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        repres = self.end_conv_2(x)
        # repres: [bs, d, var_num, 1]

        repres = repres.squeeze(3).permute(0,2,1)
        return repres

    