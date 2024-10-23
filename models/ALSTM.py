import torch
import torch.nn as nn


class ALSTMModel(nn.Module): 
    def __init__(self, configs, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)

    def get_repre(self, inputs):
        # x: [Batch, Input length, Channel]
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        repres = torch.cat((rnn_out[:, -1, :], out_att), dim=1) # [batch, seq_len, num_directions * hidden_size]
        return repres
    
    def predict(self, repres):
        out = self.fc_out(repres)
        return out[..., 0]
    
    def forward(self, inputs):
        repres = self.get_repre(inputs)
        pred = self.predict(repres)
        return pred


    # 这里的predict函数, 可能需要把linear 2维度 改成三维   ALSTM这种bottleneck 是不是太小了   不确定