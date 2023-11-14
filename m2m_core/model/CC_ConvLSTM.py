import torch.nn as nn
import torch
from .CBAM import ChannelAttentionModule
from torch.nn import Module, Sequential, Conv2d


class LSTM_cell(Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 filter_size,
                 img_size,
                 tensor_dtype=torch.float32,
                 ce_iterations=5,
                 elementwise_affine: bool = True,
                 use_ce: bool = True):
        
        super(LSTM_cell, self).__init__()
        self.tensor_dtype = tensor_dtype
        self.input_size = input_size
        self.filter_size = filter_size
        self.hidden_size = hidden_size

        self.padding = int((filter_size - 1) / 2)  # in this way the output has the same size
        self._forget_bias = 1.0
        self.elementwise_affine = elementwise_affine

        self.use_ce = use_ce
        # print('CE BLOCK:', use_ce)
        self.norm_cell = nn.LayerNorm([self.hidden_size, img_size, img_size])
        # Convolutional Layers
        self.conv_i2h = Sequential(
            Conv2d(self.input_size, 4 * self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([4 * self.hidden_size, img_size, img_size], elementwise_affine=elementwise_affine)
        )
        self.conv_h2h = Sequential(
            Conv2d(self.hidden_size, 4 * self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([4 * self.hidden_size, img_size, img_size], elementwise_affine=elementwise_affine)
        )

        # CE block
        self.ceiter = ce_iterations
        self.convQ = Sequential(
            Conv2d(self.hidden_size, self.input_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([self.input_size, img_size, img_size], elementwise_affine=elementwise_affine)
        )

        self.convR = Sequential(
            Conv2d(self.input_size, self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([self.hidden_size, img_size, img_size], elementwise_affine=elementwise_affine)
        )

        # hidden states buffer, [h, c]
        # self.hiddens = None
        self.device_param = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.h_state = None
        self.c_state = None
        self.register_parameter('device_param', self.device_param)

        self.cell_attentions = None

        self.dropout = nn.Dropout(p=0.1)
        self.count = 0


    def CEBlock(self, xt, ht):
        for i in range(1, self.ceiter + 1):
            if i % 2 == 0:
                ht = (2 * torch.sigmoid(self.convR(xt))) * ht
            else:
                xt = (2 * torch.sigmoid(self.convQ(ht))) * xt
        return xt, ht

    def forward(self, x:torch.Tensor, init_hidden: bool = False):
        # initialize the hidden states, consists of hidden state: h and cell state: c
        if init_hidden:
            self.init_hiddens(x)
        cur_h, cur_c = self.h_state, self.c_state
        if self.use_ce:
            x, cur_h = self.CEBlock(x, cur_h)
        # caculate i2h, h2h
        i2h = self.conv_i2h(x)
        h2h = self.conv_h2h(cur_h)
        (i, f, g, o) = torch.split(i2h + h2h, self.hidden_size, dim=1)
        # caculate next h and c
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self._forget_bias)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        # ---------------------------------------------------
        next_c = f * cur_c + i * g
        next_c = self.norm_cell(next_c)
        next_h = o * torch.tanh(next_c)

        # next_h, next_c, self.cell_attentions = self.SEBlock(next_h, next_c)

        self.h_state = next_h
        self.c_state = next_c
        self.count += 1
        return next_h

    def init_hiddens(self, x):
        b, c, h, w = x.size()
        self.h_state = torch.zeros(b, self.hidden_size, h, w).to(self.device_param.device).to(self.device_param.dtype)
        self.c_state = torch.zeros(b, self.hidden_size, h, w).to(self.device_param.device).to(self.device_param.dtype)



class CC_ConvLSTM(Module):
    def __init__(self,
                 input_chans:int,
                 output_chans:int,
                 hidden_size:int,
                 filter_size:int,
                 num_layers:int,
                 img_size:int,
                 in_len: int,
                 out_len: int,
                 use_ce: bool = True):
        super(CC_ConvLSTM, self).__init__()
        affine = True
        self.n_layers = num_layers
        # embedding layer
        self.use_attention = True
        if self.use_attention:
            self.channel_attention = ChannelAttentionModule(input_chans)
        self.embed = Conv2d(input_chans, hidden_size, 1, 1, 0)
        self.in_len = in_len
        self.out_len = out_len
        self.dtype = torch.float32
        # lstm layers
        lstm = [LSTM_cell(hidden_size, 
                             hidden_size, 
                             filter_size, 
                             img_size, 
                             self.dtype, 
                             elementwise_affine = affine,
                             use_ce = use_ce) for l in
                range(
                    num_layers)]
        self.lstm = nn.ModuleList(lstm)
        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        

    def forward(self,
                x: torch.Tensor,
                spa_vars: torch.Tensor,
                mask: list,
                test: bool = False
                ) -> torch.Tensor:
        # use_mask = True if mask else False
        inputs = torch.cat((x[:, 0], spa_vars), 1)
        x_gen = self.forward_step(inputs, init_hidden=True)
        gn_imgs = [x_gen]
        for t in range(1, self.in_len + self.out_len - 1):
            if t < self.in_len:
                inputs = torch.cat((x[:, t], spa_vars), 1)
            else:
                if test:
                    inputs = x_gen
                else:
                    mask_t = mask[t - self.in_len]
                    inputs = mask_t * x[:, t] + (1 - mask_t) * x_gen
                inputs = torch.cat((inputs, spa_vars), 1)
            x_gen = self.forward_step(inputs, init_hidden=False)
            gn_imgs.append(x_gen)

        gn_imgs = torch.stack(gn_imgs, 1)
        return gn_imgs

    def forward_step(self, inputs, init_hidden):
        if self.use_attention:
            ct = self.channel_attention(inputs)
            inputs = ct * inputs

        h_in = self.embed(inputs)
        for l in range(self.n_layers):  # for every layer
            h_in = self.lstm[l](h_in, init_hidden)

        output = self.output(h_in)
        output = self.sigmoid(output)
        return output



