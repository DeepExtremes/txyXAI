import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, convs_per_layer=1, 
                 dilation=1, residual=False, no_memory=False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.residual = residual
        self.no_memory= no_memory

        #If convs_per_layer > 1, the first layer is always a 1x1 convolution
        seq_list=[]
        for i in range(convs_per_layer):
            seq_list.append( nn.Conv2d(
                  in_channels=(self.input_dim + self.hidden_dim) if i==0 else (4 * self.hidden_dim),
                  out_channels=4 * self.hidden_dim, bias=self.bias,
                  kernel_size=self.kernel_size if i>0 or convs_per_layer==1 else (1,1),
                  padding=(self.padding if i>0 or convs_per_layer==1 else (0,0)) \
                        if dilation==1 or self.kernel_size[0]==1 else dilation,
                  dilation=dilation) )
            if i < convs_per_layer -1 : seq_list.append(nn.SiLU())
        if self.residual:
            self.conv = nn.ModuleList(seq_list)
        else:
            self.conv = nn.Sequential(*seq_list)
        self.device= seq_list[0].weight.device
            
        self.hidden_c = None
        self.hidden_h = None
        self.init_hidden()

    def forward(self, input_tensor, cur_state):
        #Extract input components and combine them
        if len(input_tensor) == 3:
            txyp, xyp, tp= input_tensor #(b c h w), (b c h w), (b c)
            input_tensor= torch.concatenate(
                [txyp, xyp, tp[...,None,None].repeat(1,1,xyp.shape[-2],xyp.shape[-1])], axis=1)
        elif len(input_tensor) == 1:
            input_tensor= input_tensor[0]
        else:
            raise RuntimeError(f'{len(input_tensor)=} should be 1 or 3')
        
        h_cur, c_cur = cur_state
        #return input_tensor[:,:60], input_tensor[:,:60] #For XAI debugging

        #Broadcast [1, с, 1, 1] -> [b, c, h, w]
        b, _, h, w = input_tensor.shape
        h_cur = h_cur.expand([b, -1, h, w])
        c_cur = c_cur.expand([b, -1, h, w])
        
        if self.no_memory:
            #Just for ablation tests
            h_cur= h_cur*0.
            c_cur= c_cur*0.
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        if self.residual:
            out= self.conv[0](combined)
            residual= out.clone()
            for i, conv_or_act in enumerate(self.conv):
                if i==0: continue #skip 0, since we already did it
                out= conv_or_act(out)                
                if ( (i+1) % (int(self.residual)*2) ) == 0:
                    residual, out= out.clone(), out + residual                    
        else:
            combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self):
        if self.hidden_c is None or self.hidden_h is None:
            # [bs, c, h, w]
            hidden_state = (torch.zeros(1, self.hidden_dim, 1, 1, device=self.device),
                torch.zeros(1, self.hidden_dim, 1, 1, device=self.device))
            hidden_state = list(map(lambda x: nn.Parameter(x), hidden_state))
            self.hidden_c = hidden_state[0]
            self.hidden_h = hidden_state[1]
        return self.hidden_h, self.hidden_c


class ConvLSTMCustom(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 convs_per_layer=1, batch_first=True, bias=True, return_all_layers=False, 
                 dilation=1, residual=False, no_memory=False):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          convs_per_layer=convs_per_layer,
                                          dilation=dilation, residual=residual,
                                          no_memory=no_memory))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, chunk_size=None):
        """

        Parameters
        ----------
        input_tensor: todo
            tuple of tensors (tp, xyp, txyp)) with shapes ( (b t c), (b c h w), (b t c h w) )
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list: list
            [[h, c] * n_layers], where h and c are the last step tensors ([bs, c, h, w]).
            If return_all_layers=True returns values ([[h, c]]) for last layer only.
        layer_output:
            [h * n_layers], where h is stacked along time dimension ([t, bs, c, h, w])
            If return_all_layers=True returns values ([h]) for last layer only.
        """
        
        assert self.batch_first, 'This is the only configuration tested so far'
        txyp, xyp, tp= input_tensor #(b t c h w), (b c h w), (b t c)
        if self.batch_first:
            # Prioritizing time dimension gives a slight boost in performance
            txyp = txyp.permute(1, 0, 2, 3, 4) # (b, t, c, h, w) -> (t, b, c, h, w)
            tp = tp.permute(1, 0, 2) # (b, t, c) -> (t, b, c)

        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self._init_hidden()
            
        layer_output_list = []
        last_state_list = []
            
        seq_len = txyp.size(0)
        cur_layer_input = txyp

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                        input_tensor=(cur_layer_input[t], xyp, tp[t]) if layer_idx==0 else (cur_layer_input[t],), 
                                      cur_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list
        
#         #Set up loop control
#         start_idx= 0
#         chunk_size= txyp.shape[0] if chunk_size is None else chunk_size
#         end_idx= start_idx + chunk_size
        
#         #Create list of lists for outputs and states
#         layer_output_list_all= []
#         #layer_state_list_all= []
        
#         #Iterate over chunks
#         while True:
#             #Get chunks
#             tp_chunk, xyp_chunk, txyp_chunk= tp[start_idx:end_idx], xyp, txyp[start_idx:end_idx] #xyp has no time dim

#             seq_len = txyp_chunk.size(0)
#             cur_layer_input = txyp_chunk
                         
#             layer_output_list = []
#             last_state_list = []

#             for layer_idx in range(self.num_layers):

#                 h, c = hidden_state[layer_idx]
#                 output_inner = []
#                 for t in range(seq_len):
#                     h, c = self.cell_list[layer_idx](
#                         input_tensor=(cur_layer_input[t], xyp_chunk, tp_chunk[t]) if layer_idx==0 else (cur_layer_input[t],), 
#                                       cur_state=[h, c])
#                     output_inner.append(h)

#                 layer_output = torch.stack(output_inner, dim=0)
#                 cur_layer_input = layer_output

#                 layer_output_list.append(layer_output)
#                 last_state_list.append([h, c])
            
#             #Save in meta-list
#             layer_output_list_all.append(layer_output_list) 
#             #layer_state_list_all.append(last_state_list) 
               
#             #Loop control
#             if end_idx == txyp.shape[0]: break
#             end_idx+= chunk_size
#             start_idx+= chunk_size
#             if end_idx > txyp.shape[0]: end_idx= txyp.shape[0]
            
#         #Concatenate all outputs and states
#         layer_output_list= [torch.concatenate(items, axis=0) for items in zip(*layer_output_list_all)]
#         #layer_state_list= [torch.concatenate(items, axis=0) for items in zip(*layer_state_list_all)]

#         if not self.return_all_layers:
#             layer_output_list = layer_output_list[-1:]
#             last_state_list = last_state_list[-1:]
#         else:
#             raise NotImplementedError('Idk how to deal with last_state_list when chunking')

        # return layer_output_list, last_state_list

    def _init_hidden(self):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden())
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
