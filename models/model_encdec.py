import torch
import torch.nn as nn


class model_encdec(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a GRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    """
    def __init__(self, settings):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        channel_in = 3
        # channel_in = 6
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # # temporal encoding
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        # encoder-decoder
        self.encoder_past = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        # self.encoder_past = nn.GRU(channel_in, self.dim_embedding_key, 1, batch_first=True)
        self.decoder = nn.GRU(self.dim_embedding_key, self.dim_embedding_key, 1, batch_first=False)
        self.FC_output = torch.nn.Linear(self.dim_embedding_key, channel_in)

        # activation function
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # weight initialization: kaiming
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.FC_output.weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)

    def forward(self, past):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key)
        prediction = torch.Tensor()
        # present = past[:, -1, :3].unsqueeze(1)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()
            prediction = prediction.cuda()

        # temporal encoding for past
        past = torch.transpose(past, 1, 2)
        past_embed = self.relu(self.conv_past(past))
        past_embed = torch.transpose(past_embed, 1, 2)
        # past_embed = torch.transpose(past, 1, 2)

        # sequence encoding
        output_past, state_past = self.encoder_past(past_embed)

        # state concatenation and decoding
        state_conc = state_past
        input_fut = state_conc
        state_fut = zero_padding
        for i in range(self.past_len):
            # output_decoder, state_fut = self.decoder(input_fut, state_fut)
            # displacement_next = self.FC_output(output_decoder)
            # coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            # prediction = torch.cat((prediction, coords_next), 1)
            # present = coords_next
            # input_fut = zero_padding
            output_decoder, state_fut = self.decoder(input_fut, state_fut)
            coords_next = self.FC_output(output_decoder).squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            input_fut = zero_padding
        return prediction, state_past