# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}

        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        init_packet = [self.word_embedding_size, self.hidden_state_size]
        init_kwargs = {'dropout': self.dropout,
                       'num_layers': self.num_hidden_layers,
                       'bidirectional': True}

        if self.cell_type == 'gru':
            initializer = torch.nn.GRU
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNN
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTM
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.rnn = initializer(*init_packet, **init_kwargs)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        return self.embedding(F)

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        # F_lens, perm_idx = F_lens.sort(0, descending=True)
        # _, unperm_idx = perm_idx.sort(0)
        # x = x[:, perm_idx, :]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        outputs, _ = self.rnn.forward(x)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, padding_value=h_pad)
        # outputs = outputs[unperm_idx]
        return outputs


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
        print(f"{self.target_vocab_size}")
        init_packet = [self.word_embedding_size,
                       self.hidden_state_size]
        if self.cell_type == 'gru':
            initializer = torch.nn.GRUCell
        elif self.cell_type == 'rnn':
            initializer = torch.nn.RNNCell
        elif self.cell_type == 'lstm':
            initializer = torch.nn.LSTMCell
        else:
            raise ValueError(f"cell type: '{self.cell_type}' not valid.")
        self.cell = initializer(*init_packet)
        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat

        mid = self.hidden_state_size // 2
        return torch.cat([h[-1, F_lens, :mid],
                          h[0, F_lens, mid:]], dim=1).squeeze(1)

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        # assert False, "Fill me"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mask = torch.where(E_tm1 == torch.tensor([self.pad_id]).to(device),
                           torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).to(device)
        xtilde_t = self.embedding(E_tm1) * mask.view(-1, 1)
        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        # assert False, "Fill me"
        return self.cell(xtilde_t, htilde_tm1[:, :self.hidden_state_size])

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        # assert False, "Fill me"
        logits = self.ff.forward(htilde_t)
        return logits


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']

          init_packet = [self.word_embedding_size+self.hidden_state_size,
                         self.hidden_state_size]
          if self.cell_type == 'gru':
              initializer = torch.nn.GRUCell
          elif self.cell_type == 'rnn':
              initializer = torch.nn.RNNCell
          elif self.cell_type == 'lstm':
              initializer = torch.nn.LSTMCell
          else:
              raise ValueError(f"cell type: '{self.cell_type}' not valid.")
          self.cell = initializer(*init_packet)
          self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                              self.word_embedding_size,
                                              padding_idx=self.pad_id)
          self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)
          # self.attn = torch.nn.Linear(self.hidden_size * 2, self.word_embedding_size)

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        return torch.zeros_like(h[-1])

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mask = torch.where(E_tm1 == torch.tensor([self.pad_id]).to(device),
                           torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).to(device)
        if self.cell_type == 'lstm':
            htilde_tm1 = htilde_tm1[0]  # take the hidden states
        return torch.stack([self.embedding(E_tm1) * mask.view(-1, 1), self.attend(htilde_tm1, h, F_lens)], 1)

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        alpha = self.get_attention_weights(htilde_t, h, F_lens)  # (S, N)
        alpha = alpha.tranpose(0, 1).unsqueeze(1)  # (N, 1, S)
        h.permute(0, 1)  # (N, S, 2*H)
        return torch.bmm(alpha, h).squeeze(1)  # (N, 2 * H) as desired.

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        # h = h.permute(1, 2, 0)  # (N, S, 2*H) so batches front
        # scale = torch.inverse(torch.sqrt(self.hidden_state_size * 2))
        # htilde = htilde_t.unsqueeze(1)  # (N, 1, 2*H)
        # energy = scale * torch.bmm(h, htilde)  # (N, S, 1)
        # energy.squeeze(2).transpose(0, 1)  # (S, N) as desired
        energy = torch.zeros(h.size()[:2])
        for s in range(h.size()[0]):
          energy[s] = torch.nn.functional.cosine_similarity(htilde_t,
                                                            h[s], dim=1)
        return energy

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)
        self.encoder.init_submodules()
        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type)
        self.decoder.init_submodules()

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)

        # initialize the first hidden state
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
        logits = []  # for holding logits as we do all steps in time
        for t in range(E.size()[0]-1):  # T-1
            l, h_tilde_tm1 = self.decoder.forward(E[t], htilde_tm1, h, F_lens)
            logits.append(l)
        logits = torch.stack(logits, 0)
        return logits

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]
        V = logpy_t.size()[-1]
        all_paths = logpb_tm1.unsqueeze(-1) + logpy_t  # (N, K, V), add logprobs for new extensions
        all_paths = all_paths.view((all_paths.shape[0], -1))  # (N, K*V)
        logpb_t, v = all_paths.topk(self.beam_width,
                                    -1,
                                    largest=True,
                                    sorted=True)  # take beam_width best possible extensions
        logpb_t = logpb_t  # (N, K)
        # v is (N, K)
        # v are the indices of the maximal values.
        paths = torch.div(v, V)  # paths chosen to be kept
        v = torch.remainder(v, V)  # the indices of the extended words that are kept
        # choose the paths from b_tm1_1 that were kept in our next propogation
        b_tm1_1 = b_tm1_1.gather(2, paths.unsqueeze(0).expand_as(b_tm1_1))
        # choose the htdile that coorespond to the taken baths
        if self.cell_type == 'lstm':
          b_t_0 = (htilde_t[0].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[0])),
                   htilde_t[1].gather(1, paths.unsqueeze(-1).expand_as(htilde_t[0])))
        else:
          b_t_0 = htilde_t.gather(1, paths.unsqueeze(-1).expand_as(htilde_t))
        v = v.unsqueeze(0)  # (1, N, K)
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)
        return b_t_0, b_t_1, logpb_t
