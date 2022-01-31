import torch
from allennlp.modules.elmo import ElmoLstm  # , _ElmoBiLm
from allennlp.modules.elmo import Elmo

# from transformers.utils.dummy_pt_objects import ElectraModel


#########################################################
################## ElmoClassifier Class #################
#########################################################


OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class ElmoLstmWithTransformation(torch.nn.Module):
    """Appends a linear transformation (via inner `M` matrix) to ElmoLstm."""
    def __init__(self, lstm: ElmoLstm, embedding_dim=512): # embedding_dim matches ElmoLstm default
        super().__init__()
        self.lstm = lstm
        self.M = torch.nn.Parameter(
            torch.eye(embedding_dim, dtype=torch.float32), requires_grad=True
        )
        self.register_parameter('M', self.M) # TODO does this do anything?

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        # Parameters
        inputs : `torch.Tensor`, required.
            A Tensor of shape `(batch_size, sequence_length, hidden_size)`.
        mask : `torch.BoolTensor`, required.
            A binary mask of shape `(batch_size, sequence_length)` representing the
            non-padded elements in each sequence in the batch.
        # Returns
        `torch.Tensor`
            A `torch.Tensor` of shape (num_layers, batch_size, sequence_length, hidden_size),
            where the num_layers dimension represents the LSTM output from that layer.
        """
        inputs = torch.matmul(inputs, self.M)
        return self.lstm(inputs, mask)

class ElmoClassifier(torch.nn.Module):
    """
    Uses ELMo embeddings to perform QQP task; allows for modification
    w/ orthogonal `M` matrix via ElmoLstmWithTransformation class.

    Args:
        options_file (str): ELMo options file
        weight_file (str): ELMo weight file
        num_output_representations (int): creates a list of len(num_output_representations)
            of the same embeddings to be used with .pop(); see allennlp repo for details
        requires_grad (bool): allow ELMo embeddings to learn (True) 
            or freeze embeddings (False)
        dropout (float): dropout percentage
        embedding_dim (int): size of ElmoLstm dim to create size of `M`
        linear_hidden_dim (int): size of the linear classifier hidden dimensions
        m_transform (bool): use `M` from ElmoLstmWithTransformation 

    https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py
    """
    #js added requires_grad argument to switch for freezing/un-freezing weights
    #js added m_transform argument
    def __init__(self, options_file: str = OPTIONS_FILE, weight_file: str = WEIGHT_FILE, 
                 num_output_representations: int=1, requires_grad: bool=False, 
                 dropout: float=0, embedding_dim: int=512, # embedding_dim matches ElmoLstm default
                 linear_hidden_dim: int=512, m_transform: bool=True):
        super().__init__()
        self.elmo = Elmo(options_file=options_file, weight_file=weight_file,
                         num_output_representations = num_output_representations,
                         requires_grad=requires_grad, dropout=dropout)
        
        # Wrap the inner LSTM in an nn.Module that applies a matrix transformation
        # to the embeddings before passing them to the LSTM.
        if m_transform:
            lstm_with_transformation = ElmoLstmWithTransformation(
                self.elmo._elmo_lstm._elmo_lstm, embedding_dim=embedding_dim
            )
            self.elmo._elmo_lstm._elmo_lstm = lstm_with_transformation
        self.elmo_hidden_size = self.elmo.get_output_dim() 
        
        #TODO change out_features to 50 for simplicity?
        #TODO change to nn.Sequential for readability?
        self.linear1 = torch.nn.Linear(in_features=self.elmo_hidden_size*3 + 1, out_features=linear_hidden_dim) # 1024*3+1 = 3073
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=linear_hidden_dim, out_features=1)


    def forward(self, sents: torch.Tensor) -> torch.Tensor:
        """
        Inputs: sents of shape [batch_size, 2, max_seq_length, max_word_length]
            where dim=1 is is (sent1,sent2)

        Outputs: logits for classification of 0 (non-paraphrase) or 1 (paraphrase)
        """
            # sents has shape B x 2 x self.max_length x word_length==50
            #    where dim=1 is (sent1,sent2) which is why we index as follows
        B = sents.shape[0] # batch_size because drop_last = False
        # try:
        #     assert sents.shape == (B,2,40,50)
        # except AssertionError:
        #     print(f'sents.shape: {sents.shape}\nintended shape: {(B,2,40,50)}')
        
        assert sents.shape == (B,2,40,50)
        sent_1 = sents[:,0,:,:] # shape: B x 1 x MAX_LENGTH==40 x 50
        sent_2 = sents[:,1,:,:] # shape: B x 1 x MAX_LENGTH==40 x 50
        
        elmo_sent_1 = self.elmo(sent_1)  # shape: B x MAX_LENGTH==40 x 1024
        elmo_sent_2 = self.elmo(sent_2)
        average_sent_1 = elmo_sent_1['elmo_representations'][0].mean(dim=1) #B x 1024
        average_sent_2 = elmo_sent_2['elmo_representations'][0].mean(dim=1) #B x 1024

        u,v = average_sent_1, average_sent_2  
        assert u.shape == (B,1024)
        assert v.shape == (B,1024)

        # FROM GLUE baseline: https://github.com/nyu-mll/GLUE-baselines/blob/b1c82396d960fd9725517089822d15e31b9882f5/src/models.py#L330
        # return torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)
        cat_vect = torch.cat((u, v, torch.norm(u-v, p=1, dim=1, keepdim=True), u*v), dim=1)  #NOTE: 2022-01-26; changed to p=1 after looking at GLUE baseline code
        assert cat_vect.shape == (u.shape[0],u.shape[1]*3+1) # shape [batch, 1024*3+1]
        x = self.linear1(cat_vect)
        x = self.relu(x)
        logits = self.linear2(x)
        assert logits.shape == (B,1)
        return logits
    
    # https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py




 #TODO:
    # 1. based on num_output_representations...should we average between those?  Use just the "top" layer?  Is that index 0 or index -1? Right now I'm grabbing elmo_out['elmo_representations'][0]
    # https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py
