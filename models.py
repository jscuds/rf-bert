#########################################################
################## ElmoClassifier Class #################
#########################################################


class ElmoClassifier(torch.nn.Module):

    def __init__(self, options_file: str, weight_file: str, num_output_representations: int, requires_grad: bool=True, dropout: float =0):
        super().__init__()
        self.elmo = Elmo(options_file = options_file, weight_file = weight_file, 
                         num_output_representations = num_output_representations, requires_grad=requires_grad, dropout=dropout)
        self.elmo_hidden_size = self.elmo.get_output_dim()
        self.linear1 = torch.nn.Linear(in_features=self.elmo_hidden_size*3 + 1, out_features=512) #TODO change out_features to 50 for simplicity?
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=512, out_features=1)

    def forward(self, sents: torch.Tensor) -> torch.Tensor:
            # sents has shape B x 2 x self.max_length x word_length==50
            #    where dim=1 is (sent1,sent2) which is why we index as follows
        # try:
        #     assert sents.shape == (BATCH_SIZE,2,40,50)
        # except AssertionError:
        #     print(f'sents.shape: {sents.shape}\nintended shape: {(BATCH_SIZE,2,40,50)}')
        B = sents.shape[0] # batch_size because drop_last =False
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
    
    #TODO:
    # 1. add a cosine similarity comparison between two sentences


    # https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py