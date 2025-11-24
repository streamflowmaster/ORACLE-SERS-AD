from dataclasses import dataclass
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

mse_loss = nn.MSELoss()
@dataclass

class GPTConfig:
    if_cls_tokens: bool = False
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 16
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = True
    cls_num: int = 2
    apply_ehr: bool = False
    apply_gene: bool = False
    apply_mri: bool = False
    apply_blood_test: bool = False
    apply_neurological_test: bool = False
    apply_spec: bool = False
    apply_protein: bool = False
    apply_spec_meta: bool = False
    spec_norm: bool = False
    if_data_balance: bool = True
    signal_length: int = 512
    device: str = 'cuda:1'
    random_mask_rate: float = 0.5
    spectra_encoder: str = 'transformer'
    add_noise: bool = False

class transformer_cls(nn.Module):

    def __init__(self, config: GPTConfig):
        super(transformer_cls, self).__init__()
        print(config.if_cls_tokens)
        self.mri_embedding = nn.Embedding(20, config.n_embd)
        self.gender_embedding = nn.Embedding(2, config.n_embd)
        self.education_embedding = nn.Embedding(6, config.n_embd)
        self.geneC130_embedding = nn.Embedding(4, config.n_embd)
        self.geneR176_embedding = nn.Embedding(4, config.n_embd)
        self.genetic_relation_embedding = nn.Embedding(3, config.n_embd)
        encoder_layers = nn.TransformerEncoderLayer(d_model=config.n_embd,
                                                    nhead=config.n_head,
                                                    dim_feedforward=config.n_embd,
                                                    dropout=config.dropout,
                                                    activation='gelu',)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=config.n_layer,
                                                          norm=nn.LayerNorm(config.n_embd))
        self.vaccum_embedding = nn.Parameter(torch.zeros(1, config.n_embd))
        self.fc = nn.Linear(config.n_embd, config.cls_num)
        self.config = config
        self.device =  config.device
        self.spec_num = 5
        self.exception_value = 999
        self.age_embedding = nn.Linear(1, config.n_embd)
        self.education_duration_embedding = nn.Linear(1, config.n_embd)
        self.MMSE_embedding = nn.Linear(1, config.n_embd)
        self.moca_embedding = nn.Linear(1, config.n_embd)
        self.neurological_test_embedding = nn.Linear(1, config.n_embd)
        self.family_history_embedding = nn.Embedding(2, config.n_embd)
        self.blood_test_embedding = nn.Linear(1, config.n_embd)
        self.protein_embedding = nn.Linear(1, config.n_embd)
        self.spec_meta_embedding = nn.Linear(201*2, config.n_embd)
        self.continous_embedders = nn.ModuleDict({
            'age': self.age_embedding,
            'education_duration': self.education_duration_embedding,
            'MMSE': self.MMSE_embedding,
            'moca': self.moca_embedding,
            'neurological_test': self.neurological_test_embedding,
            'blood_test': self.blood_test_embedding,
            'protein': self.protein_embedding,
            'spec_meta': self.spec_meta_embedding
        })
        self.continous_modalities = []
        self.discrete_modalities = []
        if self.config.apply_ehr:
            self.continous_modalities.extend(['age','education_duration'])
            self.discrete_modalities.extend(['family_history','gender','education'])

        if self.config.apply_gene:
            self.discrete_modalities.extend(['gene_c130','gene_r176','genetic_relation'])

        if self.config.apply_blood_test:
            self.continous_modalities.append('blood_test')

        if self.config.apply_neurological_test:
            self.continous_modalities.extend(['MMSE','moca','neurological_test'])

        if self.config.apply_protein:
            self.continous_modalities.append('protein')

        if self.config.apply_spec_meta:
            self.continous_modalities.extend(['spec_meta'])

        # self.continous_modalities = ['age', 'education_duration', 'MMSE', 'moca', 'neurological_test', 'blood_test']
        self.positional_embedding = nn.Parameter(torch.zeros(1, 200, config.n_embd))
        self.discrete_embedders = nn.ModuleDict({
            'gender': self.gender_embedding,
            'education': self.education_embedding,
            'gene_c130': self.geneC130_embedding,
            'gene_r176': self.geneR176_embedding,
            'genetic_relation': self.genetic_relation_embedding,
            'family_history': self.family_history_embedding,
        })

        # self.discrete_modalities = ['gender', 'education', 'gene_c130', 'gene_r176', 'genetic_relation']
        self.spec_encoder = spectra_encoder(config)
        self.to(self.device)
        if config.if_cls_tokens:
            self.cls_tokens = nn.Parameter(torch.zeros(1, 5, config.n_embd))

        self.modality_apply_positions = {
            'MRI':[0,1,2,3,4,5],
            'age': [6],
            'education_duration': [7],
            'blood_test': [8+i for i in range(26)],
            'MMSE': [34],
            'moca': [35],
            'neurological_test': [36+i for i in range(25)],
            'protein': [61],
            'spec_meta': [62+i for i in range(27)],
            'family_history': [89],
            'gender': [90],
            'education': [91],
            'gene_c130':[92],
            'gene_r176':[93],
            'genetic_relation':[94],
            'spec':{95,96,97,98,99}
        }
        self.modality_pos = None

    def embedding_continous(self, embedder, to_embed):
        if type(to_embed) == torch.Tensor:
            # print(to_embed.shape)
            to_embed = to_embed.float()
            if self.training and self.config.add_noise:
                to_embed = to_embed + torch.randn_like(to_embed) * 0.1

            if len(to_embed.shape) == 1:
                embed = torch.zeros(to_embed.shape[0],
                                    self.config.n_embd, device=self.device)
                embed[torch.argwhere(to_embed != self.exception_value)[:, 0]] = embedder(
                    to_embed[torch.argwhere(to_embed != self.exception_value)])
                embed[torch.argwhere(to_embed == self.exception_value)[:, 0]] = self.vaccum_embedding
                embed = embed.unsqueeze(1)
            else:
                to_embed_shape = to_embed.shape
                embeded_shape = list(to_embed_shape)
                embeded_shape[-1] = self.config.n_embd
                # last dim is the feature dim, the first dim is the batch size,other dims remain the same
                # make the code more general to anylength of the to_embed
                batch_size = to_embed.shape[0]
                feature_size = to_embed.shape[-1]
                to_embed = to_embed.reshape(batch_size,-1, feature_size)
                embed = embedder(to_embed).reshape(embeded_shape)

        else:
            # print('to_embed',to_embed[7])
            to_embed = torch.stack(to_embed).float()
            embed = torch.zeros(to_embed.shape[0],to_embed.shape[1],self.config.n_embd).to(self.device)
            embed[torch.argwhere(to_embed != self.exception_value)[:, 0],
                  torch.argwhere(to_embed != self.exception_value)[:, 1]] = embedder(
                to_embed[torch.argwhere(to_embed != self.exception_value)[:, 0],
                         torch.argwhere(to_embed != self.exception_value)[:, 1]].unsqueeze(1))

            embed = embed.permute(1,0,2)
        return embed

    def embedding_discrete(self, embedder, to_embed):
        embed = torch.zeros(to_embed.shape[0],1, self.config.n_embd, device=self.device)
        # print(embedder(to_embed[torch.argwhere(to_embed != self.exception_value)]).shape)

        embed[torch.argwhere(to_embed != self.exception_value)[:, 0]] = embedder(to_embed[torch.argwhere(to_embed != self.exception_value)])
        embed[torch.argwhere(to_embed == self.exception_value)[:, 0]] = self.vaccum_embedding
        return embed

    def count_seq_len(self,info):
        length = 0
        for key in info:
            # length+=info[key].shape[1]
            # print(key,length)
            if type(info[key]) == torch.Tensor:
                if len(info[key].shape) >= 2 and key in self.continous_modalities:
                    length += info[key].shape[1]

                elif key == 'spec':
                    length += self.spec_num

                elif key == 'MRI':
                    length += info[key].shape[1]

                else:
                    length += 1

            else:
                length += len(info[key])
            # print(key,length)
        return length

    # def count_seq_key(self,info):
    #     keys = []
    #     for key in info:
    #         if type(info[key]) == torch.Tensor:
    #             if len(info[key].shape) >= 2 and key in self.continous_modalities:
    #                 keys.extend([key + str(i) for i in range(info[key].shape[1])])
    #             else:
    #                 keys.append(key)
    #         else:
    #             keys.extend([key + str(i) for i in range(len(info[key]))])
    #     return keys

    def count_seq_key(self,info):
        keys = ['MRI']*info['MRI'].shape[1]
        # print( "During Counting")
        for key in self.continous_modalities:
            # print(key,len(keys))
            if type(info[key]) == torch.Tensor:
                if len(info[key].shape) >= 2 and key in self.continous_modalities:
                    keys.extend([key + str(i) for i in range(info[key].shape[1])])
                else:
                    keys.append(key)
            else:
                keys.extend([key + str(i) for i in range(len(info[key]))])
            # print(len(keys))

        for key in self.discrete_modalities:
            # print(key,len(keys))
            if type(info[key]) == torch.Tensor:
                if len(info[key].shape) == 2:
                    keys.append(key)
                else:
                    keys.append(key)
            else:
                keys.extend([key + str(i) for i in range(len(info[key]))])
            # print(len(keys))

        if self.config.apply_spec:

            keys.extend(['spec']*self.spec_num)
        # print(keys)
        return keys

    def count_seq_pos(self,info):
        modality_pos = []

        if 'MRI' in info:
            modality_pos.extend(self.modality_apply_positions['MRI'])

        # print( "During Counting")
        for key in self.continous_modalities:
            modality_pos.extend(self.modality_apply_positions[key])

        for key in self.discrete_modalities:
            modality_pos.extend(self.modality_apply_positions[key])

        if self.config.apply_spec:
            modality_pos.extend(self.modality_apply_positions['spec'])

        self.modality_pos = modality_pos
        return modality_pos

    def embedding_all(self,info):
        '''

        Args:
            info:
            info_dict: a dictionary containing the information of the patient
                label: the label of the patient
                MRI: the MRI data of the patient
                age: the age of the patient
                gender: the gender of the patient
                education: the education of the patient
                education_duration: the education duration of the patient
                MMSE: the MMSE score of the patient
                moca: the MoCA score of the patient
                neurological_test: the neurological test of the patient
                blood_test: the blood test of the patient
        Returns:

        '''

        batch_size = info['MRI'].shape[0]
        seq_len = self.count_seq_len(info)
        # print('seq_len',seq_len)
        embeddings = torch.zeros(batch_size, seq_len, self.config.n_embd, device=self.device)
        # embed the MRI data
        mri_embed= self.mri_embedding(info['MRI'])
        current_len = mri_embed.shape[1]
        embeddings[:, :info['MRI'].shape[1]] = mri_embed
        # print('During Embedding')
        for key in self.continous_modalities:
            embed = self.embedding_continous(self.continous_embedders[key], info[key])
            # print(key,current_len,current_len + embed.shape[1],seq_len)
            embeddings[:, current_len:current_len + embed.shape[1]] = embed
            current_len += embed.shape[1]
        for key in self.discrete_modalities:
            embed = self.embedding_discrete(self.discrete_embedders[key], info[key])
            embeddings[:, current_len:current_len + embed.shape[1]] = embed
            # print(key, current_len, current_len + embed.shape[1], seq_len)
            current_len += embed.shape[1]
        # print('all embedding', embeddings.shape)
        if self.config.apply_spec:
            if self.training and self.config.add_noise:
                info['spec'] = info['spec'] + torch.randn_like(info['spec']) * 0.1
            spec_embed = self.spec_encoder(info['spec'])
            # print('spec',info['spec'].shape,spec_embed.shape)
            embeddings[:, current_len:current_len + self.spec_num] = spec_embed
        # print('all embedding',embeddings.shape)
        return embeddings

    def forward(self, info):
        if self.modality_pos is None:
            self.modality_pos = self.count_seq_pos(info)
        # print(self.modality_pos)

        seq = self.embedding_all(info)
        # print("SEQ",seq.shape)
        # print(seq.shape)
        # random mask some tokens during training
        # if self.training:
        # random select some tokens to mask, random select the token idx on the batch dim and seq dim
        if self.training:
            mask = torch.rand(seq.shape[:2]) < self.config.random_mask_rate
            seq[mask] = self.vaccum_embedding
        if self.config.if_cls_tokens:
            seq = torch.cat([self.cls_tokens.repeat(seq.shape[0], 1, 1), seq], dim=1)
        seq = seq + self.positional_embedding[:, self.modality_pos]
        output = self.transformer_encoder(seq)
        if self.config.if_cls_tokens:
            # print(output[:,:5].shape)
            # print(output[:,:5].mean(dim=1).shape)
            # print(self.fc(output[:,:5].mean(dim=1)).shape)
            output = self.fc(output[:,:5].mean(dim=1))
        else:
            output = torch.mean(output, dim=1)
            output = self.fc(output)
        output = torch.softmax(output, dim=1)
        return output

    def forward_to_embeddings(self, info):
        if self.modality_pos is None:
            self.modality_pos = self.count_seq_pos(info)
        # print(self.modality_pos)

        seq = self.embedding_all(info)
        # print("SEQ",seq.shape)
        # print(seq.shape)
        # random mask some tokens during training
        # if self.training:
        # random select some tokens to mask, random select the token idx on the batch dim and seq dim
        if self.training:
            mask = torch.rand(seq.shape[:2]) < self.config.random_mask_rate
            seq[mask] = self.vaccum_embedding
        if self.config.if_cls_tokens:
            seq = torch.cat([self.cls_tokens.repeat(seq.shape[0], 1, 1), seq], dim=1)
        seq = seq + self.positional_embedding[:, self.modality_pos]
        output = self.transformer_encoder(seq)
        if self.config.if_cls_tokens:
            # print(output[:,:5].shape)
            # print(output[:,:5].mean(dim=1).shape)
            # print(self.fc(output[:,:5].mean(dim=1)).shape)
            output = self.fc(output[:,:5].mean(dim=1))
        else:
            output = torch.mean(output, dim=1)
            output = self.fc(output)
        return output


    def forward_pretrain(self,info):
        # bert like pretrain
        seq = self.embedding_all(info)
        # random mask some tokens during training
        # if self.training:
        # random select some tokens to mask, random select the token idx on the batch dim and seq dim

        mask = torch.rand(seq.shape[:2]) < self.config.random_mask_rate
        seq[mask] = self.vaccum_embedding
        if self.config.if_cls_tokens:
            seq = torch.cat([self.cls_tokens.repeat(seq.shape[0], 1, 1), seq], dim=1)
        seq_ps = seq + self.positional_embedding[:, :seq.shape[1]]
        output = self.transformer_encoder(seq_ps)
        # loss to recover the original tokens
        loss = mse_loss(output[mask], seq[mask])
        return loss

    def load_checkpoint(self, path,device):
        self.load_state_dict(torch.load(path,map_location=device))
        self.eval()

    def assert_none(self,data):
        for key in data:
            if data[key] is None:
                data[key] = torch.tensor([0])
        return data

def spectra_encoder(config: GPTConfig):
    if config.spectra_encoder == 'transformer':
        return SpectraTransformerEncoder(config)
    elif config.spectra_encoder == 'gcn':
        return SpectraGCNEncoder(config)
    else:
        raise ValueError('spectra_encoder must be transformer or gcn')

class SpectraTransformerEncoder(nn.Module):
    def __init__(self, config: GPTConfig):
        super(SpectraTransformerEncoder, self).__init__()
        self.spec_embedding = nn.Linear(config.signal_length, config.n_embd)
        self.config = config
        self.cls_tokens = nn.Parameter(torch.zeros(1, 15, config.n_embd))
        if config.spec_norm:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=config.n_embd,
                       nhead=config.n_head, dim_feedforward=config.n_embd,
                       dropout=config.dropout), num_layers=config.n_layer,
                norm=nn.LayerNorm(config.n_embd))
        else:
            self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.n_embd,
            nhead=config.n_head, dim_feedforward=config.n_embd,
            dropout=config.dropout), num_layers=config.n_layer)


    def forward(self,spectra):
        spectra = torch.nn.functional.interpolate(spectra, size=(self.config.signal_length,))
        # apply cls tokens
        spectra = self.spec_embedding(spectra)
        spectra = torch.cat([self.cls_tokens.repeat(spectra.shape[0], 1, 1), spectra], dim=1)
        output = self.transformer(spectra)[:,:5]
        return output

class gcn(nn.Module):
    def __init__(self,config: GPTConfig):
        super(gcn, self).__init__()
        torch.manual_seed(12345)
        device = config.device
        self.config = config
        node_features = config.signal_length
        graph_embed_dim = config.n_embd
        hidden_channels = torch.sqrt(torch.tensor(node_features * graph_embed_dim)).int().item()

        self.conv1 = GCNConv(node_features, hidden_channels).to(device)
        self.conv2 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv3 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv4 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv5 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.lin = nn.Linear(hidden_channels,graph_embed_dim).to(device)
        self.device = device

    def forward(self, x, edge_index):
        edge_index = edge_index.int().to(self.device)
        # print(x.shape)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # print(x.shape)
        x = x.max(0)[0]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class SpectraGCNEncoder(torch.nn.Module):
    def __init__(self,config: GPTConfig):
        super(SpectraGCNEncoder, self).__init__()
        self.config = config
        torch.manual_seed(12345)
        self.gcn1 = gcn(config)
        self.gcn2 = gcn(config)
        self.gcn3 = gcn(config)
        self.gcn4 = gcn(config)
        self.gcn5 = gcn(config)
        self.device = config.device
        self.gcns = [self.gcn1,self.gcn2,self.gcn3,self.gcn4,self.gcn5]

    def forward_one_graph(self,spectra,edge):
        embedding = torch.zeros(len(self.gcns), self.gcn1.lin.out_features).to(self.device)
        for idx,gcn in enumerate(self.gcns):
            embedding[idx] = gcn(spectra,edge)
        return embedding

    def build_spectra_graph(self,patient,edge_num):
        # patient: N*L
        # N: number of nodes
        # L: number of features
        N = patient.shape[0]
        index = torch.zeros((2,edge_num))
        affine_value = torch.mm(patient, patient.t())/(torch.norm(patient,dim=1).unsqueeze(1)*torch.norm(patient,dim=1).unsqueeze(0))
        # affine_value = torch.fill_diagonal_(affine_value, 0)
        topk,idx = torch.topk(affine_value.flatten(), k=edge_num)
        index[0] = idx//N
        index[1] = idx%N
        # print(index.shape,topk.shape)
        return index,topk


    def build_spectra_graphs(self,spectra,edge_num):
        # patients: B*N*L
        indices = []
        values = []
        for i in range(spectra.shape[0]):
            index,value = self.build_spectra_graph(spectra[i],edge_num)
            indices.append(index)
            values.append(value)
        indices = torch.stack(indices,dim=0)
        values = torch.stack(values,dim=0)
        # print(indices.shape,values.shape)
        return indices,values

    def forward(self,spectra,edge_num=500):
        B,N,L = spectra.shape
        indices,values = self.build_spectra_graphs(spectra,edge_num)
        embeddings = torch.zeros(B,len(self.gcns), self.config.n_embd).to(self.device)
        for i in range(B):
            embeddings[i] = self.forward_one_graph(spectra[i],indices[i])
        return embeddings



class MLP(nn.Module):
    def __init__(self, embed,out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed, embed*3)
        self.fc2 = nn.Linear(embed*3, embed*9)
        self.fc3 = nn.Linear(embed*9, out)

    def forward(self,input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


