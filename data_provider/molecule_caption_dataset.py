import torch
from torch_geometric.data import Dataset
import os

class MoleculeCaption(Dataset):
    def __init__(self, root, text_max_len, prompt=None, filtered_cid_path=None):
        super(MoleculeCaption, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

        if filtered_cid_path is not None:
            print('before filtering', len(self.graph_name_list), len(self.text_name_list), len(self.smiles_name_list))
            with open(filtered_cid_path, 'r') as f:
                self.filtered_cid_set = [line.strip() for line in f.readlines()]
                self.filtered_cid_set = set(self.filtered_cid_set)
            filtered_graph_name_list = []
            for g in self.graph_name_list:
                cid = g.split('_')[1][:-3]
                if cid in self.filtered_cid_set:
                    filtered_graph_name_list.append(g)
            self.graph_name_list = filtered_graph_name_list
            filtered_text_name_list = []
            for t in self.text_name_list:
                cid = t.split('_')[1][:-4]
                if cid in self.filtered_cid_set:
                    filtered_text_name_list.append(t)
            self.text_name_list = filtered_text_name_list
            filtered_smiles_name_list = []
            for s in self.smiles_name_list:
                cid = s.split('_')[1][:-4]
                if cid in self.filtered_cid_set:
                    filtered_smiles_name_list.append(s)
            self.smiles_name_list = filtered_smiles_name_list
            print('after filtering', len(self.graph_name_list), len(self.text_name_list), len(self.smiles_name_list))

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        # load and process text
        text_path = os.path.join(self.root, 'text', text_name)
        
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            text_list.append(line.strip('\n'))
            if count > 100:
                break
        text = ' '.join(text_list) + '\n'

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return data_graph, text, smiles_prompt
    
    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        return sentence_token

if __name__ == '__main__':
    import numpy as np
    pretrain = MoleculeCaption('../data/PubChem324k/pretrain/', 1000, '', filtered_cid_path='../data/PubChem324k/filtered_pretrain_cids.txt')
    train = MoleculeCaption('../data/PubChem324k/train/', 1000, '')
    valid = MoleculeCaption('../data/PubChem324k/valid/', 1000, '')
    test = MoleculeCaption('../data/PubChem324k/test/', 1000, '')

    for subset in [pretrain, train, valid, test]:
        g_lens = []
        t_lens = []
        for i in range(len(subset)):  
            data_graph, text, _ = subset[i]
            g_lens.append(len(data_graph.x))
            t_lens.append(len(text.split()))
            # print(len(data_graph.x))
        g_lens = np.asarray(g_lens)
        t_lens = np.asarray(t_lens)
        print('------------------------')
        print(g_lens.mean())
        print(g_lens.min())
        print(g_lens.max())
        print(t_lens.mean())
        print(t_lens.min())
        print(t_lens.max())