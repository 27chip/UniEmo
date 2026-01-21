import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class UtteranceDataset(Dataset):
    # content is a single conversation's [utterance1, utterance2, ...]
    # utterances is a list of conversations: [conversation1, conversation2, ...]
    #def __init__(self, filename1, filename2, filename3, filename4, filename5, filename6,filenamev,filenamea,filenamed):
    def __init__(self, filename1, filename2,  filename4, filename5, filename6,filenamev_diff,filenamea_diff,filenamed_diff,filenamev,filenamea,filenamed):
        # initialize empty lists to store different parts of the data
        utterances, labels, subindex, loss_mask, speakers,diff_valence,diff_arousal,diff_dominance,valence,arousal,dominance = [], [], [], [], [], [],[],[],[],[],[]
    
        # conversations
        with open(filename1) as f:
            for line in f:
            # strip and split by tab, take elements from second onward
            content = line.strip().split('\t')[1:]
            # append processed content to utterances list
            utterances.append(content)
    
        # labels
        with open(filename2) as f:
            for line in f:
            # strip and split by tab, take elements from second onward
            content = line.strip().split('\t')[1:]
            # convert to integers and append to labels list
            labels.append([int(l) for l in content])
        '''
        # transfer labels
        with open(filename3) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append processed content to sublabels list
                sublabels.append([int(l) for l in content])
        '''

        # subindex (previous-utterance indices)
        with open(filename4) as f:
            for line in f:
            # strip and split by tab, take elements from second onward
            content = line.strip().split('\t')[1:]
            # convert to integers and append to subindex list
            subindex.append([int(l) for l in content])

        # loss_mask
        with open(filename5) as f:
            for line in f:
            # strip and split by tab, take elements from second onward
            content = line.strip().split('\t')[1:]
            # convert to integers and append to loss_mask list
            loss_mask.append([int(l) for l in content])
        
        # read the sixth file (speakers)
        with open(filename6) as f:
            for line in f:
            # strip and split by tab, take elements from second onward
            content = line.strip().split('\t')[1:]
            # append to speakers list
            speakers.append(content)

        with open(filenamev_diff) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append float values to diff_valence list
                diff_valence.append([float(v) for v in content])
        
        with open(filenamea_diff) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append float values to diff_arousal list
                diff_arousal.append([float(a) for a in content]) 
        
        with open(filenamed_diff) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append float values to diff_dominance list
                diff_dominance.append([float(d) for d in content])

        with open(filenamev) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append float values to valence list
                valence.append([float(v) for v in content])
        
        with open(filenamea) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append float values to arousal list
                arousal.append([float(a) for a in content]) 
        
        with open(filenamed) as f:
            for line in f:
                # strip and split by tab, take elements from second onward
                content = line.strip().split('\t')[1:]
                # append float values to dominance list
                dominance.append([float(d) for d in content])


        # assign processed data to class attributes
        self.utterances = utterances
        self.labels = labels
        #self.sublabels = sublabels
        self.subindex = subindex
        self.loss_mask = loss_mask
        self.speakers = speakers

        self.diff_valence = diff_valence
        self.diff_arousal = diff_arousal
        self.diff_dominance = diff_dominance
        
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index): 
        # get the index-th utterance list
        s = self.utterances[index]
        # get corresponding labels
        l = self.labels[index]
        # get sublabels for this conversation
        #sb = self.sublabels[index]
        # get subindex for this conversation
        i = self.subindex[index]
        # get loss_mask for this conversation
        m = self.loss_mask[index]
        # get speakers for this conversation
        sp = self.speakers[index]

        diff_v = self.diff_valence[index]
        diff_a = self.diff_arousal[index]
        diff_d = self.diff_dominance[index]  

        v = self.valence[index]
        a = self.arousal[index]
        d = self.dominance[index]   
        return s, l,i, m, sp,diff_v,diff_a,diff_d,v,a,d
        #return s, l, sb, i, m, sp,v,a,d
        
    
    def collate_fn(self, data):
        # convert data to a DataFrame and return each column as a list
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
#def DialogLoader(filename1, filename2, filename3, filename4, filename5, filename6, filenamev,filenamea,filenamed,batch_size, shuffle):
def DialogLoader(filename1, filename2, filename4, filename5, filename6, filenamev_diff,filenamea_diff,filenamed_diff,filenamev,filenamea,filenamed,batch_size, shuffle):
    
    # create an UtteranceDataset object
    # arguments are the file paths
    dataset = UtteranceDataset(filename1, filename2, filename4, filename5, filename6,filenamev_diff,filenamea_diff,filenamed_diff,filenamev,filenamea,filenamed)

    # create a DataLoader
    # arguments: dataset, shuffle flag, batch size, and collate function
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)

    return loader

