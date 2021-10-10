import s3prl
import torch
import torchaudio
from transformers import BertTokenizer, BertModel

SAMPLE_RATE=16000

class BertFeatureExtractor:

    def __init__(self, name: str, device: str):
        self.tokenizer = BertTokenizer.from_pretrained(name)
        self.model = BertModel.from_pretrained(name)
        self.model.to(device)
        self.device = device

    def extract_and_align(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        w_lists, feature_lists = [], []

        for i in range(len(inputs['input_ids'])):
            ids = inputs['input_ids'][i].cpu()
            features = [layer_feat[i].cpu() for layer_feat in outputs['hidden_states']]
            w_list, feature_list = self.align(
                ids, 
                features,
            )
            w_lists.append(w_list)
            feature_lists.append(feature_list)
        
        return w_lists, feature_lists
    
    def align(self, ids, features):
        
        segs = self._get_segs(ids)

        w_list = []
        feature_list = []

        for w, start, end in segs:

            if w == '[CLS]' or w == '[SEP]' or w == '[PAD]':
                continue

            w_list.append(w)
            feature_list.append([
                self._reduce(layer_feat[start:end+1])
                for layer_feat in features
            ])
        
        return w_list, feature_list


    def _get_segs(self, ids):
        labels = self.tokenizer.convert_ids_to_tokens(ids)
        segs = []
        head = 0
        while head < len(labels):
            w = labels[head]
            tail = head + 1
            while tail < len(labels) and labels[tail][:2] == '##':
                w += labels[tail][3:]
                tail += 1
            segs.append((w, head, tail-1))
            head = tail
        return segs

    def _reduce(self, feature, mode='mean'):

        if mode == 'mean':
            return torch.mean(feature, dim=0)
        if mode == 'max':
            return torch.max(feature, dim=0)[0]
        else:
            raise NotImplementedError
        

class WavFeatureExtractor:

    def __init__(self, feature: str, device: str, feature_selection='hidden_states'):
        
        self.extractor = FeatureExtractor(feature, device, feature_selection)
    
    def extract_and_align(self, wavs, tgs, tg_index=0):
        
        assert len(wavs) == len(tgs)
        raw_features = self.extractor(wavs)

        w_lists = []
        feature_lists = []

        for i in range(len(tgs)):

            w_list, feature_list = self.align(
                [layer_feat[i].cpu() for layer_feat in raw_features],
                tgs[i][tg_index]
            )
            w_lists.append(w_list)
            feature_lists.append(feature_list)
    
        return w_lists, feature_lists
    
    def align(self, feature, tg):

        w_list = []
        feature_list = []

        feature_len = len(feature[0])

        for seg in tg:
            w = seg.mark
            if w == '':
                continue
            begin = int(seg.minTime*self.extractor.frame_rate)
            end = min(int(seg.maxTime*self.extractor.frame_rate), feature_len)
            wfeat = [
                self._reduce(layer_feat[begin:end], mode='mean')
                for layer_feat in feature
            ]

            w_list.append(w)
            feature_list.append(wfeat)
        
        return w_list, feature_list

    def _reduce(self, feature, mode='mean'):

        if mode == 'mean':
            return torch.mean(feature, dim=0)
        if mode == 'max':
            return torch.max(feature, dim=0)[0]
        else:
            raise NotImplementedError
        

#wrapper for s3prl feature extractor
class FeatureExtractor:

    def __init__(self, feature: str, device: str, feature_selection='hidden_states'):
        
        self.extractor = torch.hub.load('s3prl/s3prl', feature).to(device)
        self.extractor.eval()
        self.feature_selection = feature_selection

        with torch.no_grad():
            fake_wavs = [torch.rand(SAMPLE_RATE).to(device)]

        fake_features = self.extractor(fake_wavs)[self.feature_selection]
        self.feat_dim = fake_features[0].size(-1)
        self.frame_rate = 100 if fake_features[0].size(1) > 50 else 50
        self.n_layers = len(fake_features)

    def __call__(self, wavs: list):

        with torch.no_grad():
            feature = self.extractor(wavs)[self.feature_selection]
        return feature

