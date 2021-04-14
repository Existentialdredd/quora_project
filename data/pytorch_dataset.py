import torch
from torch.utils.data import Dataset
from data.schemas import QuoraObservationSet


class TorchQuoraDataset(Dataset):

    def __init__(self, data: QuoraObservationSet):
        self.data = data

    def __len__(self):
        return len(self.data.observations)

    def __getitem__(self, idx):
        """
        """
        output = dict()
        output['token_ids'] = self.data.observations[idx].token_ids
        output['token_type_ids'] = self.data.observations[idx].token_types
        output['attention_mask'] = self.data.observations[idx].attention_mask
        output['label'] = torch.tensor(self.data.observations[idx].label)

        return output
