from torch.utils.data import Dataset
import torch

class InferenceDataset(Dataset):
    """
    Dataset for inference.
    
    Arguments:
        dataset: The dataset to use.
        prefix: The prefix to use for the prompt.
    """

    def __init__(self, dataset, prefix = 'Convert text into SQL statements by providing a database schema and a query, and generate the corresponding SQL statement.'):
        self.dataset = dataset
        self.prefix = prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]

        return self.prefix + sample['input_text'].split('<|sql|>')[0] + '<|sql|>'