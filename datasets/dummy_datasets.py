import torch
import torch.utils.data as data
'''
AdApted from https://github.com/corenel/pytorch-atda/
'''


class DummyDataset(data.Dataset):
    """Slice dataset and eeplace labels of dataset with pseudo ones."""

    def __init__(self, original_dataset, excerpt, pseudo_labels):
        """Init DummyDataset."""
        super(DummyDataset, self).__init__()
        assert len(excerpt) == pseudo_labels.size(0), \
            "Size of excerpt images({}) and pseudo labels({}) aren't equal." \
            .format(len(excerpt), pseudo_labels.size(0))
        self.dataset = original_dataset
        self.excerpt = excerpt
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, index):
        """Get images and target for data loader."""
        idx = self.excerpt[index]
        images, _, metadata = self.dataset[idx.squeeze().cpu().numpy()]
        return images, self.pseudo_labels[index], metadata

    def __len__(self):
        """Return size of dataset."""
        return len(self.excerpt)


def get_dummy(original_dataset, excerpt, pseudo_labels,
              need_dataset=False,
              batch_size=16):
    """Get DummyDataset loader."""
    dummy_dataset = DummyDataset(original_dataset, excerpt, pseudo_labels)

    if need_dataset:
        return dummy_dataset
    else:
        dummy_data_loader = torch.utils.data.DataLoader(
            dataset=dummy_dataset,
            batch_size=batch_size,
            shuffle=True)
        return dummy_data_loader
