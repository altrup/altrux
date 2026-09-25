import torch

from experiments.lama_ckl.training import epoch_batches, train_document_epoch


def test_epoch_batches_are_fixed_by_seed_and_cover_each_document_once():
    first = epoch_batches(7, 3, 42)
    second = epoch_batches(7, 3, 42)

    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(7))
    assert [len(batch) for batch in first] == [3, 3, 1]


def test_train_document_epoch_masks_padding_and_updates_once_per_batch():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.output = torch.nn.Linear(4, 16)

        def forward(self, ids, state=None):
            assert state is None
            return self.output(self.embedding(ids)), object()

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.output.weight.detach().clone()

    result = train_document_epoch(
        model,
        optimizer,
        [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
        [[0, 1], [2]],
        pad_id=0,
        device="cpu",
        label="test",
    )

    assert result["optimizer_steps"] == 2
    assert result["token_gradients"] == 6
    assert not torch.equal(before, model.output.weight)
