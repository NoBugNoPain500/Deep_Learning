from types import SimpleNamespace

from src.data import get_transforms, load_splits, make_loaders


def test_split_labels_and_disjoint(fake_data):
    root, split_dir = fake_data
    train, val, test = load_splits(root, split_dir)
    assert train.set_index("file").label.isin([0, 1]).all()
    assert set(train.file).isdisjoint(val.file)
    assert set(train.file).isdisjoint(test.file)
    assert set(val.file).isdisjoint(test.file)
    assert train[train.gender == "Male"].label.eq(0).all()


def test_eval_transform_has_no_random_ops():
    names = [type(item).__name__ for item in get_transforms(False).transforms]
    assert names == ["Resize", "ToTensor", "Normalize"]


def test_smoke_loaders(fake_data):
    root, split_dir = fake_data
    cfg = SimpleNamespace(
        data_dir=root,
        split_dir=split_dir,
        seed=42,
        batch=8,
        num_workers=0,
        augment=True,
    )
    loaders = make_loaders(cfg, smoke=True)
    assert [len(loader.dataset) for loader in loaders] == [28, 4, 8]
