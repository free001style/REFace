import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_size = len(dataset_items)
    shape = dataset_items[0]["target_img"].shape
    result_batch = {
        "target_img": torch.zeros((batch_size, *shape)),
        "inpaint_img": torch.zeros((batch_size, *shape)),
        "corrupt_img": torch.zeros((batch_size, *shape))
        if dataset_items[0]["corrupt_img"] is not None
        else None,
        "source_img": torch.zeros((batch_size, *shape)),
        "mask": torch.zeros((batch_size, 1, shape[-2], shape[-1])),
    }
    for i in range(batch_size):
        result_batch["target_img"][i] = dataset_items[i]["target_img"]
        result_batch["inpaint_img"][i] = dataset_items[i]["inpaint_img"]
        result_batch["source_img"][i] = dataset_items[i]["source_img"]
        if result_batch["corrupt_img"] is not None:
            result_batch["corrupt_img"][i] = dataset_items[i]["corrupt_img"]
        result_batch["mask"][i] = dataset_items[i]["mask"].unsqueeze(0)

    return result_batch
