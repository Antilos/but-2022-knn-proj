import torch

def relative_loss(outputs, labels):
    """_summary_

    Args:
        outputs (tensor): shape (batch, 1, width, height) (second dim is channel)
        labels (tensor): shape (batch, query, 5), where third dim is (x1, y1, x2, y2, r)

    Returns:
        _type_: _description_
    """
    batch_size = outputs.size()[0]

    outputs = outputs.squeeze() # remove channel

    # get indexes
    row1 = labels[:,:,:4][:,:,0]
    col1 = labels[:,:,:4][:,:,1]
    row2 = labels[:,:,:4][:,:,2]
    col2 = labels[:,:,:4][:,:,3]

    # get values on query indexes
    left = outputs[:, row1, col1][:,0] # it duplicated queries over batches, that's why there is that weird index at the end
    right = outputs[:, row2, col2][:,0]

    # compute loss, zero where rank doesn't match
    zeros = torch.zeros_like(labels[:,:,4], dtype=torch.float)

    val1 = torch.where(labels[:,:,4] == 1 , torch.log(1 + torch.exp(-left + right)), zeros)
    val2 = torch.where(labels[:,:,4] == -1 , torch.log(1 + torch.exp(left - right)), zeros)
    val3 = torch.where(labels[:,:,4] == 0 , torch.square(left - right).type(torch.float), zeros)

    return torch.sum(val1 + val2 + val3, dim=1)