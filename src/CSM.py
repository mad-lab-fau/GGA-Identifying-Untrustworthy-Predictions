import torch
import torch.nn.functional as F

def cosine_similarity_maps(model: torch.nn.Module,
                           X: torch.Tensor,
                           sign:bool=True,
                           rescale:bool=True) -> torch.Tensor:
    '''
    computes the cosine similarity map for given input images X

    Parameters
    ---------
    model: torch model
    X: torch tensor; shape: (Batch_Size, Channels, Width, Height)
    sign: use sign of gradients to calculate cosine similarity maps
    rescale: rescale the logits before applying softmax -> solves gradient obfuscation problem of large logits

    Returns
    ---------
    return: cosine_similarity_map:
    '''

    deltas = []  # saliency maps w.r.t. all possible output classes
    X_grad = X.clone().requires_grad_()

    logits = model(X_grad)  # network output

    # rescale network output to avoid gradient obfuscation
    if rescale:
        logits = logits / torch.max(torch.abs(logits), 1, keepdim=True).values * 10

    B = logits.shape[0]  # batch size
    classes = logits.shape[-1] # output classes

    for c in range(classes):
        #  calculate loss and compute gradient w.r.t. the input of the current class
        y = torch.ones(B, device="cuda", dtype=torch.long) * c
        loss = F.cross_entropy(logits, y)
        loss.backward(retain_graph=True)
        grad = X_grad.grad

        #  take sign of gradient as in the original paper
        if sign:
            grad = torch.sign(grad)
        deltas.append(grad.detach().clone())
        X_grad.grad.zero_()
    model.zero_grad()
    deltas = torch.stack(deltas, dim=0)

    deltas = torch.max(deltas, dim=-3).values  #  take only the maximum value of all channels to compute the cosine similarity

    #  compute cosine similarity matrices
    deltas = deltas.view(classes, B, -1)
    norm = torch.norm(deltas, p=2, dim=2, keepdim=True)
    deltas = deltas / norm
    deltas = deltas.transpose(0, 1)
    csm = torch.matmul(deltas, deltas.transpose(1, 2))

    #  division by zero can lead to NaNs
    if torch.isnan(csm).any():
        raise Exception("NaNs in CSM!")
    return csm