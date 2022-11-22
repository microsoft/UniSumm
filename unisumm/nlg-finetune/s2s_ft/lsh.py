import torch


def hash_vectors( vectors, num_buckets, num_attention_heads, attention_mask=None, increase_num_buckets=False, hash_seed=528):
    batch_size = vectors.shape[0]

    # See https://arxiv.org/pdf/1509.02897.pdf
    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    assert (
            num_buckets % 2 == 0
    ), "There should be an even number of bucktes, but `self.num_bucktes`: {}".format(num_buckets)
    rotation_size = num_buckets
    num_buckets = num_buckets

    # remove gradient
    vectors = vectors.detach()

    torch.manual_seed(hash_seed)

    rotations_shape = (num_attention_heads, vectors.shape[-1],  rotation_size // 2)
    # create a random self.attention_head_size x num_hashes x num_buckets/2
    random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)

    # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
    rotated_vectors = torch.einsum("bmtd,mdr->bmtr", vectors, random_rotations)

    if isinstance(num_buckets, int) or len(num_buckets) == 1:
        rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
        buckets = torch.argmax(rotated_vectors, dim=-1)
    else:
        # Get the buckets for them and combine.
        buckets, cur_sum, cur_product = None, 0, 1
        for bucket_factor in num_buckets:
            rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
            cur_sum = cur_sum + bucket_factor // 2
            rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
            if buckets is None:
                buckets = torch.argmax(rotated_vectors_factor, dim=-1)
            else:
                buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

            cur_product = cur_product * bucket_factor

    if attention_mask is not None and (attention_mask.sum().item() < batch_size * attention_mask.shape[-1]):
        # add an extra bucket for padding tokens only
        num_buckets = num_buckets + 1
        # assign padding tokens extra bucket
        buckets_mask = attention_mask.to(torch.uint8)[:, None, None, :].expand(buckets.shape)
        buckets = torch.where(
            buckets_mask, buckets, torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device)
        )

    # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
    # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.

    # expand to batch size and num attention heads
    # buckets = buckets.flatten(start_dim=2, end_dim=3)

    return buckets


def _stable_argsort(vector, dim):
    # this function scales the vector so that torch.argsort is stable.
    # torch.argsort is not stable on its own
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return torch.argsort(scaled_vector, dim=dim)


def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx( buckets):
    # no gradients are needed
    with torch.no_grad():
        # hash-based sort
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

        # create simple indices to scatter to, to have undo sort
        indices = (
            torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
            .view(1, 1, -1)
            .expand(sorted_bucket_idx.shape)
        )

        # get undo sort
        undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

    return sorted_bucket_idx, undo_sorted_bucket_idx

def _gather_by_expansion(vectors, idxs,  attention_head_size):
    """
    expand dims of idxs and vectors for all hashes and gather
    """
    expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, attention_head_size)
    return torch.gather(vectors, 2, expanded_idxs)

seq_len = 50
num_hashes = 5
num_attention_heads = 16
attention_head_size = 512//num_attention_heads
a = torch.randn((10,num_attention_heads,50,attention_head_size))

buckets = hash_vectors(a, 8,  num_attention_heads)

sorted_bucket_idx, undo_sorted_bucket_idx = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(buckets)
# sorted_bucket_idx_per_hash = sorted_bucket_idx % seq_len
aa = _gather_by_expansion(a, sorted_bucket_idx,  attention_head_size)

print(sorted_bucket_idx)
