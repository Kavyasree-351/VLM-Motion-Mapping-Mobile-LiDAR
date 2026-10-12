"""Check the Windows attention adapter preserves broadcast semantics."""
import torch
import try_sam as trial
torch.manual_seed(2026)
for batches in [(1,1,3),(3,3,3),(3,1,3)]:
    q=torch.randn(batches[0],2,17,8)
    k=torch.randn(batches[1],2,19,8)
    v=torch.randn(batches[2],2,19,8)
    expected=trial.original_attention(q,k,v)
    actual=trial.portable_attention(q,k,v)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    print(batches,'max_difference',float((actual-expected).abs().max()))
print('Three attention broadcast cases passed')
