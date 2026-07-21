import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EagerDiffSBA(nn.Module):
    def __init__(self, n_head, head_dim):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.raw_lambda = nn.Parameter(torch.full((1, n_head, 1, 1), math.log(0.8)))

    def forward(self, q1, k1, q2, k2, v, mask=None):
        lamb = torch.exp(self.raw_lambda)
        logits1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        work1 = logits1.float()
        work2 = logits2.float()
        lamb_val = lamb.float()
        
        if mask is not None:
            valid = mask.to(device=logits1.device, dtype=torch.bool)
            floor = torch.full_like(work1, torch.finfo(work1.dtype).min)
            zeros = torch.zeros_like(work1)
            l1_pos = torch.where(valid, work1, floor)
            l1_neg = torch.where(valid, -work1, floor)
            l2_pos = torch.where(valid, work2, floor)
            l2_neg = torch.where(valid, -work2, floor)
            row_max = torch.maximum(
                torch.maximum(l1_pos.max(dim=-1, keepdim=True).values, l1_neg.max(dim=-1, keepdim=True).values),
                torch.maximum(l2_pos.max(dim=-1, keepdim=True).values, l2_neg.max(dim=-1, keepdim=True).values),
            )
            e1_pos = torch.where(valid, torch.exp(l1_pos - row_max), zeros)
            e1_neg = torch.where(valid, torch.exp(l1_neg - row_max), zeros)
            e2_pos = torch.where(valid, torch.exp(l2_pos - row_max), zeros)
            e2_neg = torch.where(valid, torch.exp(l2_neg - row_max), zeros)
            denom = (e1_pos + e1_neg + e2_pos + e2_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work1.dtype).tiny)
            weights = torch.where(valid, ((e1_pos - e1_neg) - lamb_val * (e2_pos - e2_neg)) / denom, zeros)
        else:
            row_max = torch.maximum(
                torch.maximum(work1.max(dim=-1, keepdim=True).values, (-work1).max(dim=-1, keepdim=True).values),
                torch.maximum(work2.max(dim=-1, keepdim=True).values, (-work2).max(dim=-1, keepdim=True).values),
            )
            e1_pos = torch.exp(work1 - row_max)
            e1_neg = torch.exp(-work1 - row_max)
            e2_pos = torch.exp(work2 - row_max)
            e2_neg = torch.exp(-work2 - row_max)
            denom = (e1_pos + e1_neg + e2_pos + e2_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work1.dtype).tiny)
            weights = ((e1_pos - e1_neg) - lamb_val * (e2_pos - e2_neg)) / denom
            
        weights = weights.to(dtype=q1.dtype)
        return torch.matmul(weights, v)

class SDPADiffSBA(nn.Module):
    def __init__(self, n_head, head_dim):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.raw_lambda = nn.Parameter(torch.full((1, n_head, 1, 1), math.log(0.8)))

    def forward(self, q1, k1, q2, k2, v, mask=None):
        # 1. Q: shape (B, H, T, 2d)
        q = torch.cat([q1, q2], dim=-1)
        q_scaled = q * math.sqrt(2.0)
        
        # 2. K: shape (B, H, 4T, 2d)
        zeros_d = torch.zeros_like(k1)
        k1_pos = torch.cat([k1, zeros_d], dim=-1)
        k1_neg = torch.cat([-k1, zeros_d], dim=-1)
        k2_pos = torch.cat([zeros_d, k2], dim=-1)
        k2_neg = torch.cat([zeros_d, -k2], dim=-1)
        k_concat = torch.cat([k1_pos, k1_neg, k2_pos, k2_neg], dim=-2)
        
        # 3. V: shape (B, H, 4T, d)
        lamb_val = torch.exp(self.raw_lambda)
        v1_pos = v
        v1_neg = -v
        v2_pos = -lamb_val * v
        v2_neg = lamb_val * v
        v_concat = torch.cat([v1_pos, v1_neg, v2_pos, v2_neg], dim=-2)
        
        # 4. Mask: shape (B, H, T, 4T)
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            mask_concat = torch.cat([mask_bool, mask_bool, mask_bool, mask_bool], dim=-1)
        else:
            mask_concat = None
            
        # 5. SDPA
        y = F.scaled_dot_product_attention(
            q_scaled, k_concat, v_concat, attn_mask=mask_concat, dropout_p=0.0, is_causal=False
        )
        return y

def test_equivalence():
    torch.manual_seed(42)
    B, H, T, D = 4, 8, 32, 64
    
    q1 = torch.randn(B, H, T, D, requires_grad=True)
    k1 = torch.randn(B, H, T, D, requires_grad=True)
    q2 = torch.randn(B, H, T, D, requires_grad=True)
    k2 = torch.randn(B, H, T, D, requires_grad=True)
    v = torch.randn(B, H, T, D, requires_grad=True)
    
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T).expand(B, H, T, T)
    
    eager_model = EagerDiffSBA(H, D)
    sdpa_model = SDPADiffSBA(H, D)
    
    # Copy parameters
    sdpa_model.raw_lambda.data.copy_(eager_model.raw_lambda.data)
    
    # Forward Pass
    y_eager = eager_model(q1, k1, q2, k2, v, mask=mask)
    y_sdpa = sdpa_model(q1, k1, q2, k2, v, mask=mask)
    
    max_diff_fwd = (y_eager - y_sdpa).abs().max().item()
    print(f"Max Fwd Diff: {max_diff_fwd:.2e}")
    assert max_diff_fwd < 1e-5, f"Forward mismatch: {max_diff_fwd}"
    
    # Backward Pass
    loss_eager = y_eager.sum()
    loss_eager.backward()
    
    grad_q1_eager = q1.grad.clone()
    grad_k1_eager = k1.grad.clone()
    grad_q2_eager = q2.grad.clone()
    grad_k2_eager = k2.grad.clone()
    grad_v_eager = v.grad.clone()
    grad_lamb_eager = eager_model.raw_lambda.grad.clone()
    
    q1.grad.zero_()
    k1.grad.zero_()
    q2.grad.zero_()
    k2.grad.zero_()
    v.grad.zero_()
    
    loss_sdpa = y_sdpa.sum()
    loss_sdpa.backward()
    
    grad_q1_sdpa = q1.grad.clone()
    grad_k1_sdpa = k1.grad.clone()
    grad_q2_sdpa = q2.grad.clone()
    grad_k2_sdpa = k2.grad.clone()
    grad_v_sdpa = v.grad.clone()
    grad_lamb_sdpa = sdpa_model.raw_lambda.grad.clone()
    
    diff_q1 = (grad_q1_eager - grad_q1_sdpa).abs().max().item()
    diff_k1 = (grad_k1_eager - grad_k1_sdpa).abs().max().item()
    diff_q2 = (grad_q2_eager - grad_q2_sdpa).abs().max().item()
    diff_k2 = (grad_k2_eager - grad_k2_sdpa).abs().max().item()
    diff_v = (grad_v_eager - grad_v_sdpa).abs().max().item()
    diff_lamb = (grad_lamb_eager - grad_lamb_sdpa).abs().max().item()
    
    print(f"Max Grad Q1 Diff: {diff_q1:.2e}")
    print(f"Max Grad K1 Diff: {diff_k1:.2e}")
    print(f"Max Grad Q2 Diff: {diff_q2:.2e}")
    print(f"Max Grad K2 Diff: {diff_k2:.2e}")
    print(f"Max Grad V Diff:  {diff_v:.2e}")
    print(f"Max Grad Lambda Diff: {diff_lamb:.2e}")
    
    assert diff_q1 < 1e-4, f"Q1 Grad mismatch: {diff_q1}"
    assert diff_k1 < 1e-4, f"K1 Grad mismatch: {diff_k1}"
    assert diff_q2 < 1e-4, f"Q2 Grad mismatch: {diff_q2}"
    assert diff_k2 < 1e-4, f"K2 Grad mismatch: {diff_k2}"
    assert diff_v < 1e-4, f"V Grad mismatch: {diff_v}"
    assert diff_lamb < 1e-4, f"Lambda Grad mismatch: {diff_lamb}"
    
    print("SUCCESS: Math Equivalence Test Passed successfully!")

if __name__ == "__main__":
    test_equivalence()
