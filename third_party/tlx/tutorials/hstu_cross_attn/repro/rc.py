import os as _os, sys as _sys
_D=_os.path.dirname(_os.path.abspath(__file__)); _sys.path[:0]=[_D, _os.path.join(_D, '..')]  # find l10b + kernels
import sys,os; sys.path.insert(0,'.'); os.environ.pop("TRITON_USE_META_WS",None)
import torch, l10b
V=os.environ.get("V","autows"); l10b.force(2)
import triton_bw_cross_attention as xa
D=128;H=1;Z=1;Lq=128;Lkv=128
if V=="autows": xa.set_bwd_variant(xa.BwdVariant.TRITON_AUTOWS); os.environ["TRITON_USE_META_WS"]="1"
elif V=="tlx": xa.set_bwd_variant(xa.BwdVariant.TLX); os.environ["TRITON_USE_META_WS"]="0"
q=torch.randn(Z*Lq,H,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(Z*Lkv,H,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(Z*Lkv,H,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
so_kv=torch.tensor([0,Lkv],device='cuda',dtype=torch.int64)
so_q=torch.tensor([0,Lq],device='cuda',dtype=torch.int64)
asc=torch.tensor(1.0/Lkv,device='cuda',dtype=torch.float32)
do=torch.randn(Z*Lq,H,D,device='cuda',dtype=torch.bfloat16)
out=xa.triton_bw_hstu_mha_wrapper(max_seq_len=Lkv,alpha=1.0/D,q=q,k=k,v=v,seq_offsets=so_kv,attn_scale=asc,max_q_len=Lq,seq_offsets_q=so_q,num_softmax_heads=H,shared_kv=False,enable_tma=True)
out.backward(do); torch.cuda.synchronize(); print("done",V)
