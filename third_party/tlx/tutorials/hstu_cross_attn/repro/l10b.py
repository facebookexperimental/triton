import os as _os, sys as _sys
_D=_os.path.dirname(_os.path.abspath(__file__)); _sys.path[:0]=[_D, _os.path.join(_D, '..')]  # find l10b + kernels
# Vary KV length independently: does bad-Q-region depend on #KV blocks or #Q blocks?
import os,sys; sys.path.insert(0,'.')
os.environ.pop("TRITON_USE_META_WS",None)
import torch, triton_bw_cross_attention as xa
D=128;H=2;Z=2
def force(ns):
    for fn in (xa._attn_fwd_triton,):
        if hasattr(fn,"configs"): c=fn.configs[0]; c.num_stages=2; fn.configs=[c]
    c=xa._hstu_attn_bwd_redq.configs[0]; c.num_stages=ns; c.kwargs["BLOCK_M"]=64; c.kwargs["BLOCK_N"]=64; xa._hstu_attn_bwd_redq.configs=[c]
    xa.set_fwd_variant(xa.FwdVariant.TRITON)
def go(Lq, Lkv, ns=2):
    # asymmetric: Q length Lq, KV length Lkv, per sequence
    force(ns)
    torch.manual_seed(0); total_q=Z*Lq; total_kv=Z*Lkv
    q=torch.randn(total_q,H,D,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn(total_kv,H,D,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    v=torch.randn(total_kv,H,D,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    so_kv=torch.tensor([0,Lkv,2*Lkv],device="cuda",dtype=torch.int64)
    so_q=torch.tensor([0,Lq,2*Lq],device="cuda",dtype=torch.int64)
    asc=torch.tensor(1.0/Lkv,device="cuda",dtype=torch.float32)
    do=torch.randn(total_q,H,D,device="cuda",dtype=torch.bfloat16)
    def run(variant,ws):
        for t in (q,k,v):
            if t.grad is not None: t.grad=None
        xa.set_bwd_variant(variant); os.environ["TRITON_USE_META_WS"]=ws
        out=xa.triton_bw_hstu_mha_wrapper(max_seq_len=Lkv,alpha=1.0/D,q=q,k=k,v=v,seq_offsets=so_kv,attn_scale=asc,max_q_len=Lq,seq_offsets_q=so_q,num_softmax_heads=H,shared_kv=False,enable_tma=True)
        out.backward(do); return q.grad.clone()
    dqr=run(xa.BwdVariant.TRITON_REDQ,"0"); dqw=run(xa.BwdVariant.TRITON_AUTOWS,"1")
    pr=(dqw-dqr).abs().amax(dim=(1,2)); rm=torch.arange(total_q,device="cuda")%Lq; bad=pr>5e-3
    bm=sorted(set(rm[bad].tolist()))
    print(f"Lq={Lq}({Lq//64}blk) Lkv={Lkv}({Lkv//64}blk) ns={ns}: dq bad={bad.sum().item()}/{total_q} Qmod=[{bm[0] if bm else '-'}..{bm[-1] if bm else '-'}]")
if __name__=="__main__":
    import sys
    go(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 2)
