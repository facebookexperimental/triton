import os as _os, sys as _sys
_D=_os.path.dirname(_os.path.abspath(__file__)); _sys.path[:0]=[_D, _os.path.join(_D, '..')]  # find l10b + kernels
import sys,os; sys.path.insert(0,'.'); os.environ.pop("TRITON_USE_META_WS",None)
import torch as T, l10b, triton_bw_cross_attention as xa
l10b.force(2); D=128;H=1;Lq=128;Lkv=128
def wrap(var,ws,q,k,v,do,so_kv,mkv,asc):
    for t in (q,): t.grad=None
    xa.set_bwd_variant(var); os.environ["TRITON_USE_META_WS"]=ws
    out=xa.triton_bw_hstu_mha_wrapper(max_seq_len=mkv,alpha=1.0/D,q=q,k=k,v=v,seq_offsets=so_kv,attn_scale=asc,max_q_len=Lq,seq_offsets_q=T.tensor([0,Lq],device='cuda',dtype=T.int64),num_softmax_heads=0,shared_kv=False,enable_tma=True)
    out.backward(do); return q.grad.clone()
T.manual_seed(0)
q=T.randn(Lq,H,D,device='cuda',dtype=T.bfloat16,requires_grad=True)
k=T.randn(Lkv,H,D,device='cuda',dtype=T.bfloat16)
v=T.randn(Lkv,H,D,device='cuda',dtype=T.bfloat16)
do=T.randn(Lq,H,D,device='cuda',dtype=T.bfloat16)
asc=T.tensor(1.0/Lkv,device='cuda',dtype=T.float32)
sk2=T.tensor([0,Lkv],device='cuda',dtype=T.int64); sk1=T.tensor([0,64],device='cuda',dtype=T.int64)
def K(s): kk=k[s].clone(); vv=v[s].clone(); return kk,vv
dqr=wrap(xa.BwdVariant.TRITON_REDQ,"0",q,*K(slice(0,Lkv)),do,sk2,Lkv,asc)
dqa=wrap(xa.BwdVariant.TRITON_AUTOWS,"1",q,*K(slice(0,Lkv)),do,sk2,Lkv,asc)
perr=(dqa-dqr).abs().amax(dim=(1,2)); bad=perr>5e-3
print("bad Q blocks:", sorted(set((T.arange(Lq,device='cuda')[bad]//64).tolist())), "of 2")
p0=wrap(xa.BwdVariant.TRITON_REDQ,"0",q,*K(slice(0,64)),do,sk1,64,asc)     # kv block 0 only
p1=wrap(xa.BwdVariant.TRITON_REDQ,"0",q,*K(slice(64,128)),do,sk1,64,asc)   # kv block 1 only
print("decomp valid? |dqr-(p0+p1)| max =", (dqr-(p0+p1)).abs().max().item())
b=slice(64,128)  # Q-block 1 (the bad one)
for nm,c in [("dqr correct",dqr),("p0 (kv1 dropped)",p0),("p1 (kv0 dropped)",p1),("p0+2p1",p0+2*p1),("2p0+p1",2*p0+p1),("p0+p1 (=dqr)",p0+p1)]:
    print(f"  dqa[Qblk1] vs {nm:20s}: maxabs {(dqa[b]-c[b]).abs().max().item():.3e}  relL2 {(T.norm(dqa[b].float()-c[b].float())/(T.norm(c[b].float())+1e-9)).item():.3e}")

# Isolate the kv1-term that autoWS reduced into the bad Q-block1:
kv1term = dqa[b] - p0[b]          # remove the (correct) kv0 contribution
print("\n=== what kv1-term did autoWS put in Q-block1? ===")
for nm,c in [("p1[q1] CORRECT (kv1,q1)", p1[b]),
             ("p1[q0] stale prev-Q (kv1,q0)", p1[0:64]),
             ("p0[q0] (kv0,q0)", p0[0:64]),
             ("zero (dropped)", T.zeros_like(p1[b]))]:
    print(f"  kv1term vs {nm:28s}: maxabs {(kv1term-c).abs().max().item():.3e}  relL2 {(T.norm(kv1term.float()-c.float())/(T.norm(c.float())+1e-9)).item():.3e}")
print("  |kv1term| max =", kv1term.abs().max().item(), " |p1[q1]| max =", p1[b].abs().max().item())
