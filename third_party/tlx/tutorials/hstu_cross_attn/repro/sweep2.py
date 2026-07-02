import os as _os, sys as _sys
_D=_os.path.dirname(_os.path.abspath(__file__)); _sys.path[:0]=[_D, _os.path.join(_D, '..')]  # find l10b + kernels
import sys; sys.path.insert(0,'.')
import l10b
def s(Lq,Lkv,ns):
    try: l10b.go(Lq,Lkv,ns)
    except Exception as e: print(f"Lq={Lq} Lkv={Lkv} ns={ns}: ERR {str(e)[:50]}")
print("### does the bug need >=2 Q-blocks (peel)? ###")
s(64,128,2)   # 1 Q-block, 2 KV-blocks
s(64,192,2)   # 1 Q-block, 3 KV-blocks
s(128,128,2)  # 2 Q-block, 2 KV (baseline: expect last block bad)
