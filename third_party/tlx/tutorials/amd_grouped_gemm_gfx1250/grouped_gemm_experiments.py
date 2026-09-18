"""Opt-in code-generation experiments for the square grouped GEMM hybrid.

These transformations are scoped to this tutorial's kernel. They use Triton's
cache-keyed stage hook so the ordinary compiler and tensor API stay unchanged.
The LLVM rewrite deliberately rejects unfamiliar code generation.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re

from triton import knobs


def validate_cluster_config(m_list, n, num_programs, *, block_m, block_n, block_k, group_m, tdm_pipeline_depth,
                            l2_prefetch_distance, c_staging_mode, cross_tile_prefetch, auto_config, xcd_remap_mode,
                            num_xcds, xcd_chunk):
    """Require identical cluster control flow and the tested operand-sharing map."""
    if (block_m, block_n, block_k, group_m, tdm_pipeline_depth, l2_prefetch_distance,
            c_staging_mode) != (256, 256, 128, 4, 2, 0, 0) or not cross_tile_prefetch or auto_config:
        raise ValueError("cluster experiments require the 256x256x128 depth-2 hybrid, group_m=4, "
                         "no L2 prefetch, and auto_config=False")
    if (xcd_remap_mode, num_xcds, xcd_chunk) != ("chunked", 8, 2):
        raise ValueError("cluster experiments require chunked XCD remapping with num_xcds=8 and xcd_chunk=2")
    if not m_list or min(m_list) <= 0 or len(set(m_list)) != 1 or m_list[0] % (4 * block_m) or n % (2 * block_n):
        raise ValueError(
            "cluster experiments require equal nonempty groups, M divisible by 1024, and N divisible by 512")
    tiles_per_group = (m_list[0] // block_m) * (n // block_n)
    if num_programs % 16 or tiles_per_group % num_programs:
        raise ValueError("cluster experiments require num_programs divisible by 16 and each group's tile count "
                         "divisible by num_programs; set num_programs explicitly if necessary")


def _reuse_adjacent_wmma(asm):
    """Reuse only an unchanged physical operand of the immediately preceding WMMA."""
    previous = None
    output = []
    operand = r"v\[(\d+):(\d+)\](?: /\*v\[(\d+):(\d+)\]\*/)?"
    instruction = re.compile(r"\s*v_wmma_f32_16x16x32_f16\s+" + r",\s*".join([operand] * 4) + r"\s*$")
    for line in asm.splitlines(keepends=True):
        if not line.strip() or line.lstrip().startswith((".loc\t", ".loc ", ";", "//")):
            output.append(line)
            continue
        match = instruction.fullmatch(line)
        if match is None:
            # Labels, register-bank changes, and all intervening instructions
            # invalidate the conservative adjacency proof.
            previous = None
        else:
            regs = match.groups()
            # LLVM prints the complete physical register range in a comment
            # when the operand uses a high bank. Operand banks can differ even
            # without an intervening s_set_vgpr_msb.
            current = tuple(
                tuple(map(int, regs[i + 2:i + 4] if regs[i + 2] is not None else regs[i:i + 2]))
                for i in range(0, len(regs), 4))
            flags = []
            if previous is not None:
                for index, flag in ((1, "matrix_a_reuse"), (2, "matrix_b_reuse")):
                    lo, hi = current[index]
                    overwritten = not (previous[0][1] < lo or hi < previous[0][0])
                    if previous[index] == current[index] and not overwritten:
                        flags.append(flag)
            if flags:
                line = line.rstrip() + " " + " ".join(flags) + "\n"
            previous = current
        output.append(line)
    return "".join(output)


def _cluster_llir(src, size, multicast, synchronization):
    # Name numeric SSA values and the implicit entry block before inserting
    # instructions, which would otherwise invalidate LLVM's numbering.
    header = re.search(r"(?m)^define amdgpu_kernel [^\n]+\{\n", src)
    if header is None or '"amdgpu-cluster-dims"="1,1,1"' not in src:
        raise RuntimeError("cluster experiment expected a single-workgroup LLVM kernel")
    entry = max(map(int, re.findall(r"%(\d+)\b", header[0]))) + 1
    src = src[:header.end()] + str(entry) + ":\n" + src[header.end():]
    src = re.sub(r"%(\d+)\b", r"%v\1", src)
    src = re.sub(r"(?m)^(\d+):", r"v\1:", src)
    src = src.replace('"amdgpu-cluster-dims"="1,1,1"', f'"amdgpu-cluster-dims"="{size},1,1"')

    barrier = "call void @llvm.amdgcn.s.barrier()"
    pieces = src.split(barrier)
    if len(pieces) < 5:
        raise RuntimeError("cluster experiment could not identify the input handoff barriers")
    src = pieces[0]
    for piece in pieces[1:]:
        # Every workgroup issues its own TDM request and waits for its local
        # completion. The cluster rendezvous protects remote input slots from
        # overwrite; C slots are private. Search calls, not declarations.
        refill = "call void @llvm.amdgcn.tensor.load.to.lds(" in piece
        use_cluster = synchronization == "all" or refill
        src += ("call void @llvm.amdgcn.s.cluster.barrier()" if use_cluster else barrier) + piece
    src += "\ndeclare void @llvm.amdgcn.s.cluster.barrier()\n"
    if not multicast:
        return src

    # Fused A/B loads assign waves 0/1 to A and 2/3 to B. Require the exact
    # wave-id predicate rather than guessing which descriptor is B.
    wave = re.search(r"(%[\w.]+) = (?:tail )?call i32 @llvm.amdgcn.wave.id\(\)", src)
    cond = re.search(r"(%[\w.]+) = icmp ugt i32 " + re.escape(wave[1]) + r", 1\n", src) if wave else None
    if cond is None:
        raise RuntimeError("cluster experiment could not identify the fused A/B wave split")
    entry = re.search(r"(?m)^v\d+:\n", src)
    if re.search(r"(?m)^[\w.$]+:", src[entry.end():wave.start()]):
        raise RuntimeError("cluster experiment expected wave.id in the entry block")
    # A single group lets LLVM move the existing predicate into the tile loop,
    # after the priming loads. Define our predicate in the entry block so it
    # dominates every multicast load, including priming and peeled refills.
    wave_end = src.index("\n", wave.end()) + 1
    src = src[:wave_end] + f"  %cluster_is_b = icmp ugt i32 {wave[1]}, 1\n" + src[wave_end:]
    if size == 4:
        entry = re.search(r"(?m)^v\d+:\n", src)
        rank = ("  %cluster_rank = call i32 @llvm.amdgcn.cluster.workgroup.id.x()\n"
                "  %cluster_row = and i32 %cluster_rank, 1\n"
                "  %cluster_col = and i32 %cluster_rank, 2\n"
                "  %cluster_amask = shl i32 5, %cluster_row\n"
                "  %cluster_bmask = shl i32 3, %cluster_col\n")
        src = src[:entry.end()] + rank + src[entry.end():]
        src += "\ndeclare i32 @llvm.amdgcn.cluster.workgroup.id.x()\n"

    count = 0

    def patch_load(match):
        nonlocal count
        count += 1
        mask, desc = f"%cluster_mask_{count}", f"%cluster_desc_{count}"
        if size == 2:
            code = (f"  {mask} = select i1 %cluster_is_b, <8 x i32> <i32 3, i32 0, i32 0, i32 0, "
                    "i32 0, i32 0, i32 0, i32 0>, <8 x i32> zeroinitializer\n")
        else:
            code = (f"  %cluster_scalar_{count} = select i1 %cluster_is_b, i32 %cluster_bmask, i32 %cluster_amask\n"
                    f"  {mask} = insertelement <8 x i32> zeroinitializer, i32 %cluster_scalar_{count}, i64 0\n")
        return code + f"  {desc} = or <8 x i32> {match[2]}, {mask}\n" + match[1] + desc + match[3]

    src = re.sub(
        r"(  (?:tail )?call void @llvm.amdgcn.tensor.load.to.lds\(<4 x i32> %[\w.]+, <8 x i32> )"
        r"(%[\w.]+)(,[^\n]+\n)", patch_load, src)
    if count < 6 or count != src.count("call void @llvm.amdgcn.tensor.load.to.lds("):
        raise RuntimeError("cluster experiment did not recognize every fused input load")
    return src


@contextmanager
def experimental_codegen(cluster_size=1, cluster_multicast=True, cluster_sync="all", operand_reuse=False,
                         jit_kernel=None):
    """Temporarily install a cache-keyed hook; restore the caller's hook on exit.

    The launch grid counts clusters when cluster_size > 1. The caller must
    divide its physical program count by cluster_size and validate the shape.
    Pass jit_kernel when launching to isolate the native specialization cache.
    Like Triton's runtime knobs, this context is intended for serial use.
    """
    if cluster_size not in (1, 2, 4) or cluster_sync not in ("all", "refill"):
        raise ValueError("cluster_size must be 1, 2, or 4; cluster_sync must be all or refill")
    if cluster_size == 1 and not operand_reuse:
        yield
        return
    previous = knobs.runtime.add_stages_inspection_hook
    key = repr((cluster_size, cluster_multicast, cluster_sync, operand_reuse)) + Path(__file__).read_text()
    if previous is not None:
        key += previous()[0]
    digest = hashlib.sha256(key.encode()).hexdigest()

    def hook(backend=None, stages=None, options=None, language=None, capability=None):
        if stages is None:
            return key, digest
        if previous is not None:
            previous(backend, stages, options, language, capability)
        if getattr(options, "arch", None) != "gfx1250":
            raise ValueError("grouped GEMM code-generation experiments require gfx1250")
        original_llir, original_amdgcn = stages["llir"], stages["amdgcn"]

        def make_llir(src, metadata):
            llir = original_llir(src, metadata)
            if "define amdgpu_kernel void @grouped_gemm_tdm_kernel(" in llir and cluster_size > 1:
                llir = _cluster_llir(llir, cluster_size, cluster_multicast, cluster_sync)
                # Preserve per-workgroup tensor layouts, but keep the LLVM
                # cluster attribute and the HIP launch metadata consistent.
                metadata["num_ctas"] = cluster_size
            return llir

        def make_amdgcn(src, metadata):
            asm = original_amdgcn(src, metadata)
            if metadata["name"] == "grouped_gemm_tdm_kernel" and operand_reuse:
                asm = _reuse_adjacent_wmma(asm)
            return asm

        stages["llir"], stages["amdgcn"] = make_llir, make_amdgcn

    with knobs.runtime.scope():
        knobs.runtime.add_stages_inspection_hook = hook
        # The runtime bypasses C-cache lookup under a stage hook, but still
        # populates it after compiling. That cache does not include the hook's
        # key. Prevent experimental launches from replacing the ordinary entry.
        previous_c_cache = jit_kernel.c_cache if jit_kernel is not None else None
        try:
            if jit_kernel is not None:
                jit_kernel.c_cache = False
            yield
        finally:
            if jit_kernel is not None:
                jit_kernel.c_cache = previous_c_cache
