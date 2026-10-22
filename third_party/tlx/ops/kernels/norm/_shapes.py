from __future__ import annotations

from itertools import product
from typing import NamedTuple

from .._shape_suites import FocusRegistry, FocusSuite


class NormShape(NamedTuple):
    m: int
    n: int
    family: str = 'plain'
    norm: str = 'layer'
    direction: str = 'fwd'
    variant: str = 'default'
    qrows: int = 1
    jagged: bool = False


def _geo_cases():
    result = []

    def add(family, norm, m, n, **options):
        result.append({'family': family, 'norm': norm, 'm': m, 'n': n, **options})

    for (m, n), (family, norm), direction in product(
        ((128, 384), (8192, 384), (1048576, 384), (8192, 1024), (8192, 4096), (1024, 16384)),
        (('plain', 'layer'), ('plain', 'rms'), ('softmax', 'softmax')), ('fwd', 'bwd')):
        add(family, norm, m, n, direction=direction)
    for norm in ('layer', 'rms'):
        for m, n in ((128, 384), (8192, 384), (1048576, 384), (4096, 256), (4096, 512), (4096, 2048)):
            add('mxfp8', norm, m, n)
        add('mxfp8', norm, 102400, 384, jagged=True)
        for variant in ('raw', 'normalized', 'fp8'):
            add('saved', norm, 8192, 384, variant=variant)
        for m in (128, 8192, 131072):
            for n in (128, 256, 384):
                add('mxfp4', norm, m, n)
            for qrows in (1, 16):
                add('nvfp4', norm, m, 384, qrows=qrows)
    for family in ('preserving', 'two_pass', 'autograd', 'gdpa', 'gdpa_precomputed', 'stacked', 'nvfp4_backward'):
        add(family, 'rms' if family == 'two_pass' else 'layer', 8192, 384)
    for n, variant in product((256, 384), ('weighted', 'bf16', 'from_output')):
        add('dx_quant', 'rms', 8192, n, variant=variant)
    for direction in ('fwd', 'bwd'):
        add('softmax', 'softmax', 128, 65536, direction=direction)
    for c in result:
        c.setdefault('direction', 'bwd' if c['family'] in ('saved', 'dx_quant', 'nvfp4_backward') else 'fwd')
    return result


SYNTHETIC = tuple(
    NormShape(33, 128, family, norm, direction)
    for family in ('plain', 'softmax', 'mxfp8', 'mxfp4', 'nvfp4')
    for norm in (('softmax', ) if family == 'softmax' else ('layer', 'rms'))
    for direction in (('fwd', 'bwd') if family in ('plain', 'softmax') else ('fwd', )))
GEO = tuple(
    NormShape(
        **{
            key: c.get(key, default)
            for key, default in zip(NormShape._fields, (0, 0, 'plain', 'layer', 'fwd', 'default', 1, False))
        })
    for c in _geo_cases())
FOCUS_SUITES = (FocusSuite(name='sm100_1', op='norm_fwd', shapes=GEO),
                FocusSuite(name='sm90_1', op='norm_fwd',
                           shapes=tuple(s for s in GEO if s.family in ('plain', 'softmax'))))
FOCUS = FocusRegistry('norm_fwd', FOCUS_SUITES, {'sm100': ('sm100_1', ), 'sm90': ('sm90_1', )})
CORRECTNESS_SHAPES = tuple(dict.fromkeys((*SYNTHETIC, *GEO)))


def label(shape):
    return f'{shape.family}:{shape.norm}:{shape.m}x{shape.n}:{shape.variant}:q{shape.qrows}:j{int(shape.jagged)}'


def flops(shape):
    return shape.m * shape.n * (12 if shape.direction == 'bwd' else 8)
