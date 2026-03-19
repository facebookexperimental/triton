from triton._C.libtriton import ir, passes
import hashlib
import pathlib


def get_key():
    return pathlib.Path(__file__).read_text()


def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

