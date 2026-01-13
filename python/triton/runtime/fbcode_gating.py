# facebook begin T177165732
import importlib

IS_FBCODE = None


def is_fbcode_dependant():
    global IS_FBCODE
    if IS_FBCODE is not None:
        return IS_FBCODE
    # TODO: Stop doing import sniffing to test if you're in fbcode or not;
    # it should just be immediately obvious from the build system (see what
    # we did for caffe2/fb/_utils_internal.py in D65833409)
    if importlib.util.find_spec("triton.fb") is not None:
        IS_FBCODE = True
    else:
        IS_FBCODE = False
    return IS_FBCODE


# facebook end T177165732
