#include <pybind11/pybind11.h>

void init_triton_tlx_wave(pybind11::module &&m) {
  m.doc() = "TLX Wave backend Python module";
}
