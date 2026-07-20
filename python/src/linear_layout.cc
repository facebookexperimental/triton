#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <iostream>
#include <optional>
#include <stdexcept>

namespace py = pybind11;
using LinearLayout = mlir::triton::LinearLayout;

namespace {

mlir::MLIRContext *getLinearLayoutContext() {
  static PyObject *ctxObject = []() {
    py::module irMod = py::module::import("triton._C.libtriton.ir");
    // Keep the Python object alive for the life of the process without running
    // its destructor during interpreter shutdown (avoids segfaults).
    py::object ctx = irMod.attr("context")();
    return ctx.release().ptr();
  }();
  return py::cast<mlir::MLIRContext *>(py::handle(ctxObject));
}

mlir::triton::gpu::LocalMemOpTile
makeLocalMemOpTile(const std::vector<int32_t> &laneContig,
                   const std::vector<int32_t> &laneAddr) {
  mlir::triton::gpu::LocalMemOpTile tile;
  tile.laneContig.append(laneContig.begin(), laneContig.end());
  tile.laneAddr.append(laneAddr.begin(), laneAddr.end());
  return tile;
}

LinearLayout canonicalizeDistributedInputs(const LinearLayout &layout) {
  auto *ctx = layout.getInDimNames().begin()->getContext();
  auto original = layout.getBases();
  LinearLayout::BasesT bases;
  for (llvm::StringRef name : {"register", "lane", "warp", "block"}) {
    auto dim = mlir::StringAttr::get(ctx, name);
    auto it = original.find(dim);
    bases[dim] =
        it == original.end() ? std::vector<std::vector<int32_t>>{} : it->second;
  }
  for (const auto &[dim, dimBases] : original)
    if (!bases.contains(dim))
      bases[dim] = dimBases;
  return LinearLayout(std::move(bases), layout.getOutDims(),
                      layout.isSurjective());
}

std::vector<int32_t>
registerOrder(const mlir::triton::ColumnAction &permutation,
              mlir::StringAttr kReg, int32_t registerCount) {
  auto identity = LinearLayout::identity1D(registerCount, kReg, kReg);
  auto permuted = permutation.apply(identity);
  std::vector<int32_t> order;
  order.reserve(registerCount);
  for (int32_t reg = 0; reg < registerCount; ++reg) {
    auto result = permuted.apply({{kReg, reg}});
    order.push_back(result.front().second);
  }
  return order;
}

struct VectorizedLayoutPlan {
  LinearLayout layout;
  std::vector<int32_t> registerOrder;
  int32_t vectorElements;
};

VectorizedLayoutPlan vectorizeLayout(const LinearLayout &layout,
                                     int32_t bitwidth) {
  auto *ctx = layout.getInDimNames().begin()->getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto [vectorElements, permutation] =
      mlir::triton::largestVectorisation(ctx, layout, bitwidth);
  return {
      permutation.apply(layout),
      registerOrder(permutation, kReg, layout.getInDimSize(kReg)),
      vectorElements,
  };
}

py::dict optimalSwizzledLdStPlan(const LinearLayout &src,
                                 const LinearLayout &dst, int32_t bitwidth,
                                 int32_t numBanks,
                                 const std::vector<int32_t> &srcLaneContig,
                                 const std::vector<int32_t> &srcLaneAddr,
                                 const std::vector<int32_t> &dstLaneContig,
                                 const std::vector<int32_t> &dstLaneAddr) {
  if (bitwidth <= 0 || bitwidth > 128 || 128 % bitwidth != 0)
    throw std::invalid_argument("bitwidth must be a positive divisor of 128");
  if (numBanks <= 0 || !llvm::isPowerOf2_32(numBanks))
    throw std::invalid_argument("num_banks must be a positive power of two");
  auto srcLayout = canonicalizeDistributedInputs(src);
  auto dstLayout = canonicalizeDistributedInputs(dst);
  if (!mlir::triton::actionRemoveBroadcastedRegs(srcLayout).isIdentity() ||
      !mlir::triton::actionRemoveBroadcastedRegs(dstLayout).isIdentity())
    throw std::invalid_argument(
        "optimal swizzled ld/st plans require non-broadcast register "
        "layouts");

  auto srcTile = makeLocalMemOpTile(srcLaneContig, srcLaneAddr);
  auto dstTile = makeLocalMemOpTile(dstLaneContig, dstLaneAddr);
  auto smem = mlir::triton::gpu::optimalSwizzlingLdSt(
      srcLayout, dstLayout, bitwidth, numBanks, srcTile, dstTile);
  auto [readBankConflicts, writeBankConflicts] =
      mlir::triton::gpu::bankConflictsLdSt(srcLayout, dstLayout, smem, bitwidth,
                                           numBanks, srcTile, dstTile);

  auto *ctx = srcLayout.getInDimNames().begin()->getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto kBlock = mlir::StringAttr::get(ctx, "block");
  auto kReps = mlir::StringAttr::get(ctx, "reps");
  auto kOffset = mlir::StringAttr::get(ctx, "offset");
  int32_t repetitions = smem.getInDimSize(kReps);
  auto reps = LinearLayout::identity1D(repetitions, kReg, kReps);

  auto totalStore = srcLayout.invertAndCompose(smem);
  auto totalLoad = mlir::triton::invertAndComposeBlockLocal(smem, dstLayout);
  auto storePermutation =
      mlir::triton::regPermForDivide(totalStore, reps, /*left=*/false);
  auto loadPermutation =
      mlir::triton::regPermForDivide(totalLoad, reps, /*left=*/false);
  if (!storePermutation || !loadPermutation)
    throw std::runtime_error(
        "optimal swizzle did not produce separable repetition layouts");

  auto outerStoreOrder =
      registerOrder(*storePermutation, kReg, totalStore.getInDimSize(kReg));
  auto outerLoadOrder =
      registerOrder(*loadPermutation, kReg, totalLoad.getInDimSize(kReg));
  totalStore = storePermutation->apply(totalStore);
  totalLoad = loadPermutation->apply(totalLoad);

  auto maybeStore = divideRight(totalStore, reps);
  auto maybeLoad = divideRight(totalLoad, reps);
  if (!maybeStore || !maybeLoad)
    throw std::runtime_error(
        "optimal swizzle repetition layouts could not be divided");
  auto store = *maybeStore;
  auto load = *maybeLoad;
  int32_t storeBlocks = store.getInDimSize(kBlock);
  int32_t loadBlocks = load.getInDimSize(kBlock);
  store =
      store.reshapeOuts({{kOffset, store.getTotalOutDimSize() / storeBlocks},
                         {kBlock, storeBlocks}});
  load = load.reshapeOuts({{kOffset, load.getTotalOutDimSize() / loadBlocks},
                           {kBlock, loadBlocks}});

  auto vectorStore = vectorizeLayout(store, bitwidth);
  auto vectorLoad = vectorizeLayout(load, bitwidth);
  int32_t storeTileSize = vectorStore.layout.getInDimSize(kReg);
  int32_t loadTileSize = vectorLoad.layout.getInDimSize(kReg);
  if (store.getOutDimSize(kOffset) != load.getOutDimSize(kOffset) ||
      static_cast<int32_t>(outerStoreOrder.size()) !=
          storeTileSize * repetitions ||
      static_cast<int32_t>(outerLoadOrder.size()) != loadTileSize * repetitions)
    throw std::runtime_error(
        "optimal swizzle produced inconsistent repetition tile sizes");

  std::vector<int32_t> storeRegisters;
  std::vector<int32_t> loadRegisters;
  storeRegisters.reserve(outerStoreOrder.size());
  loadRegisters.reserve(outerLoadOrder.size());
  for (int32_t rep = 0; rep < repetitions; ++rep) {
    for (int32_t reg : vectorStore.registerOrder)
      storeRegisters.push_back(outerStoreOrder[rep * storeTileSize + reg]);
    for (int32_t reg : vectorLoad.registerOrder)
      loadRegisters.push_back(outerLoadOrder[rep * loadTileSize + reg]);
  }

  py::dict result;
  result["store_layout"] = py::cast(vectorStore.layout);
  result["load_layout"] = py::cast(vectorLoad.layout);
  result["store_registers"] = py::cast(storeRegisters);
  result["load_registers"] = py::cast(loadRegisters);
  result["store_vector_elements"] = vectorStore.vectorElements;
  result["load_vector_elements"] = vectorLoad.vectorElements;
  result["store_tile_size"] = storeTileSize;
  result["load_tile_size"] = loadTileSize;
  result["repetitions"] = repetitions;
  result["scratch_elements"] = store.getOutDimSize(kOffset);
  result["read_bank_conflicts"] = readBankConflicts;
  result["write_bank_conflicts"] = writeBankConflicts;
  return result;
}

} // namespace

void init_linear_layout(py::module &&m) {
  py::class_<LinearLayout>(m, "LinearLayout", py::module_local(false))
      .def(py::init<>())
      .def_static(
          "identity_1d",
          [](int32_t size, std::string inDim, std::string outDim) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::identity1D(size,
                                            mlir::StringAttr::get(ctx, inDim),
                                            mlir::StringAttr::get(ctx, outDim));
          },
          py::arg("size"), py::arg("inDim"), py::arg("outDim"))
      .def_static(
          "strided_1d",
          [](int32_t size, int32_t stride, std::string inDim,
             std::string outDim) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::strided1D(size, stride,
                                           mlir::StringAttr::get(ctx, inDim),
                                           mlir::StringAttr::get(ctx, outDim));
          },
          py::arg("size"), py::arg("stride"), py::arg("inDim"),
          py::arg("outDim"))
      .def_static(
          "zeros_1d",
          [](int32_t size, std::string inDim, std::string outDim,
             int32_t outDimSize) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::zeros1D(
                size, mlir::StringAttr::get(ctx, inDim),
                mlir::StringAttr::get(ctx, outDim), outDimSize);
          },
          py::arg("size"), py::arg("inDim"), py::arg("outDim"),
          py::arg("outDimSize") = 1)
      .def_static(
          "from_bases",
          [](const std::vector<std::pair<
                 std::string, std::vector<std::vector<int32_t>>>> &bases,
             const std::vector<std::string> &outDimNames,
             std::optional<std::vector<int32_t>> outDimSizes,
             bool requireSurjective) {
            auto *ctx = getLinearLayoutContext();

            std::vector<
                std::pair<mlir::StringAttr, std::vector<std::vector<int32_t>>>>
                convertedBases;
            convertedBases.reserve(bases.size());
            for (const auto &entry : bases) {
              std::vector<std::vector<int32_t>> converted;
              converted.reserve(entry.second.size());
              for (const auto &vec : entry.second)
                converted.emplace_back(vec.begin(), vec.end());
              convertedBases.emplace_back(
                  mlir::StringAttr::get(ctx, entry.first),
                  std::move(converted));
            }

            if (outDimSizes) {
              if (outDimSizes->size() != outDimNames.size())
                throw std::invalid_argument("out_dim_names and out_dim_sizes "
                                            "must have the same length");
              std::vector<std::pair<mlir::StringAttr, int32_t>> outDims;
              outDims.reserve(outDimNames.size());
              for (auto it : llvm::enumerate(outDimNames))
                outDims.emplace_back(mlir::StringAttr::get(ctx, it.value()),
                                     (*outDimSizes)[it.index()]);
              return LinearLayout(convertedBases, outDims, requireSurjective);
            }

            if (!requireSurjective)
              throw std::invalid_argument("out_dim_sizes must be provided when "
                                          "require_surjective is false");

            std::vector<mlir::StringAttr> convertedNames;
            convertedNames.reserve(outDimNames.size());
            for (const auto &name : outDimNames)
              convertedNames.push_back(mlir::StringAttr::get(ctx, name));
            return LinearLayout(convertedBases, convertedNames);
          },
          py::arg("bases"), py::arg("out_dim_names"),
          py::arg("out_dim_sizes") = py::none(),
          py::arg("require_surjective") = true)
      .def("compose", &LinearLayout::compose)
      .def("invert_and_compose", &LinearLayout::invertAndCompose)
      .def("invert", &LinearLayout::invert)
      .def("pseudoinvert", &LinearLayout::pseudoinvert)
      .def("is_surjective", &LinearLayout::isSurjective)
      .def("is_injective", &LinearLayout::isInjective)
      .def("is_invertible", &LinearLayout::isInvertible)
      .def("get_in_dim_names",
           [](const LinearLayout &self) {
             std::vector<std::string> dims;
             dims.reserve(self.getNumInDims());
             for (mlir::StringAttr dim : self.getInDimNames())
               dims.push_back(dim.str());
             return dims;
           })
      .def("get_out_dim_names",
           [](const LinearLayout &self) {
             std::vector<std::string> dims;
             dims.reserve(self.getNumOutDims());
             for (mlir::StringAttr dim : self.getOutDimNames())
               dims.push_back(dim.str());
             return dims;
           })
      .def_property_readonly(
          "bases",
          [](const LinearLayout &self) {
            auto bases = self.getBases();
            pybind11::list result;
            for (const auto &it : bases) {
              pybind11::list dimBases;
              for (const auto &vec : it.second)
                dimBases.append(pybind11::cast(
                    std::vector<int32_t>(vec.begin(), vec.end())));
              result.append(pybind11::make_tuple(it.first.str(), dimBases));
            }
            return result;
          })
      .def_property_readonly(
          "out_dims",
          [](const LinearLayout &self) {
            pybind11::list result;
            for (const auto &it : self.getOutDims()) {
              result.append(pybind11::make_tuple(it.first.str(), it.second));
            }
            return result;
          })
      .def_property_readonly("num_in_dims", &LinearLayout::getNumInDims)
      .def_property_readonly("num_out_dims", &LinearLayout::getNumOutDims)
      .def("__mul__", [](const LinearLayout &lhs,
                         const LinearLayout &rhs) { return lhs * rhs; })
      .def(
          "__imul__",
          [](LinearLayout &lhs, const LinearLayout &rhs) -> LinearLayout & {
            lhs *= rhs;
            return lhs;
          },
          py::return_value_policy::reference_internal)
      .def("__eq__", [](const LinearLayout &lhs,
                        const LinearLayout &rhs) { return lhs == rhs; })
      .def("__ne__", [](const LinearLayout &lhs,
                        const LinearLayout &rhs) { return lhs != rhs; })
      .def("__repr__", [](const LinearLayout &self) { return self.toString(); })
      .def("__str__", [](const LinearLayout &self) { return self.toString(); })
      .def("get_shared_view",
           [](const LinearLayout &self, bool useHWPointOfView) {
             return mlir::triton::gpu::getSharedLayoutStr(
                 const_cast<LinearLayout &>(self), useHWPointOfView);
           })
      .def("get_distributed_view",
           [](const LinearLayout &self, bool useHWPointOfView) {
             return mlir::triton::gpu::getDistributedLayoutStr(
                 const_cast<LinearLayout &>(self), useHWPointOfView);
           })
      .def(
          "apply",
          [](const LinearLayout &self, py::dict inputsDict) {
            std::vector<std::pair<std::string, int32_t>> inputs;
            inputs.reserve(inputsDict.size());
            for (auto item : inputsDict) {
              inputs.emplace_back(py::cast<std::string>(item.first),
                                  py::cast<int32_t>(item.second));
            }
            auto *ctx = getLinearLayoutContext();
            std::vector<std::pair<mlir::StringAttr, int32_t>> converted;
            converted.reserve(inputs.size());
            for (const auto &it : inputs) {
              converted.emplace_back(mlir::StringAttr::get(ctx, it.first),
                                     it.second);
            }
            auto outputs = self.apply(converted);
            py::dict result;
            for (const auto &out : outputs) {
              result[py::str(out.first.str())] = out.second;
            }
            return result;
          },
          py::arg("inputs"))
      .def("get_matrix_view", [](const LinearLayout &self) {
        std::unique_ptr<uint64_t[]> matrix = self.getGF2Matrix();
        auto nRows = self.getTotalOutDimSizeBits();
        auto nCols = self.getTotalInDimSizeLog2();
        std::vector<std::vector<int>> result(nRows, std::vector<int>(nCols));
        for (size_t i = 0; i < nRows; ++i) {
          for (size_t j = 0; j < nCols; ++j) {
            result[i][j] = (matrix[i] >> j) & 1;
          }
        }
        return result;
      });

  m.def(
      "get_vec_bitwidth_ld_st",
      [](const LinearLayout &src, const LinearLayout &dst, int32_t bitwidth) {
        return mlir::triton::gpu::getVecBitwidthLdSt(
            canonicalizeDistributedInputs(src),
            canonicalizeDistributedInputs(dst), bitwidth);
      },
      py::arg("src"), py::arg("dst"), py::arg("bitwidth"));
  m.def("optimal_swizzled_ldst_plan", &optimalSwizzledLdStPlan, py::arg("src"),
        py::arg("dst"), py::arg("bitwidth"), py::arg("num_banks") = 32,
        py::arg("src_lane_contig") = std::vector<int32_t>{},
        py::arg("src_lane_addr") = std::vector<int32_t>{},
        py::arg("dst_lane_contig") = std::vector<int32_t>{},
        py::arg("dst_lane_addr") = std::vector<int32_t>{});
}
