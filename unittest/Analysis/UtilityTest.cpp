#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "unittest/Tools/MixedRadixLayouts.h"
#include "unittest/Util/ScopedAllowNpot.h"

#include "mlir/Parser/Parser.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

#include <deque>
#include <set>
#include <type_traits>

#include <deque>
#include <set>

namespace mlir {

TEST(Analysis, ReduceCommutativity) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  struct TestCase {
    StringRef name;
    StringRef body;
    bool commutative;
  };
  TestCase cases[] = {
      {"independent sums", R"mlir(
        %r0 = arith.addf %a0, %b0 : f32
        %r1 = arith.addf %b1, %a1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       true},
      {"complex multiplication", R"mlir(
        %rr = arith.mulf %a0, %b0 : f32
        %ii = arith.mulf %a1, %b1 : f32
        %ri = arith.mulf %a0, %b1 : f32
        %ir = arith.mulf %a1, %b0 : f32
        %real = arith.subf %rr, %ii : f32
        %imag = arith.addf %ri, %ir : f32
        tt.reduce.return %real, %imag : f32, f32
      )mlir",
       true},
      {"shared symmetric subexpressions", R"mlir(
        %s0 = arith.addf %a0, %b0 : f32
        %s1 = arith.addf %a1, %b1 : f32
        %product = arith.mulf %s0, %s1 : f32
        %difference = arith.subf %s0, %s1 : f32
        tt.reduce.return %product, %difference : f32, f32
      )mlir",
       true},
      {"duplicate constants and opaque capture", R"mlir(
        %one0 = arith.constant 1.0 : f32
        %one1 = arith.constant 1.0 : f32
        %scaled0 = arith.mulf %a0, %c : f32
        %scaled1 = arith.mulf %b0, %c : f32
        %biased0 = arith.addf %scaled0, %one0 : f32
        %biased1 = arith.addf %scaled1, %one1 : f32
        %r0 = arith.addf %biased0, %biased1 : f32
        tt.reduce.return %r0, %c : f32, f32
      )mlir",
       true},
      {"output order matters", R"mlir(
        %r0 = arith.addf %a0, %b1 : f32
        %r1 = arith.addf %a1, %b0 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"noncommutative root", R"mlir(
        %square0 = arith.mulf %a0, %a0 : f32
        %square1 = arith.mulf %b0, %b0 : f32
        %r0 = arith.subf %square0, %square1 : f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"distinct constants", R"mlir(
        %one = arith.constant 1.0 : f32
        %two = arith.constant 2.0 : f32
        %biased0 = arith.addf %a0, %one : f32
        %biased1 = arith.addf %b0, %two : f32
        %r0 = arith.addf %biased0, %biased1 : f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"distinct captures", R"mlir(
        %scaled0 = arith.mulf %a0, %c : f32
        %scaled1 = arith.mulf %b0, %d : f32
        %r0 = arith.addf %scaled0, %scaled1 : f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"comparison properties matter", R"mlir(
        %p0 = arith.cmpf olt, %a0, %c : f32
        %p1 = arith.cmpf ogt, %b0, %c : f32
        %p = arith.xori %p0, %p1 : i1
        %r0 = arith.select %p, %c, %d : f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"result indices matter", R"mlir(
        %ai = arith.bitcast %a0 : f32 to i32
        %bi = arith.bitcast %b0 : f32 to i32
        %three = arith.constant 3 : i32
        %lo0, %hi0 = arith.mulsi_extended %ai, %three : i32
        %lo1, %hi1 = arith.mulsi_extended %bi, %three : i32
        %bits = arith.xori %lo0, %hi1 : i32
        %r0 = arith.bitcast %bits : i32 to f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"do not reassociate floating point", R"mlir(
        %s0 = arith.addf %a0, %a1 : f32
        %s1 = arith.addf %s0, %b0 : f32
        %r0 = arith.addf %s1, %b1 : f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"side effects are unsupported", R"mlir(
        tt.store %ptr, %a0 : !tt.ptr<f32>
        %r0 = arith.addf %a0, %b0 : f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
      {"nested regions are unsupported", R"mlir(
        %r0 = "tt.reduce"(%x) <{axis = 0 : i32}> ({
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          tt.reduce.return %sum : f32
        }) : (tensor<2xf32>) -> f32
        %r1 = arith.addf %a1, %b1 : f32
        tt.reduce.return %r0, %r1 : f32, f32
      )mlir",
       false},
  };
  for (const TestCase &test : cases) {
    SCOPED_TRACE(test.name.str());
    std::string source = R"mlir(
      module {
        tt.func @test(%x: tensor<2xf32>, %y: tensor<2xf32>,
                      %c: f32, %d: f32, %ptr: !tt.ptr<f32>) {
          %r:2 = "tt.reduce"(%x, %y) <{axis = 0 : i32}> ({
          ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
    )mlir";
    source += test.body.str();
    source += R"mlir(
          }) : (tensor<2xf32>, tensor<2xf32>) -> (f32, f32)
          tt.return
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    auto function = cast<triton::FuncOp>(module->getBody()->front());
    auto reduce = cast<triton::ReduceOp>(function.getBody().front().front());
    EXPECT_EQ(reduce.isCommutative(), test.commutative);
  }
}

TEST(Analysis, reorder) {
  SmallVector<int> shape({10, 20, 30});
  {
    SmallVector<unsigned> order({2, 1, 0});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 30);
    EXPECT_EQ(reordered[1], 20);
    EXPECT_EQ(reordered[2], 10);
  }
  {
    SmallVector<unsigned> order({1, 0, 2});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 20);
    EXPECT_EQ(reordered[1], 10);
    EXPECT_EQ(reordered[2], 30);
  }
}

TEST(Analysis, SharedMemoryEffectsPreserveResourceAndKind) {
  MLIRContext context;
  context.getOrLoadDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #shared = #ttg.swizzled_shared<{
      vec = 1, perPhase = 1, maxPhase = 1, order = [0]
    }>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
      tt.func @access(%payload: !ttg.memdesc<2xi64, #shared, #smem>,
                      %barrier: !ttg.memdesc<1xi64, #shared, #smem>) {
        ttng.inval_barrier %barrier : !ttg.memdesc<1xi64, #shared, #smem>
        ttng.clc_try_cancel %payload, %barrier :
            !ttg.memdesc<2xi64, #shared, #smem>,
            !ttg.memdesc<1xi64, #shared, #smem>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);

  SmallVector<triton::gpu::SharedKind> kinds;
  module->walk([&](MemoryEffectOpInterface op) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    op.getEffects(effects);
    for (const auto &effect : effects) {
      auto shared = dyn_cast<triton::gpu::SharedMemoryEffect>(&effect);
      ASSERT_TRUE(shared);
      EXPECT_EQ(effect.getResource(), triton::gpu::SharedMemory::get());
      kinds.push_back(shared.getKind());
    }
  });
  using Kind = triton::gpu::SharedKind;
  EXPECT_EQ(kinds,
            (SmallVector<Kind>{Kind::Generic, Kind::Async, Kind::Barrier}));

  auto function = cast<triton::FuncOp>(module->getBody()->front());
  auto invalidate = cast<triton::nvidia_gpu::InvalBarrierOp>(
      function.getBody().front().front());
  auto cancel =
      cast<triton::nvidia_gpu::CLCTryCancelOp>(invalidate->getNextNode());
  EXPECT_TRUE(
      triton::hasSharedAccess(invalidate, Kind::Generic, triton::RW::Write));
  EXPECT_FALSE(
      triton::hasSharedAccess(invalidate, Kind::Generic, triton::RW::Read));
  EXPECT_FALSE(triton::hasSharedAccess(invalidate, Kind::Async));
  EXPECT_TRUE(triton::hasSharedAccess(cancel));
  EXPECT_EQ(
      triton::getMemoryAccesses(cancel, Kind::Async, triton::RW::Write).size(),
      1);
  EXPECT_EQ(triton::getMemoryAccesses(cancel, Kind::Barrier, triton::RW::Write)
                .size(),
            1);
  EXPECT_TRUE(triton::getMemoryAccesses(cancel, std::nullopt, triton::RW::Read)
                  .empty());

  MemoryEffects::EffectInstance allocation(MemoryEffects::Allocate::get(),
                                           triton::gpu::SharedMemory::get());
  MemoryEffects::EffectInstance unrelated(MemoryEffects::Read::get());
  EXPECT_FALSE(dyn_cast<triton::gpu::SharedMemoryEffect>(&allocation));
  EXPECT_FALSE(dyn_cast<triton::gpu::SharedMemoryEffect>(&unrelated));
}

TEST(Analysis, AddressSetExhaustiveEightUnitUniverse) {
  constexpr unsigned universe = 8;
  auto fromMask = [](unsigned mask) {
    triton::AddressSet result;
    for (unsigned bit = 0; bit < universe; ++bit)
      if (mask & (1u << bit))
        result.set(bit);
    return result;
  };

  for (unsigned lhsMask = 0; lhsMask < (1u << universe); ++lhsMask) {
    triton::AddressSet lhs = fromMask(lhsMask);

    for (unsigned rhsMask = 0; rhsMask < (1u << universe); ++rhsMask) {
      triton::AddressSet rhs = fromMask(rhsMask);
      EXPECT_EQ(lhs.intersects(rhs), (lhsMask & rhsMask) != 0);
      EXPECT_EQ(lhs.contains(rhs), (rhsMask & ~lhsMask) == 0);
      EXPECT_EQ(lhs.intersection(rhs), fromMask(lhsMask & rhsMask));

      triton::AddressSet difference = lhs;
      difference.subtract(rhs);
      EXPECT_EQ(difference, fromMask(lhsMask & ~rhsMask));
    }
  }
}

TEST(Analysis, BufferRegionViewPreservesSubviewProvenance) {
  triton::BufferRegion region{
      /*baseOffset=*/16,
      /*length=*/8,
      {{0, triton::AddressSet::fromRange(/*begin=*/16, /*length=*/8)}}};
  triton::BufferRegionView fromSubview{region, /*storageBase=*/0,
                                       /*affineOffset=*/16};
  triton::BufferRegionView fromAllocation{region, /*storageBase=*/16,
                                          /*affineOffset=*/0};

  EXPECT_EQ(fromSubview.region, fromAllocation.region);
  EXPECT_FALSE(fromSubview == fromAllocation);
  triton::RegionInfo joined = triton::RegionInfo::join(
      triton::RegionInfo({fromSubview}), triton::RegionInfo({fromAllocation}));
  EXPECT_EQ(joined.views.size(), 2);

  std::set<triton::BufferRegion> physicalRegions;
  for (const triton::BufferRegionView &view : joined.views)
    physicalRegions.insert(view.region);
  EXPECT_EQ(physicalRegions.size(), 1);
}

TEST(Analysis, RegionInfoAssignmentPreservesImmutableViews) {
  using triton::BufferRegionView;
  using triton::RegionInfo;
  RegionInfo source({BufferRegionView{{0, 8}}});
  RegionInfo target({BufferRegionView{{16, 8}}});
  target = source;
  EXPECT_EQ(target, source);
  const RegionInfo &self = target;
  target = self;
  EXPECT_EQ(target, source);
  target = RegionInfo::getPessimisticValueState();
  EXPECT_TRUE(target.isUnknown());
  EXPECT_TRUE(target.views.empty());
  RegionInfo moved(source);
  target = std::move(moved);
  EXPECT_EQ(target, source);
}

TEST(Analysis, RegionInfoPreservesDistinctFootprints) {
  using triton::AddressSet;
  using triton::BufferRegionView;
  using triton::RegionInfo;
  struct ConstantHash {
    size_t operator()(const triton::BufferRegionView &) const { return 0; }
  };
  std::unordered_set<triton::BufferRegionView, ConstantHash> collided;
  std::set<BufferRegionView> expected;
  RegionInfo::ViewList forward, reverse;
  SmallVector<BufferRegionView> values;
  for (unsigned mask = 0; mask < 256; ++mask) {
    AddressSet addresses;
    for (unsigned bit = 0; bit < 8; ++bit)
      if (mask & (1u << bit))
        addresses.set(bit);
    BufferRegionView view{{0, 8, {{0, addresses}}}};
    values.push_back(view);
    expected.insert(view);
    forward.insert(view);
    collided.insert(view);
    collided.insert(view);
  }
  for (const auto &view : llvm::reverse(values))
    reverse.insert(view);
  EXPECT_EQ(forward.size(), expected.size());
  EXPECT_EQ(collided.size(), expected.size());
  for (const auto &view : expected) {
    EXPECT_EQ(forward.count(view), 1);
    EXPECT_EQ(collided.count(view), 1);
  }
  RegionInfo lhs(forward), rhs(reverse);
  EXPECT_EQ(lhs, rhs);
  EXPECT_EQ(RegionInfo::join(lhs, rhs), lhs);
  EXPECT_EQ(RegionInfo::join(rhs, lhs), lhs);
}

TEST(Analysis, BufferRegionFootprintUnknownIsNotEmpty) {
  MLIRContext context;
  context.getOrLoadDialect<triton::gpu::TritonGPUDialect>();
  Attribute space = triton::gpu::SharedMemorySpaceAttr::get(&context);
  triton::BufferRegionView view{
      {0, 8, {{0, triton::AddressSet::fromRange(0, 8)}}}, 0, 0, {}, 0, 0, 1};
  triton::BufferRegionFootprint known{space, triton::RegionInfo({view})};
  triton::BufferRegionFootprint uninitialized{space, {}};
  triton::BufferRegionFootprint unknown{
      space, triton::RegionInfo::getPessimisticValueState()};
  triton::BufferRegionFootprint noCandidates{
      space, triton::RegionInfo(triton::RegionInfo::ViewList{})};

  const triton::BufferRegionFootprint *incomplete[] = {nullptr, &uninitialized,
                                                       &unknown, &noCandidates};
  for (const auto *footprint : incomplete) {
    EXPECT_TRUE(triton::mayOverlap(footprint, &known));
    EXPECT_TRUE(triton::mayOverlap(&known, footprint));
  }

  triton::BufferRegionView emptyView{{}, 0, 0, {}, 0, 0, 1};
  triton::BufferRegionFootprint empty{space, triton::RegionInfo({emptyView})};
  EXPECT_FALSE(triton::mayOverlap(&empty, &known));
  EXPECT_FALSE(triton::mayOverlap(&known, &empty));
  EXPECT_TRUE(triton::mayOverlap(&empty, &unknown));
}

namespace {

uint64_t toBits(const llvm::SmallBitVector &mask) {
  assert(mask.size() <= 64);
  uint64_t result = 0;
  for (unsigned bit = 0; bit < mask.size(); ++bit)
    if (mask.test(bit))
      result |= uint64_t{1} << bit;
  return result;
}

void expectFullCoveragePartition(ArrayRef<triton::BufferRegion> regions) {
  ASSERT_EQ(regions.size(), 64u);
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  ASSERT_EQ(plan.numLanes, 64u);
  ASSERT_EQ(plan.regionMasks.size(), regions.size());
  EXPECT_EQ(toBits(plan.regionMasks.front()), ~uint64_t{0});
  for (unsigned tile = 0; tile < 63; ++tile)
    EXPECT_EQ(toBits(plan.regionMasks[tile + 1]), uint64_t{1} << tile)
        << "tile " << tile;
}

triton::BufferRegion makeRegion(unsigned id, unsigned addressMask,
                                unsigned universe) {
  triton::AddressSet addresses;
  for (unsigned bit = 0; bit < universe; ++bit)
    if (addressMask & (1u << bit))
      addresses.set(bit);
  triton::BufferRegion region{/*baseOffset=*/id * 16, /*length=*/universe};
  if (!addresses.empty())
    region.ctaAddresses.emplace_back(0, std::move(addresses));
  return region;
}

bool planNeverMissesHazard(ArrayRef<unsigned> addressMasks, unsigned universe) {
  SmallVector<triton::BufferRegion> regions;
  for (auto [id, mask] : llvm::enumerate(addressMasks))
    regions.push_back(makeRegion(id, mask, universe));
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  if (plan.numLanes > 64)
    return false;

  using State = std::pair<unsigned, uint64_t>;
  std::deque<State> worklist;
  std::set<State> seen;
  worklist.push_back({0, 0});
  seen.insert({0, 0});
  while (!worklist.empty()) {
    auto [exactState, planState] = worklist.front();
    worklist.pop_front();
    for (unsigned region = 0; region < regions.size(); ++region) {
      uint64_t check = toBits(plan.regionMasks[region]);
      bool exactHazard = (exactState & addressMasks[region]) != 0;
      bool planHazard = (planState & check) != 0;
      if (exactHazard && !planHazard)
        return false;

      State updated = {exactState | addressMasks[region],
                       planState | toBits(plan.regionMasks[region])};
      if (seen.insert(updated).second)
        worklist.push_back(updated);

      State completed = {exactState & ~addressMasks[region],
                         planState & ~toBits(plan.regionMasks[region])};
      if (seen.insert(completed).second)
        worklist.push_back(completed);
    }
  }
  return true;
}

bool planPublishesEveryAliasingAtom(ArrayRef<unsigned> addressMasks,
                                    unsigned universe) {
  SmallVector<triton::BufferRegion> regions;
  for (auto [id, mask] : llvm::enumerate(addressMasks))
    regions.push_back(makeRegion(id, mask, universe));
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  if (plan.numLanes > 64)
    return false;

  for (unsigned generic = 0; generic < regions.size(); ++generic) {
    for (unsigned async = 0; async < regions.size(); ++async) {
      bool exactOverlap = (addressMasks[generic] & addressMasks[async]) != 0;
      bool published = (toBits(plan.regionMasks[generic]) &
                        toBits(plan.regionMasks[async])) != 0;
      if (exactOverlap != published)
        return false;
    }
  }
  return true;
}

} // namespace

TEST(Analysis, BufferStatePlanExhaustiveThreeViewsFourAddresses) {
  constexpr unsigned universe = 4;
  constexpr unsigned setCount = 1u << universe;
  for (unsigned a = 0; a < setCount; ++a)
    for (unsigned b = 0; b < setCount; ++b)
      for (unsigned c = 0; c < setCount; ++c) {
        ASSERT_TRUE(planNeverMissesHazard({a, b, c}, universe))
            << "address masks: " << a << ", " << b << ", " << c;
        ASSERT_TRUE(planPublishesEveryAliasingAtom({a, b, c}, universe))
            << "proxy publication masks: " << a << ", " << b << ", " << c;
      }
}

TEST(Analysis, BufferStatePlanUsesAtomsForSparsePartition) {
  constexpr unsigned universe = 8;
  SmallVector<unsigned> addressMasks = {
      0xff, // full
      0x55, // even
      0xaa, // odd
  };
  SmallVector<triton::BufferRegion> regions;
  for (auto [id, mask] : llvm::enumerate(addressMasks))
    regions.push_back(makeRegion(id, mask, universe));

  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  EXPECT_EQ(plan.numLanes, 2);

  for (unsigned exactState = 0; exactState < (1u << universe); ++exactState) {
    uint64_t planState = 0;
    if (exactState & 0x55)
      planState |= toBits(plan.regionMasks[1]);
    if (exactState & 0xaa)
      planState |= toBits(plan.regionMasks[2]);
    for (unsigned region = 0; region < regions.size(); ++region) {
      EXPECT_EQ((exactState & addressMasks[region]) != 0,
                (planState & toBits(plan.regionMasks[region])) != 0);
    }
  }
}

TEST(Analysis, BufferStatePlanKeepsPartialOverlapExact) {
  SmallVector<triton::BufferRegion> regions = {
      makeRegion(0, 0b0011, 4),
      makeRegion(1, 0b0110, 4),
  };
  triton::BufferStatePlan plan = triton::createBufferStatePlan(regions);
  EXPECT_EQ(plan.numLanes, 3);
  EXPECT_EQ(toBits(plan.regionMasks[0]), 0b011);
  EXPECT_EQ(toBits(plan.regionMasks[1]), 0b110);
  EXPECT_TRUE(planNeverMissesHazard({0b0011, 0b0110}, 4));
  EXPECT_TRUE(planPublishesEveryAliasingAtom({0b0011, 0b0110}, 4));
}

TEST(Analysis, BufferStatePlanKeepsLargePaddedIntervalsExact) {
  constexpr uint32_t tileCount = 64;
  constexpr uint32_t tileLength = 3648;
  SmallVector<triton::BufferRegion> regions;
  regions.reserve(tileCount);
  regions.push_back(
      {0,
       tileCount * tileLength,
       {{0, triton::AddressSet::fromRange(0, tileCount * tileLength)}}});
  for (uint32_t tile = 0; tile < tileCount - 1; ++tile) {
    uint32_t base = tile * tileLength;
    regions.push_back({base,
                       tileLength,
                       {{0, triton::AddressSet::fromRange(base, tileLength)}}});
  }

  expectFullCoveragePartition(regions);
}

TEST(Analysis, BufferStatePlanKeepsLargeSparsePartitionsExact) {
  constexpr uint32_t tileCount = 64;
  constexpr uint32_t stripeCount = 4;
  constexpr uint32_t stripeLength = 912;
  constexpr uint32_t tileLength = stripeCount * stripeLength;
  SmallVector<triton::BufferRegion> regions;
  regions.reserve(tileCount);
  regions.push_back(
      {0,
       tileCount * tileLength,
       {{0, triton::AddressSet::fromRange(0, tileCount * tileLength)}}});

  for (uint32_t tile = 0; tile < tileCount - 1; ++tile) {
    triton::AddressSet addresses;
    for (uint32_t stripe = 0; stripe < stripeCount; ++stripe)
      addresses.insert(triton::AddressSet::fromRange(
          (stripe * tileCount + tile) * stripeLength, stripeLength));
    regions.push_back(
        {tile * stripeLength, tileLength, {{0, std::move(addresses)}}});
  }

  expectFullCoveragePartition(regions);
}

namespace {

using Bases = std::vector<std::vector<int32_t>>;
using triton::exactIdentity1D;
using triton::makeRegLaneLayout;

triton::LinearLayout exactRegisters(MLIRContext &ctx, int32_t size,
                                    int32_t blockSize = 1) {
  auto S = [&](StringRef str) { return StringAttr::get(&ctx, str); };
  return exactIdentity1D(size, S("register"), S("dim0")) *
         triton::LinearLayout::zeros1D(1, S("lane"), S("dim0")) *
         triton::LinearLayout::zeros1D(blockSize, S("block"), S("dim0"));
}

void expectUnsupportedReason(const MixedRadixConversionPlan &plan,
                             StringRef message) {
  ASSERT_TRUE(plan.unsupportedReason);
  EXPECT_FALSE(plan.supported());
  EXPECT_FALSE(plan.transfer);
  EXPECT_FALSE(plan.scope);
  EXPECT_FALSE(plan.warpFactors);
  EXPECT_FALSE(plan.destinationRegisterOrder);
  EXPECT_EQ(*plan.unsupportedReason, message.str());
}

void expectShared(const MixedRadixConversionPlan &plan,
                  const triton::LinearLayout &src,
                  triton::MixedRadixConversionScope scope) {
  ASSERT_TRUE(plan.supported());
  EXPECT_EQ(plan.transfer, MixedRadixConversionTransfer::Shared);
  EXPECT_EQ(plan.scope, scope);
  EXPECT_TRUE(
      llvm::equal(plan.sourceFreeVariableMasks, src.getFreeVariableMasks()));
  EXPECT_FALSE(plan.warpFactors);
  EXPECT_FALSE(plan.destinationRegisterOrder);
}

} // namespace

TEST(Analysis, MixedRadixConversionPlanRejectsPreflightFailures) {
  MLIRContext ctx;
  auto exact = exactRegisters(ctx, 6);
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(triton::LinearLayout::empty(),
                                     triton::LinearLayout::empty(), {}),
      "NPOT mixed-radix conversion without register and lane input dimensions "
      "is not yet supported");
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(exactRegisters(ctx, 3), exact, {6}),
      "NPOT mixed-radix conversion with incomplete logical output coverage in "
      "the source layout is not yet supported");
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(exact, exactRegisters(ctx, 3), {6}),
      "NPOT mixed-radix conversion with incomplete logical output coverage in "
      "the destination layout is not yet supported");
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(exactRegisters(ctx, 6, 2), exact, {6}),
      "NPOT mixed-radix conversion with a non-singleton block input in the "
      "source layout is not yet supported");
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(exact, exactRegisters(ctx, 6, 2), {6}),
      "NPOT mixed-radix conversion with a non-singleton block input in the "
      "destination layout is not yet supported");
  // A zero radix-3 digit on power-of-two outputs has no free-variable masks.
  Bases lanes = {{0, 1}, {0, 2}, {0, 4}};
  auto mixedRadix =
      makeRegLaneLayout(ctx, {{1, 0}, {2, 0}, {0, 0}}, {2, 2, 3}, lanes, {}, 4);
  auto pow2 = makeRegLaneLayout(ctx, {{1, 0}, {2, 0}}, {2, 2}, lanes, {}, 4);
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(mixedRadix, pow2, {4, 8}),
      "NPOT mixed-radix conversion with non-radix-2 digits on power-of-two "
      "outputs in the source layout is not yet supported");
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(pow2, mixedRadix, {4, 8}),
      "NPOT mixed-radix conversion with non-radix-2 digits on power-of-two "
      "outputs in the destination layout is not yet supported");
  auto misnamed = exactIdentity1D(6, StringAttr::get(&ctx, "register"),
                                  StringAttr::get(&ctx, "dim1"));
  expectUnsupportedReason(
      planMixedRadixLayoutConversion(misnamed, misnamed, {6}),
      "NPOT mixed-radix conversion with mismatched or non-standard "
      "out dims is not yet supported");
}

TEST(Analysis, MixedRadixConversionPlanFactorsRegisterAndLanePermutations) {
  MLIRContext ctx;
  auto reg = StringAttr::get(&ctx, "register");
  auto lane = StringAttr::get(&ctx, "lane");
  // `order` is destinationRegisterOrder applied to values 0..n-1, which are
  // indexed by pReg's output.
  auto expectFactored = [&](const triton::LinearLayout &src,
                            const triton::LinearLayout &dst,
                            MixedRadixConversionTransfer transfer,
                            const Bases &pReg, const Bases &shuffleMap,
                            ArrayRef<intptr_t> order) {
    auto plan = planMixedRadixLayoutConversion(
        src, dst, {src.getOutDimSize(StringAttr::get(&ctx, "dim0")), 8});
    ASSERT_TRUE(plan.supported());
    EXPECT_EQ(plan.transfer, transfer);
    EXPECT_EQ(plan.scope, transfer == MixedRadixConversionTransfer::Register
                              ? triton::MixedRadixConversionScope::Register
                              : triton::MixedRadixConversionScope::Warp);
    EXPECT_TRUE(
        llvm::equal(plan.sourceFreeVariableMasks, src.getFreeVariableMasks()));
    ASSERT_TRUE(plan.warpFactors && plan.destinationRegisterOrder);
    EXPECT_EQ(plan.warpFactors->pReg.getBases().lookup(reg), pReg);
    EXPECT_EQ(plan.warpFactors->shuffleMap.getBases().lookup(lane), shuffleMap);
    SmallVector<Value> values;
    for (intptr_t i = 0; i < static_cast<intptr_t>(order.size()); ++i)
      values.push_back(
          Value::getFromOpaquePointer(reinterpret_cast<void *>(i)));
    SmallVector<intptr_t> ids;
    for (Value v : plan.destinationRegisterOrder->apply(values))
      ids.push_back(reinterpret_cast<intptr_t>(v.getAsOpaquePointer()));
    EXPECT_EQ(ArrayRef<intptr_t>(ids), order);
  };
  Bases lanes = {{0, 1}, {0, 2}, {0, 4}};
  Bases rotated = {{0, 2}, {0, 4}, {0, 1}};
  {
    // dst = 2 * src on the odd coordinate, so src -> dst is 3 mod 5.
    SCOPED_TRACE("odd register swap");
    expectFactored(makeRegLaneLayout(ctx, {{1, 0}}, {5}, lanes, {}, 5),
                   makeRegLaneLayout(ctx, {{2, 0}}, {5}, lanes, {}, 5),
                   MixedRadixConversionTransfer::Register, {{3}},
                   {{1}, {2}, {4}}, {0, 1, 2, 3, 4});
  }
  {
    SCOPED_TRACE("destination odd digit first");
    expectFactored(makeRegLaneLayout(ctx, {{0, 4}, {1, 0}}, {2, 5},
                                     {{0, 1}, {0, 2}}, {}, 5),
                   makeRegLaneLayout(ctx, {{1, 0}, {0, 4}}, {5, 2},
                                     {{0, 1}, {0, 2}}, {}, 5),
                   MixedRadixConversionTransfer::Register, {{1}, {2}},
                   {{1}, {2}}, {0, 2, 4, 6, 8, 1, 3, 5, 7, 9});
  }
  {
    // operator* puts the odd digit between binary ones, as {2, 3, 2} here.
    SCOPED_TRACE("lane permutation with the odd digit in the middle");
    Bases registers = {{1, 0}, {2, 0}, {0, 1}};
    expectFactored(
        makeRegLaneLayout(ctx, registers, {2, 3, 2}, {{0, 2}, {0, 4}}, {}, 6),
        makeRegLaneLayout(ctx, registers, {2, 3, 2}, {{0, 4}, {0, 2}}, {}, 6),
        MixedRadixConversionTransfer::Warp, {{1}, {4}, {2}}, {{2}, {1}},
        {0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11});
  }
  {
    SCOPED_TRACE("pow2 control");
    auto src = makeRegLaneLayout(ctx, {{1, 0}, {2, 0}}, {2, 2}, lanes, {}, 4);
    auto dst = makeRegLaneLayout(ctx, {{1, 0}, {2, 0}}, {2, 2}, rotated, {}, 4);
    expectFactored(src, dst, MixedRadixConversionTransfer::Warp, {{1}, {2}},
                   {{2}, {4}, {1}}, {0, 1, 2, 3});
    auto plan = planMixedRadixLayoutConversion(src, dst, {4, 8});
    EXPECT_TRUE(plan.warpFactors->pReg.isRadix2());
  }
  {
    // Each warp already holds every value it needs.
    SCOPED_TRACE("identical layouts replicated across warps");
    auto layout =
        exactIdentity1D(3, reg, StringAttr::get(&ctx, "dim0")) *
        triton::LinearLayout::identity1D(8, lane,
                                         StringAttr::get(&ctx, "dim1")) *
        triton::LinearLayout::zeros1D(2, StringAttr::get(&ctx, "warp"),
                                      StringAttr::get(&ctx, "dim0"));
    expectFactored(layout, layout, MixedRadixConversionTransfer::Register,
                   {{1}}, {{1}, {2}, {4}}, {0, 1, 2});
  }
}

TEST(Analysis, MixedRadixConversionPlanSharesWithoutRegisterLaneBijection) {
  MLIRContext ctx;
  using Scope = triton::MixedRadixConversionScope;
  {
    SCOPED_TRACE("register bit moves to a lane");
    auto src = makeRegLaneLayout(ctx, {{0, 1}, {1, 0}}, {2, 5},
                                 {{0, 2}, {0, 4}}, {}, 5);
    auto dst = makeRegLaneLayout(ctx, {{0, 2}, {1, 0}}, {2, 5},
                                 {{0, 1}, {0, 4}}, {}, 5);
    expectShared(planMixedRadixLayoutConversion(src, dst, {5, 8}), src,
                 Scope::Warp);
  }
  {
    SCOPED_TRACE("lane permutation is mixed-radix");
    auto src = makeRegLaneLayout(ctx, {{0, 1}, {0, 2}, {0, 4}}, {2, 2, 2},
                                 {{1, 0}}, {3}, 3);
    auto dst = makeRegLaneLayout(ctx, {{0, 1}, {0, 2}, {0, 4}}, {2, 2, 2},
                                 {{2, 0}}, {3}, 3);
    expectShared(planMixedRadixLayoutConversion(src, dst, {3, 8}), src,
                 Scope::Block);
  }
}

TEST(Analysis, MixedRadixConversionQueriesFollowThePlan) {
  triton::ScopedAllowNpot allowNpot;
  MLIRContext ctx;
  ctx.loadDialect<triton::TritonDialect, triton::gpu::TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #r384 = #ttg.linear<{register = [[1], [2], [128]], lane = [[4], [8], [16], [32], [64]], warp = [], block = [], inputRadices = {register = [2, 2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
    #r384t = #ttg.linear<{register = [[2], [128], [1]], lane = [[4], [8], [16], [32], [64]], warp = [], block = [], inputRadices = {register = [2, 3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [384]}>
    #w192 = #ttg.linear<{register = [[1], [64]], lane = [[2], [4], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [2, 3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
    #w192t = #ttg.linear<{register = [[64], [1]], lane = [[4], [2], [8], [16], [32]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [192]}>
    #s96 = #ttg.linear<{register = [[32], [16]], lane = [[1], [2], [4], [8], [0]], warp = [], block = [], inputRadices = {register = [3, 2], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
    #s96t = #ttg.linear<{register = [[32]], lane = [[1], [2], [4], [8], [16]], warp = [], block = [], inputRadices = {register = [3], lane = [2, 2, 2, 2, 2], warp = [], block = []}, outDimSizes = [96]}>
    #blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
    #swapped = #ttg.linear<{register = [[2], [1]], lane = [[4], [8], [16], [32], [64]], warp = [], block = []}>
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @f() {
        %r = tt.make_range {end = 384 : i32, start = 0 : i32} : tensor<384xi32, #r384>
        %0 = ttg.convert_layout %r : tensor<384xi32, #r384> -> tensor<384xi32, #r384t>
        %w = tt.make_range {end = 192 : i32, start = 0 : i32} : tensor<192xi32, #w192>
        %1 = ttg.convert_layout %w : tensor<192xi32, #w192> -> tensor<192xi32, #w192t>
        %s = tt.make_range {end = 96 : i32, start = 0 : i32} : tensor<96xi32, #s96>
        %2 = ttg.convert_layout %s : tensor<96xi32, #s96> -> tensor<96xi32, #s96t>
        %b = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
        %3 = ttg.convert_layout %b : tensor<128xi32, #blocked> -> tensor<128xi32, #swapped>
        tt.return
      }
    }
  )mlir",
                                            &ctx);
  ASSERT_TRUE(module);
  // {has a plan, reorders registers, needs shared memory}: Register, Warp,
  // Shared, and a power-of-two register permutation as the control.
  SmallVector<std::tuple<bool, bool, bool>> expected = {{true, true, false},
                                                        {true, false, false},
                                                        {true, false, true},
                                                        {false, true, false}};
  SmallVector<std::tuple<bool, bool, bool>> actual;
  module->walk([&](triton::gpu::ConvertLayoutOp cvt) {
    auto srcTy = cvt.getSrc().getType();
    auto dstTy = cvt.getType();
    EXPECT_EQ(cvtNeedsSharedMemory(srcTy, dstTy), cvtNeedsSharedMemory(cvt));
    actual.push_back({getMixedRadixConversionPlan(srcTy, dstTy).has_value(),
                      cvtReordersRegisters(srcTy, dstTy),
                      cvtNeedsSharedMemory(cvt)});
  });
  EXPECT_EQ(actual, expected);
}

TEST(Analysis, ReduceHelpersKeepMixedRadices) {
  triton::ScopedAllowNpot allowNpot;
  MLIRContext ctx;
  ctx.loadDialect<triton::gpu::TritonGPUDialect>();
  auto S = [&](StringRef str) { return StringAttr::get(&ctx, str); };
  Bases warps = {{4, 0}, {8, 0}};
  auto layout = [&](Bases regs, std::vector<int32_t> radices, Bases lanes,
                    int32_t dim0, int32_t dim1) {
    triton::LinearLayout::BasesT bases;
    triton::LinearLayout::RadicesT allRadices;
    bases[S("register")] = regs;
    allRadices[S("register")] = radices;
    bases[S("lane")] = lanes;
    allRadices[S("lane")].assign(lanes.size(), 2);
    bases[S("warp")] = warps;
    allRadices[S("warp")].assign(warps.size(), 2);
    bases[S("block")] = {};
    allRadices[S("block")] = {};
    return triton::LinearLayout(std::move(bases), std::move(allRadices),
                                {{S("dim0"), dim0}, {S("dim1"), dim1}},
                                /*requireSurjective=*/false);
  };
  Bases lanes = {{0, 4}, {0, 8}, {0, 16}, {1, 0}, {2, 0}};
  // [32,96] has a radix-3 register digit on dim1; [32,64] is the control.
  for (int32_t odd : {3, 2}) {
    SCOPED_TRACE(odd);
    int32_t dim1 = 32 * odd;
    auto src = layout({{0, 1}, {0, 2}, {16, 0}, {0, 32}}, {2, 2, 2, odd}, lanes,
                      32, dim1);
    auto srcTy =
        RankedTensorType::get({32, dim1}, Float32Type::get(&ctx),
                              triton::gpu::LinearEncodingAttr::get(&ctx, src));
    EXPECT_EQ(ReduceOpHelper::moveAxisBasesToFront(src, 1).apply(src),
              layout({{0, 1}, {0, 2}, {0, 32}, {16, 0}}, {2, 2, odd, 2}, lanes,
                     32, dim1));
    EXPECT_EQ(ReduceOpHelper::reducedRegLaneLayout(srcTy, 1),
              layout({{16, 0}}, {2}, {{0, 0}, {0, 0}, {0, 0}, {1, 0}, {2, 0}},
                     32, 1));
    EXPECT_EQ(ReduceOpHelper::reducedRegLaneLayout(srcTy, 0),
              layout({{0, 1}, {0, 2}, {0, 32}}, {2, 2, odd},
                     {{0, 4}, {0, 8}, {0, 16}, {0, 0}, {0, 0}}, 16, dim1));
    // Swapping dim0's warps into the zero lanes (reduced) or the first lanes
    // (src) keeps dim1's radix.
    for (const auto &ll : {ReduceOpHelper::reducedRegLaneLayout(srcTy, 0), src})
      EXPECT_EQ(
          ReduceOpHelper::getInterWarpReductionLayout(ll, 0).getInputRadices(
              S("register")),
          ll.getInputRadices(S("register")));
  }
}

TEST(Analysis, BufferRegionViewHashTracksTheImmutableValue) {
  using triton::AddressSet;
  using triton::BufferRegionView;
  using Hash = BufferRegionView::Hash;
  static_assert(std::is_const_v<decltype(BufferRegionView::region)>);
  static_assert(std::is_const_v<decltype(BufferRegionView::partitionBases)>);
  static_assert(std::is_const_v<decltype(BufferRegionView::allocation)>);
  static_assert(!std::is_copy_assignable_v<BufferRegionView>);
  triton::BufferRegion region{0, 72, {{0, AddressSet::fromRange(0, 8)}}};
  region.ctaAddresses.front().second.insert(AddressSet::fromRange(64, 8));
  region.ctaAddresses.front().second.set(10);
  BufferRegionView original{region, 5, 7, {5, 100}, 1, 2, 9};
  auto unhashedCopy = original;
  const size_t originalHash = Hash{}(original);
  EXPECT_EQ(unhashedCopy, original);
  EXPECT_EQ(Hash{}(unhashedCopy), originalHash);
  region.ctaAddresses.front().second.subtract(AddressSet::fromRange(10, 1));
  region.ctaAddresses.front().second.set(11);
  BufferRegionView changed{region, 5, 7, {5, 100}, 1, 2, 9};
  EXPECT_NE(Hash{}(changed), originalHash);
  EXPECT_FALSE(original == changed);
  EXPECT_EQ(Hash{}(original), originalHash);
  auto copy = original;
  EXPECT_EQ(copy, original);
  BufferRegionView moved(std::move(copy));
  EXPECT_EQ(moved, original);
  EXPECT_EQ(Hash{}(moved), originalHash);
  BufferRegionView translated(original.translated(16, 3));
  AddressSet translatedAddresses = AddressSet::fromRange(16, 8);
  translatedAddresses.insert(AddressSet::fromRange(80, 8));
  translatedAddresses.set(26);
  BufferRegionView expected(
      BufferRegionView{{16, 72, {{0, translatedAddresses}}},
                       21,
                       7,
                       {21, 116},
                       1,
                       2,
                       /*allocationFrame=*/3});
  EXPECT_EQ(translated, expected);
  EXPECT_EQ(Hash{}(translated), Hash{}(expected));
  EXPECT_FALSE(translated == original);
  triton::RegionInfo::ViewList views;
  views.insert(std::move(moved));
  EXPECT_EQ(*views.begin(), original);
  EXPECT_EQ(Hash{}(*views.begin()), originalHash);
}

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
