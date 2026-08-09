// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "Z3JointSolutionValidator.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::triton::gpu {
namespace {

constexpr llvm::StringLiteral kProblemSchema = "joint-solver-0.2";
constexpr llvm::StringLiteral kPlanSchema = "lowering-plan-0.1";
constexpr llvm::StringLiteral kValidationSchema = "joint-solver-validation-0.1";

enum class Relation { Always, SameWG, DifferentWG };
enum class Placement { Before, After };

struct ValidationResult {
  bool isValid() const { return violations.empty(); }
  std::vector<std::string> violations;
};

struct Node {
  int64_t id;
  int64_t fixedCycle;
  int64_t duration;
  std::string pipeline;
  std::optional<size_t> cluster;
};

struct Cluster {
  int64_t id;
  int64_t minWarps;
  std::vector<size_t> nodes;
};

struct Edge {
  size_t src;
  size_t dst;
  int64_t result;
  int64_t latency;
  int64_t distance;
  int64_t roundTrip;
  int64_t channelBytes;
  std::optional<size_t> srcCluster;
  std::optional<size_t> dstCluster;
};

struct Consumer {
  size_t node;
  int64_t latency;
  int64_t distance;
};

struct Buffer {
  int64_t id;
  size_t producer;
  int64_t sizeBytes;
  int64_t minCount;
  std::string kind;
  std::vector<Consumer> consumers;
};

struct LoweringEvent {
  int64_t id;
  size_t anchor;
  Placement placement;
  std::string pipeline;
  int64_t duration;
  size_t ownerCluster;
};

struct LoweringTemplate {
  int64_t id;
  Relation relation;
  size_t srcCluster;
  size_t dstCluster;
  std::vector<LoweringEvent> events;
};

struct Problem {
  bool jointMode;
  int64_t ii;
  int64_t maxWGs;
  int64_t committedSmem;
  int64_t fixedSmem;
  int64_t smemBudget;
  int64_t tmemBudget;
  int64_t defaultWGFootprint;
  std::optional<int64_t> regBudget;
  std::optional<size_t> canonicalRoot;
  std::vector<int64_t> warpFootprint;
  std::vector<Node> nodes;
  std::vector<Cluster> clusters;
  std::vector<Edge> edges;
  std::vector<Buffer> buffers;
  std::vector<LoweringTemplate> loweringTemplates;
  std::vector<std::pair<size_t, size_t>> warpGroupConflicts;
};

struct EventPlan {
  int64_t cycle;
  int64_t warpGroup;
  int64_t streamOrder;
};

struct TemplatePlan {
  bool present = false;
  bool active = false;
  std::map<int64_t, EventPlan> events;
};

struct Solution {
  std::vector<int64_t> cycles;
  std::vector<int64_t> warpGroups;
  int64_t usedWGs = 0;
  std::map<int64_t, int64_t> bufferDepths;
  std::vector<TemplatePlan> loweringPlans;
};

struct ExpectedEvent {
  size_t templateIndex = 0;
  size_t eventIndex = 0;
  int64_t cycle = 0;
  int64_t warpGroup = 0;
  int64_t streamOrder = 0;
  bool valid = true;
};

struct Issue {
  int64_t cycle;
  int64_t duration;
  std::string pipeline;
  std::optional<size_t> cluster;
  std::string label;
};

static bool getInteger(const llvm::json::Object &object, llvm::StringRef key,
                       int64_t &value) {
  auto parsed = object.getInteger(key);
  if (!parsed)
    return false;
  value = *parsed;
  return true;
}

static bool getString(const llvm::json::Object &object, llvm::StringRef key,
                      std::string &value) {
  auto parsed = object.getString(key);
  if (!parsed)
    return false;
  value = parsed->str();
  return true;
}

static bool getBoolean(const llvm::json::Object &object, llvm::StringRef key,
                       bool &value) {
  auto parsed = object.getBoolean(key);
  if (!parsed)
    return false;
  value = *parsed;
  return true;
}

static bool fail(std::string &error, std::string message) {
  error = std::move(message);
  return false;
}

static bool parseProblem(llvm::StringRef json, Problem &problem,
                         std::string &error) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return fail(error, "problem JSON parse failed: " +
                           llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root)
    return fail(error, "problem must be a JSON object");
  auto version = root->getString("version");
  auto mode = root->getString("mode");
  if (!version || *version != kProblemSchema || !mode ||
      (*mode != "joint" && *mode != "partition"))
    return fail(error, "validator requires a joint-solver-0.2 problem");
  problem.jointMode = *mode == "joint";

  int64_t emitterCaps = 0;
  int64_t smRegs = 0;
  int64_t defaultSlack = 0;
  if (!getInteger(*root, "emitter_caps_version", emitterCaps) ||
      emitterCaps != 2 || !getInteger(*root, "ii", problem.ii) ||
      problem.ii <= 0 || !getInteger(*root, "max_wgs", problem.maxWGs) ||
      problem.maxWGs <= 0 ||
      !getInteger(*root, "committed_smem", problem.committedSmem) ||
      !getInteger(*root, "fixed_smem", problem.fixedSmem) ||
      !getInteger(*root, "smem_budget", problem.smemBudget) ||
      !getInteger(*root, "tmem_budget_bytes", problem.tmemBudget) ||
      !getInteger(*root, "default_wg_footprint", problem.defaultWGFootprint) ||
      !getInteger(*root, "sm_regs", smRegs) ||
      !getInteger(*root, "default_slack", defaultSlack) ||
      problem.committedSmem < 0 || problem.fixedSmem < 0 ||
      problem.smemBudget < 0 || problem.tmemBudget < 0 ||
      problem.defaultWGFootprint < 0 || smRegs < 0 || defaultSlack < 0)
    return fail(error, "invalid joint problem budgets or limits");

  if (root->get("time_limit_s")) {
    auto value = root->getNumber("time_limit_s");
    if (!value || !std::isfinite(*value) || *value <= 0.0)
      return fail(error, "invalid time_limit_s");
  }

  if (root->get("reg_budget")) {
    int64_t value = 0;
    if (!getInteger(*root, "reg_budget", value) || value < 0)
      return fail(error, "invalid reg_budget");
    if (value > 0)
      problem.regBudget = value;
  }

  auto *nodeValues = root->getArray("nodes");
  auto *clusterValues = root->getArray("clusters");
  auto *edgeValues = root->getArray("edges");
  auto *bufferValues = root->getArray("buffers");
  auto *templateValues = root->getArray("lowering_templates");
  auto *footprintValues = root->getArray("warp_footprint");
  if (!nodeValues || !clusterValues || clusterValues->empty() || !edgeValues ||
      !bufferValues || !templateValues || !footprintValues)
    return fail(error, "joint problem is missing a required array");

  problem.warpFootprint.reserve(footprintValues->size());
  for (const llvm::json::Value &value : *footprintValues) {
    auto footprint = value.getAsInteger();
    if (!footprint || *footprint < 0)
      return fail(error, "warp_footprint entries must be nonnegative integers");
    problem.warpFootprint.push_back(*footprint);
  }
  if (problem.warpFootprint.size() <= 8)
    return fail(error, "warp_footprint must define indices through eight");

  std::map<int64_t, size_t> nodeIndices;
  problem.nodes.reserve(nodeValues->size());
  for (const llvm::json::Value &value : *nodeValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t cycle = 0;
    int64_t duration = 0;
    int64_t latency = 0;
    int64_t frequency = 0;
    std::string pipeline;
    if (!object || !getInteger(*object, "id", id) ||
        !getInteger(*object, "cycle", cycle) ||
        !getInteger(*object, "duration", duration) ||
        !getInteger(*object, "latency", latency) ||
        !getInteger(*object, "freq", frequency) ||
        !getString(*object, "pipeline", pipeline) || cycle < 0 ||
        duration < 0 || latency < 0 || frequency <= 0 || nodeIndices.count(id))
      return fail(error, "invalid node in joint problem");
    nodeIndices.emplace(id, problem.nodes.size());
    problem.nodes.push_back(
        Node{id, cycle, duration, std::move(pipeline), std::nullopt});
  }

  std::map<int64_t, size_t> clusterIndices;
  problem.clusters.reserve(clusterValues->size());
  for (const llvm::json::Value &value : *clusterValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t minWarps = 0;
    auto *nodes = object ? object->getArray("nodes") : nullptr;
    if (!object || !getInteger(*object, "id", id) ||
        !getInteger(*object, "min_warps", minWarps) || !nodes ||
        nodes->empty() || minWarps <= 0 || minWarps > 8 ||
        clusterIndices.count(id))
      return fail(error, "invalid cluster in joint problem");
    size_t clusterIndex = problem.clusters.size();
    clusterIndices.emplace(id, clusterIndex);
    Cluster cluster{id, minWarps, {}};
    cluster.nodes.reserve(nodes->size());
    for (const llvm::json::Value &nodeValue : *nodes) {
      auto nodeId = nodeValue.getAsInteger();
      if (!nodeId)
        return fail(error, "cluster node IDs must be integers");
      auto node = nodeIndices.find(*nodeId);
      if (node == nodeIndices.end() || problem.nodes[node->second].cluster)
        return fail(error, "cluster contains an unknown or repeated node");
      problem.nodes[node->second].cluster = clusterIndex;
      cluster.nodes.push_back(node->second);
    }
    problem.clusters.push_back(std::move(cluster));
  }
  problem.maxWGs = std::min<int64_t>(
      problem.maxWGs, static_cast<int64_t>(problem.clusters.size()));

  if (root->get("canonical_root")) {
    int64_t rootId = 0;
    if (!getInteger(*root, "canonical_root", rootId))
      return fail(error, "canonical_root must be an integer");
    auto node = nodeIndices.find(rootId);
    if (node == nodeIndices.end())
      return fail(error, "canonical_root names an unknown node");
    problem.canonicalRoot = node->second;
  }

  problem.edges.reserve(edgeValues->size());
  for (const llvm::json::Value &value : *edgeValues) {
    auto *object = value.getAsObject();
    int64_t srcId = 0;
    int64_t dstId = 0;
    int64_t result = 0;
    int64_t latency = 0;
    int64_t distance = 0;
    int64_t frequency = 0;
    int64_t roundTrip = 0;
    int64_t crossIssue = 0;
    int64_t channelBytes = 0;
    if (!object || !getInteger(*object, "src", srcId) ||
        !getInteger(*object, "dst", dstId) ||
        !getInteger(*object, "src_result_idx", result) ||
        !getInteger(*object, "latency", latency) ||
        !getInteger(*object, "distance", distance) ||
        !getInteger(*object, "freq", frequency) ||
        !getInteger(*object, "rt", roundTrip) ||
        !getInteger(*object, "xissue", crossIssue) ||
        !getInteger(*object, "chan_bytes", channelBytes) || result < 0 ||
        latency < 0 || distance < 0 || frequency <= 0 || roundTrip < 0 ||
        crossIssue < 0 || channelBytes < 0)
      return fail(error, "invalid edge in joint problem");
    auto src = nodeIndices.find(srcId);
    auto dst = nodeIndices.find(dstId);
    if (src == nodeIndices.end() || dst == nodeIndices.end())
      return fail(error, "edge names an unknown node");

    std::optional<size_t> srcCluster;
    std::optional<size_t> dstCluster;
    bool hasSrcCluster = object->get("src_cluster") != nullptr;
    bool hasDstCluster = object->get("dst_cluster") != nullptr;
    if (hasSrcCluster != hasDstCluster)
      return fail(error, "edge cluster metadata must be paired");
    if (hasSrcCluster) {
      int64_t srcClusterId = 0;
      int64_t dstClusterId = 0;
      if (!getInteger(*object, "src_cluster", srcClusterId) ||
          !getInteger(*object, "dst_cluster", dstClusterId))
        return fail(error, "edge cluster IDs must be integers");
      auto srcIt = clusterIndices.find(srcClusterId);
      auto dstIt = clusterIndices.find(dstClusterId);
      if (srcIt == clusterIndices.end() || dstIt == clusterIndices.end() ||
          problem.nodes[src->second].cluster != srcIt->second ||
          problem.nodes[dst->second].cluster != dstIt->second)
        return fail(error, "edge cluster metadata disagrees with its nodes");
      srcCluster = srcIt->second;
      dstCluster = dstIt->second;
    } else if (roundTrip != 0 || crossIssue != 0 || channelBytes != 0) {
      return fail(error, "edge communication metadata requires clusters");
    }
    problem.edges.push_back(Edge{src->second, dst->second, result, latency,
                                 distance, roundTrip, channelBytes, srcCluster,
                                 dstCluster});
  }

  std::set<int64_t> bufferIds;
  problem.buffers.reserve(bufferValues->size());
  for (const llvm::json::Value &value : *bufferValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t producerId = 0;
    int64_t sizeBytes = 0;
    int64_t count = 0;
    int64_t minCount = 0;
    std::string kind;
    auto *consumers = object ? object->getArray("consumers") : nullptr;
    if (!object || !getInteger(*object, "id", id) ||
        !getInteger(*object, "producer", producerId) ||
        !getInteger(*object, "size_bytes", sizeBytes) ||
        !getInteger(*object, "count", count) ||
        !getInteger(*object, "min_count", minCount) ||
        !getString(*object, "kind", kind) || !consumers || sizeBytes < 0 ||
        count <= 0 || minCount <= 0 || (kind != "smem" && kind != "tmem") ||
        !bufferIds.insert(id).second)
      return fail(error, "invalid buffer in joint problem");
    auto producer = nodeIndices.find(producerId);
    if (producer == nodeIndices.end())
      return fail(error, "buffer producer is unknown");
    Buffer buffer{id,       producer->second, sizeBytes,
                  minCount, std::move(kind),  {}};
    for (const llvm::json::Value &consumerValue : *consumers) {
      auto *consumer = consumerValue.getAsObject();
      int64_t nodeId = 0;
      int64_t consumerLatency = 0;
      int64_t consumerDistance = 0;
      if (!consumer || !getInteger(*consumer, "node", nodeId) ||
          !getInteger(*consumer, "latency", consumerLatency) ||
          !getInteger(*consumer, "distance", consumerDistance) ||
          consumerLatency < 0 || consumerDistance < 0)
        return fail(error, "invalid buffer consumer");
      auto node = nodeIndices.find(nodeId);
      if (node == nodeIndices.end())
        return fail(error, "buffer consumer is unknown");
      buffer.consumers.push_back(
          Consumer{node->second, consumerLatency, consumerDistance});
    }
    problem.buffers.push_back(std::move(buffer));
  }

  problem.loweringTemplates.reserve(templateValues->size());
  for (size_t templateIndex = 0; templateIndex < templateValues->size();
       ++templateIndex) {
    auto *object = (*templateValues)[templateIndex].getAsObject();
    int64_t id = 0;
    int64_t srcNodeId = 0;
    int64_t dstNodeId = 0;
    int64_t srcClusterId = 0;
    int64_t dstClusterId = 0;
    std::string relationName;
    auto *events = object ? object->getArray("events") : nullptr;
    if (!object || !getInteger(*object, "id", id) ||
        id != static_cast<int64_t>(templateIndex) ||
        !getString(*object, "relation", relationName) ||
        !getInteger(*object, "src_node", srcNodeId) ||
        !getInteger(*object, "dst_node", dstNodeId) ||
        !getInteger(*object, "src_cluster", srcClusterId) ||
        !getInteger(*object, "dst_cluster", dstClusterId) || !events)
      return fail(error, "invalid lowering template");
    Relation relation;
    if (relationName == "always")
      relation = Relation::Always;
    else if (relationName == "same_wg")
      relation = Relation::SameWG;
    else if (relationName == "different_wg")
      relation = Relation::DifferentWG;
    else
      return fail(error, "unknown lowering relation");
    auto srcNode = nodeIndices.find(srcNodeId);
    auto dstNode = nodeIndices.find(dstNodeId);
    auto srcCluster = clusterIndices.find(srcClusterId);
    auto dstCluster = clusterIndices.find(dstClusterId);
    if (srcNode == nodeIndices.end() || dstNode == nodeIndices.end() ||
        srcCluster == clusterIndices.end() ||
        dstCluster == clusterIndices.end() ||
        problem.nodes[srcNode->second].cluster != srcCluster->second ||
        problem.nodes[dstNode->second].cluster != dstCluster->second)
      return fail(error, "lowering template metadata is inconsistent");

    LoweringTemplate loweringTemplate{
        id, relation, srcCluster->second, dstCluster->second, {}};
    std::set<int64_t> eventIds;
    for (const llvm::json::Value &eventValue : *events) {
      auto *event = eventValue.getAsObject();
      int64_t eventId = 0;
      int64_t anchorId = 0;
      int64_t issueDuration = 0;
      int64_t completionLatency = 0;
      int64_t distance = 0;
      int64_t frequency = 0;
      int64_t bytes = 0;
      int64_t depth = 0;
      bool blocking = false;
      bool isAsync = false;
      std::string kind;
      std::string owner;
      std::string placementName;
      std::string pipeline;
      std::string semaphore;
      if (!event || !getInteger(*event, "id", eventId) ||
          !getString(*event, "kind", kind) ||
          (kind != "wait" && kind != "arrive" && kind != "expect" &&
           kind != "local_store" && kind != "local_load" && kind != "fence" &&
           kind != "tc_commit") ||
          !getString(*event, "owner", owner) ||
          !getInteger(*event, "anchor_node", anchorId) ||
          !getString(*event, "placement", placementName) ||
          !getString(*event, "pipeline", pipeline) ||
          !getInteger(*event, "issue_duration", issueDuration) ||
          !getInteger(*event, "completion_latency", completionLatency) ||
          !getBoolean(*event, "blocking", blocking) ||
          !getBoolean(*event, "async", isAsync) ||
          !getInteger(*event, "distance", distance) ||
          !getInteger(*event, "frequency", frequency) ||
          !getInteger(*event, "bytes", bytes) ||
          !getInteger(*event, "depth", depth) ||
          !getString(*event, "semaphore", semaphore) || eventId < 0 ||
          issueDuration < 0 || completionLatency < 0 || distance < 0 ||
          frequency <= 0 || bytes < 0 || depth < 0 ||
          !eventIds.insert(eventId).second)
        return fail(error, "invalid lowering event");
      (void)blocking;
      (void)isAsync;
      auto anchor = nodeIndices.find(anchorId);
      if (anchor == nodeIndices.end())
        return fail(error, "lowering event anchor is unknown");
      size_t ownerCluster;
      if (owner == "src")
        ownerCluster = srcCluster->second;
      else if (owner == "dst")
        ownerCluster = dstCluster->second;
      else
        return fail(error, "unknown lowering event owner");
      if (problem.nodes[anchor->second].cluster != ownerCluster)
        return fail(error, "lowering event anchor has the wrong owner cluster");
      Placement placement;
      if (placementName == "before")
        placement = Placement::Before;
      else if (placementName == "after")
        placement = Placement::After;
      else
        return fail(error, "unknown lowering event placement");
      __int128 duration = static_cast<__int128>(issueDuration) * frequency;
      if (duration > std::numeric_limits<int64_t>::max())
        return fail(error, "lowering event duration overflows int64");
      loweringTemplate.events.push_back(
          LoweringEvent{eventId, anchor->second, placement, std::move(pipeline),
                        static_cast<int64_t>(duration), ownerCluster});
    }
    problem.loweringTemplates.push_back(std::move(loweringTemplate));
  }

  if (auto *conflicts = root->getArray("warp_group_conflicts")) {
    for (const llvm::json::Value &value : *conflicts) {
      auto *pair = value.getAsArray();
      if (!pair || pair->size() != 2)
        return fail(error, "warp_group_conflicts entries must be pairs");
      auto leftId = (*pair)[0].getAsInteger();
      auto rightId = (*pair)[1].getAsInteger();
      if (!leftId || !rightId)
        return fail(error, "warp-group conflict IDs must be integers");
      auto left = clusterIndices.find(*leftId);
      auto right = clusterIndices.find(*rightId);
      if (left == clusterIndices.end() || right == clusterIndices.end())
        return fail(error, "warp-group conflict names an unknown cluster");
      problem.warpGroupConflicts.push_back({left->second, right->second});
    }
  } else if (root->get("warp_group_conflicts")) {
    return fail(error, "warp_group_conflicts must be an array");
  }
  return true;
}

static bool parseAssignments(const llvm::json::Object &root,
                             llvm::StringRef field,
                             const std::map<int64_t, size_t> &indices,
                             std::vector<int64_t> &values,
                             bool requireNonnegative, std::string &error) {
  auto *object = root.getObject(field);
  if (!object)
    return fail(error, field.str() + " must be an object");
  values.assign(indices.size(), 0);
  std::vector<bool> seen(indices.size(), false);
  for (const auto &entry : *object) {
    int64_t id = 0;
    llvm::StringRef key(entry.first);
    auto value = entry.second.getAsInteger();
    if (key.getAsInteger(10, id) || !value)
      return fail(error, field.str() + " contains a non-integer entry");
    auto index = indices.find(id);
    if (index == indices.end() || seen[index->second])
      return fail(error, field.str() + " contains an unknown or repeated ID");
    if (requireNonnegative && *value < 0)
      return fail(error, field.str() + " contains a negative value");
    seen[index->second] = true;
    values[index->second] = *value;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    return fail(error, field.str() + " is incomplete");
  return true;
}

static bool parseSolution(llvm::StringRef json, const Problem &problem,
                          Solution &solution, std::string &error) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return fail(error, "solution JSON parse failed: " +
                           llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root)
    return fail(error, "solution must be a JSON object");
  auto version = root->getString("version");
  auto status = root->getString("status");
  if (!version || *version != kProblemSchema || !status || *status != "ok")
    return fail(error, "solution must be a successful joint-solver-0.2 result");
  if (!root->getInteger("objective"))
    return fail(error, "solution objective must be an integer");
  if (root->get("lowering_objective") &&
      !root->getInteger("lowering_objective"))
    return fail(error, "lowering_objective must be an integer");

  std::map<int64_t, size_t> nodeIndices;
  for (size_t index = 0; index < problem.nodes.size(); ++index)
    nodeIndices.emplace(problem.nodes[index].id, index);
  std::map<int64_t, size_t> clusterIndices;
  for (size_t index = 0; index < problem.clusters.size(); ++index)
    clusterIndices.emplace(problem.clusters[index].id, index);
  if (!parseAssignments(*root, "cycles", nodeIndices, solution.cycles, true,
                        error) ||
      !parseAssignments(*root, "wg", clusterIndices, solution.warpGroups, true,
                        error))
    return false;
  if (!getInteger(*root, "used_wgs", solution.usedWGs) || solution.usedWGs <= 0)
    return fail(error, "invalid used_wgs");

  auto *depths = root->getObject("buffer_depths");
  if (!depths)
    return fail(error, "buffer_depths must be an object");
  std::set<int64_t> expectedDepths;
  for (const Buffer &buffer : problem.buffers)
    if (buffer.kind == "smem")
      expectedDepths.insert(buffer.id);
  for (const auto &entry : *depths) {
    int64_t id = 0;
    llvm::StringRef key(entry.first);
    auto depth = entry.second.getAsInteger();
    if (key.getAsInteger(10, id) || !depth || *depth <= 0 ||
        !expectedDepths.erase(id))
      return fail(error, "buffer_depths contains an invalid entry");
    solution.bufferDepths.emplace(id, *depth);
  }
  if (!expectedDepths.empty())
    return fail(error, "buffer_depths is incomplete");

  auto *loweringPlan = root->getObject("lowering_plan");
  if (!loweringPlan)
    return fail(error, "lowering_plan must be an object");
  auto planVersion = loweringPlan->getString("version");
  auto *plans = loweringPlan->getArray("templates");
  if (!planVersion || *planVersion != kPlanSchema || !plans)
    return fail(error, "invalid lowering_plan");
  solution.loweringPlans.resize(problem.loweringTemplates.size());
  std::map<int64_t, size_t> templateIndices;
  for (size_t index = 0; index < problem.loweringTemplates.size(); ++index)
    templateIndices.emplace(problem.loweringTemplates[index].id, index);
  for (const llvm::json::Value &value : *plans) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    bool active = false;
    auto *events = object ? object->getArray("events") : nullptr;
    if (!object || !getInteger(*object, "id", id) ||
        !getBoolean(*object, "active", active) || !events)
      return fail(error, "invalid lowering template plan");
    auto templateIt = templateIndices.find(id);
    if (templateIt == templateIndices.end() ||
        solution.loweringPlans[templateIt->second].present)
      return fail(error, "lowering_plan has an unknown or repeated template");
    TemplatePlan &plan = solution.loweringPlans[templateIt->second];
    plan.present = true;
    plan.active = active;
    std::set<int64_t> eventIds;
    for (const LoweringEvent &event :
         problem.loweringTemplates[templateIt->second].events)
      eventIds.insert(event.id);
    for (const llvm::json::Value &eventValue : *events) {
      auto *event = eventValue.getAsObject();
      int64_t eventId = 0;
      int64_t cycle = 0;
      int64_t warpGroup = 0;
      int64_t streamOrder = 0;
      if (!event || !getInteger(*event, "id", eventId) ||
          !getInteger(*event, "cycle", cycle) ||
          !getInteger(*event, "wg", warpGroup) ||
          !getInteger(*event, "stream_order", streamOrder) || warpGroup < 0 ||
          streamOrder < 0 || !eventIds.count(eventId) ||
          plan.events.count(eventId))
        return fail(error, "invalid lowering event plan");
      plan.events.emplace(eventId, EventPlan{cycle, warpGroup, streamOrder});
    }
  }
  for (const TemplatePlan &plan : solution.loweringPlans)
    if (!plan.present)
      return fail(error, "lowering_plan is incomplete");
  return true;
}

static void addViolation(ValidationResult &result, std::string message) {
  result.violations.push_back(std::move(message));
}

static std::string wideToString(__int128 value) {
  bool negative = value < 0;
  unsigned __int128 magnitude =
      negative ? static_cast<unsigned __int128>(-(value + 1)) + 1
               : static_cast<unsigned __int128>(value);
  std::string result;
  do {
    result.push_back(static_cast<char>('0' + magnitude % 10));
    magnitude /= 10;
  } while (magnitude != 0);
  if (negative)
    result.push_back('-');
  std::reverse(result.begin(), result.end());
  return result;
}

static bool templateIsActive(const LoweringTemplate &loweringTemplate,
                             const Solution &solution) {
  if (loweringTemplate.relation == Relation::Always)
    return true;
  bool same = solution.warpGroups[loweringTemplate.srcCluster] ==
              solution.warpGroups[loweringTemplate.dstCluster];
  return loweringTemplate.relation == Relation::SameWG ? same : !same;
}

static std::vector<std::vector<ExpectedEvent>>
buildExpectedEvents(const Problem &problem, const Solution &solution,
                    ValidationResult &result) {
  std::vector<std::vector<ExpectedEvent>> expected(
      problem.loweringTemplates.size());
  using EventRef = std::pair<size_t, size_t>;
  std::map<std::pair<size_t, int>, std::vector<EventRef>> groups;
  for (size_t templateIndex = 0;
       templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
    const LoweringTemplate &loweringTemplate =
        problem.loweringTemplates[templateIndex];
    expected[templateIndex].resize(loweringTemplate.events.size());
    if (!templateIsActive(loweringTemplate, solution))
      continue;
    for (size_t eventIndex = 0; eventIndex < loweringTemplate.events.size();
         ++eventIndex) {
      const LoweringEvent &event = loweringTemplate.events[eventIndex];
      groups[{event.anchor, static_cast<int>(event.placement)}].push_back(
          {templateIndex, eventIndex});
    }
  }

  std::vector<ExpectedEvent *> stream;
  for (auto &entry : groups) {
    auto &members = entry.second;
    std::sort(members.begin(), members.end(),
              [&](EventRef left, EventRef right) {
                const LoweringTemplate &leftTemplate =
                    problem.loweringTemplates[left.first];
                const LoweringTemplate &rightTemplate =
                    problem.loweringTemplates[right.first];
                return std::tie(leftTemplate.id,
                                leftTemplate.events[left.second].id) <
                       std::tie(rightTemplate.id,
                                rightTemplate.events[right.second].id);
              });
    size_t anchor = entry.first.first;
    Placement placement = static_cast<Placement>(entry.first.second);
    __int128 cursor = solution.cycles[anchor];
    if (placement == Placement::Before) {
      for (EventRef member : members)
        cursor -= problem.loweringTemplates[member.first]
                      .events[member.second]
                      .duration;
    } else {
      cursor += std::max<int64_t>(problem.nodes[anchor].duration, 1);
    }
    for (EventRef member : members) {
      const LoweringTemplate &loweringTemplate =
          problem.loweringTemplates[member.first];
      const LoweringEvent &event = loweringTemplate.events[member.second];
      ExpectedEvent &out = expected[member.first][member.second];
      out.templateIndex = member.first;
      out.eventIndex = member.second;
      out.warpGroup = solution.warpGroups[event.ownerCluster];
      if (cursor < std::numeric_limits<int>::min() ||
          cursor > std::numeric_limits<int>::max()) {
        out.valid = false;
        addViolation(result, "lowering event cycle is outside int range");
      } else {
        out.cycle = static_cast<int64_t>(cursor);
        stream.push_back(&out);
      }
      cursor += event.duration;
    }
  }

  std::sort(
      stream.begin(), stream.end(),
      [&](const ExpectedEvent *left, const ExpectedEvent *right) {
        const LoweringTemplate &leftTemplate =
            problem.loweringTemplates[left->templateIndex];
        const LoweringTemplate &rightTemplate =
            problem.loweringTemplates[right->templateIndex];
        const LoweringEvent &leftEvent = leftTemplate.events[left->eventIndex];
        const LoweringEvent &rightEvent =
            rightTemplate.events[right->eventIndex];
        int leftPlacement = leftEvent.placement == Placement::Before ? 0 : 2;
        int rightPlacement = rightEvent.placement == Placement::Before ? 0 : 2;
        return std::tie(left->warpGroup, left->cycle, leftPlacement,
                        leftTemplate.id, leftEvent.id) <
               std::tie(right->warpGroup, right->cycle, rightPlacement,
                        rightTemplate.id, rightEvent.id);
      });
  std::map<int64_t, int64_t> nextOrder;
  for (ExpectedEvent *event : stream)
    event->streamOrder = nextOrder[event->warpGroup]++;
  return expected;
}

static void
validateLoweringPlan(const Problem &problem, const Solution &solution,
                     const std::vector<std::vector<ExpectedEvent>> &expected,
                     ValidationResult &result) {
  for (size_t templateIndex = 0;
       templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
    const LoweringTemplate &loweringTemplate =
        problem.loweringTemplates[templateIndex];
    const TemplatePlan &plan = solution.loweringPlans[templateIndex];
    bool active = templateIsActive(loweringTemplate, solution);
    if (plan.active != active)
      addViolation(result, "lowering template " +
                               std::to_string(loweringTemplate.id) +
                               " has the wrong active state");
    if (!active) {
      if (!plan.events.empty())
        addViolation(result, "inactive lowering template " +
                                 std::to_string(loweringTemplate.id) +
                                 " contains events");
      continue;
    }
    if (plan.events.size() != loweringTemplate.events.size())
      addViolation(result, "active lowering template " +
                               std::to_string(loweringTemplate.id) +
                               " has an incomplete event plan");
    for (size_t eventIndex = 0; eventIndex < loweringTemplate.events.size();
         ++eventIndex) {
      const LoweringEvent &event = loweringTemplate.events[eventIndex];
      auto actual = plan.events.find(event.id);
      if (actual == plan.events.end())
        continue;
      const ExpectedEvent &wanted = expected[templateIndex][eventIndex];
      if (wanted.valid && (actual->second.cycle != wanted.cycle ||
                           actual->second.warpGroup != wanted.warpGroup ||
                           actual->second.streamOrder != wanted.streamOrder))
        addViolation(result, "lowering event " +
                                 std::to_string(loweringTemplate.id) + ":" +
                                 std::to_string(event.id) +
                                 " disagrees with deterministic placement");
    }
  }
}

static std::vector<std::pair<__int128, __int128>>
cyclicSegments(int64_t cycle, int64_t duration, int64_t ii) {
  std::vector<std::pair<__int128, __int128>> result;
  if (duration <= 0)
    return result;
  int64_t phase = cycle % ii;
  if (phase < 0)
    phase += ii;
  __int128 end = static_cast<__int128>(phase) + duration;
  if (end <= ii) {
    result.push_back({phase, end});
  } else {
    result.push_back({phase, ii});
    result.push_back({0, end - ii});
  }
  return result;
}

static bool cyclicOverlap(const Issue &left, const Issue &right, int64_t ii) {
  if (left.duration > ii || right.duration > ii)
    return true;
  for (const auto &leftSegment : cyclicSegments(left.cycle, left.duration, ii))
    for (const auto &rightSegment :
         cyclicSegments(right.cycle, right.duration, ii))
      if (std::max(leftSegment.first, rightSegment.first) <
          std::min(leftSegment.second, rightSegment.second))
        return true;
  return false;
}

static void
validateIssueOccupancy(const Problem &problem, const Solution &solution,
                       const std::vector<std::vector<ExpectedEvent>> &expected,
                       ValidationResult &result) {
  std::vector<Issue> items;
  for (size_t index = 0; index < problem.nodes.size(); ++index) {
    const Node &node = problem.nodes[index];
    items.push_back(Issue{solution.cycles[index],
                          std::max<int64_t>(node.duration, 1), node.pipeline,
                          node.cluster, "node " + std::to_string(node.id)});
  }
  for (size_t templateIndex = 0;
       templateIndex < problem.loweringTemplates.size(); ++templateIndex) {
    const LoweringTemplate &loweringTemplate =
        problem.loweringTemplates[templateIndex];
    if (!templateIsActive(loweringTemplate, solution))
      continue;
    for (size_t eventIndex = 0; eventIndex < loweringTemplate.events.size();
         ++eventIndex) {
      const LoweringEvent &event = loweringTemplate.events[eventIndex];
      const ExpectedEvent &placed = expected[templateIndex][eventIndex];
      if (event.duration <= 0 || !placed.valid)
        continue;
      items.push_back(Issue{
          placed.cycle, event.duration, event.pipeline, event.ownerCluster,
          "lowering event " + std::to_string(loweringTemplate.id) + ":" +
              std::to_string(event.id)});
    }
  }

  for (const Issue &item : items)
    if (item.duration > problem.ii)
      addViolation(result, item.label + " occupies more than one II");
  for (size_t leftIndex = 0; leftIndex < items.size(); ++leftIndex) {
    const Issue &left = items[leftIndex];
    for (size_t rightIndex = leftIndex + 1; rightIndex < items.size();
         ++rightIndex) {
      const Issue &right = items[rightIndex];
      if (!cyclicOverlap(left, right, problem.ii))
        continue;
      if (left.pipeline != "NONE" && left.pipeline == right.pipeline)
        addViolation(result, left.label + " overlaps " + right.label +
                                 " on pipeline " + left.pipeline);
      if (problem.jointMode && left.cluster && right.cluster &&
          solution.warpGroups[*left.cluster] ==
              solution.warpGroups[*right.cluster])
        addViolation(result, left.label + " overlaps " + right.label +
                                 " in the same warp group");
    }
  }
}

static __int128 bufferDepth(const Problem &problem, const Solution &solution,
                            const Buffer &buffer) {
  __int128 producerCycle = solution.cycles[buffer.producer];
  __int128 lastEnd = producerCycle;
  for (const Consumer &consumer : buffer.consumers) {
    __int128 end = static_cast<__int128>(solution.cycles[consumer.node]) +
                   consumer.latency +
                   static_cast<__int128>(consumer.distance) * problem.ii;
    lastEnd = std::max(lastEnd, end);
  }
  __int128 depth = (lastEnd - producerCycle) / problem.ii + 1;
  return std::max(depth, static_cast<__int128>(buffer.minCount));
}

static void validateMemory(const Problem &problem, const Solution &solution,
                           ValidationResult &result) {
  __int128 smem = problem.jointMode ? problem.fixedSmem : problem.committedSmem;
  __int128 tmem = 0;
  for (const Buffer &buffer : problem.buffers) {
    __int128 depth = bufferDepth(problem, solution, buffer);
    __int128 charge = static_cast<__int128>(buffer.sizeBytes) * depth;
    if (buffer.kind == "smem") {
      if (problem.jointMode)
        smem += charge;
      auto actual = solution.bufferDepths.find(buffer.id);
      if (actual == solution.bufferDepths.end() || actual->second != depth)
        addViolation(result, "SMEM buffer " + std::to_string(buffer.id) +
                                 " reports the wrong depth");
    } else {
      tmem += charge;
    }
  }

  std::set<std::tuple<size_t, size_t, int64_t>> seenChannels;
  for (const Edge &edge : problem.edges) {
    if (edge.channelBytes <= 0 || !edge.srcCluster || !edge.dstCluster ||
        *edge.srcCluster == *edge.dstCluster)
      continue;
    auto channel = std::make_tuple(edge.src, edge.dst, edge.result);
    if (!seenChannels.insert(channel).second)
      continue;
    if (solution.warpGroups[*edge.srcCluster] !=
        solution.warpGroups[*edge.dstCluster])
      smem += edge.channelBytes;
  }
  if (smem > problem.smemBudget)
    addViolation(result, "SMEM requirement " + wideToString(smem) +
                             " exceeds budget " +
                             std::to_string(problem.smemBudget));
  if (tmem > problem.tmemBudget)
    addViolation(result, "TMEM requirement " + wideToString(tmem) +
                             " exceeds budget " +
                             std::to_string(problem.tmemBudget));
}

static void validateRegisters(const Problem &problem, const Solution &solution,
                              ValidationResult &result) {
  if (!problem.regBudget)
    return;
  __int128 total = problem.defaultWGFootprint;
  for (size_t group = 0; group < problem.clusters.size(); ++group) {
    int64_t requiredWarps = 0;
    for (size_t cluster = 0; cluster < problem.clusters.size(); ++cluster)
      if (solution.warpGroups[cluster] == static_cast<int64_t>(group))
        requiredWarps =
            std::max(requiredWarps, problem.clusters[cluster].minWarps);
    int roundedWarps = requiredWarps <= 0   ? 0
                       : requiredWarps <= 1 ? 1
                       : requiredWarps <= 2 ? 2
                       : requiredWarps <= 4 ? 4
                                            : 8;
    total += problem.warpFootprint[roundedWarps];
  }
  if (total > *problem.regBudget)
    addViolation(result, "register requirement " + wideToString(total) +
                             " exceeds hard cap " +
                             std::to_string(*problem.regBudget));
}

static void validateSchedule(const Problem &problem, const Solution &solution,
                             ValidationResult &result) {
  for (size_t index = 0; index < problem.nodes.size(); ++index) {
    const Node &node = problem.nodes[index];
    bool fixed = problem.jointMode ? solution.cycles[index] / problem.ii ==
                                         node.fixedCycle / problem.ii
                                   : solution.cycles[index] == node.fixedCycle;
    if (!fixed)
      addViolation(result,
                   "node " + std::to_string(node.id) +
                       (problem.jointMode ? " moved out of its committed stage"
                                          : " moved from its fixed cycle"));
  }
  if (problem.canonicalRoot && solution.cycles[*problem.canonicalRoot] != 0)
    addViolation(result, "canonical root is not scheduled at cycle zero");

  int64_t prefixMaximum = -1;
  std::set<int64_t> distinctGroups;
  for (size_t index = 0; index < problem.clusters.size(); ++index) {
    int64_t group = solution.warpGroups[index];
    if (group < 0 || group >= static_cast<int64_t>(problem.clusters.size())) {
      addViolation(result, "cluster " +
                               std::to_string(problem.clusters[index].id) +
                               " has an invalid warp-group label");
      continue;
    }
    if ((index == 0 && group != 0) || (index > 0 && group > prefixMaximum + 1))
      addViolation(result, "warp-group labels violate canonical ordering");
    prefixMaximum = std::max(prefixMaximum, group);
    distinctGroups.insert(group);
  }
  int64_t usedWGs = prefixMaximum + 1;
  if (usedWGs != solution.usedWGs ||
      static_cast<int64_t>(distinctGroups.size()) != solution.usedWGs)
    addViolation(result, "used_wgs disagrees with the warp-group assignment");
  if (usedWGs > problem.maxWGs)
    addViolation(result, "warp-group assignment exceeds max_wgs");

  for (const auto &[left, right] : problem.warpGroupConflicts)
    if (solution.warpGroups[left] == solution.warpGroups[right])
      addViolation(result, "conflicting clusters " +
                               std::to_string(problem.clusters[left].id) +
                               " and " +
                               std::to_string(problem.clusters[right].id) +
                               " share a warp group");

  for (const Edge &edge : problem.edges) {
    const Node &src = problem.nodes[edge.src];
    const Node &dst = problem.nodes[edge.dst];
    int64_t latency =
        edge.distance == 0 ? std::max<int64_t>(edge.latency, 1) : edge.latency;
    __int128 required = static_cast<__int128>(solution.cycles[edge.src]) +
                        latency -
                        static_cast<__int128>(edge.distance) * problem.ii;
    if (problem.jointMode && solution.cycles[edge.dst] < required)
      addViolation(result, "dependence N" + std::to_string(src.id) + " -> N" +
                               std::to_string(dst.id) + " is violated");

    if (!edge.srcCluster || !edge.dstCluster ||
        *edge.srcCluster == *edge.dstCluster)
      continue;
    bool sameWG = solution.warpGroups[*edge.srcCluster] ==
                  solution.warpGroups[*edge.dstCluster];
    if ((src.pipeline == "TC" || src.pipeline == "MFMA") &&
        (dst.pipeline == "CUDA" || dst.pipeline == "SFU") && sameWG)
      addViolation(result, "tensor-core result N" + std::to_string(src.id) +
                               " shares a warp group with software reader N" +
                               std::to_string(dst.id));
    if (edge.distance > 0 && edge.roundTrip > 0 && !sameWG)
      addViolation(result, "loop-carried value N" + std::to_string(src.id) +
                               " -> N" + std::to_string(dst.id) +
                               " crosses warp groups");
    if (problem.jointMode && edge.roundTrip > 0 &&
        !(edge.distance > 0 && edge.roundTrip > 0) && !sameWG &&
        solution.cycles[edge.dst] < required + edge.roundTrip)
      addViolation(result, "cross-WG round trip N" + std::to_string(src.id) +
                               " -> N" + std::to_string(dst.id) +
                               " is violated");
  }
}

namespace v01 {

constexpr llvm::StringLiteral kProblemSchema = "joint-solver-0.1";

struct Node {
  int64_t id;
  std::string pipeline;
  int64_t duration;
  bool streaming;
};

struct Edge {
  size_t src;
  size_t dst;
  int64_t latency;
  int64_t distance;
};

enum class BufferKind { Smem, Tmem };

struct Buffer {
  int64_t id;
  size_t alloc;
  BufferKind kind;
  int64_t sizeBytes;
  int64_t tmemCols;
  std::vector<size_t> consumers;
};

struct Problem {
  int64_t minII;
  int64_t maxII;
  int64_t smemBudget;
  int64_t tmemColLimit;
  bool streamingVL;
  std::optional<size_t> canonicalRoot;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<Buffer> buffers;
};

struct Solution {
  int64_t ii;
  std::vector<int64_t> cycles;
  std::map<int64_t, int64_t> bufferDepths;
};

static bool parseProblem(llvm::StringRef json, Problem &problem,
                         std::string &error) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return fail(error, "problem JSON parse failed: " +
                           llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root || root->get("mode"))
    return fail(error, "invalid joint-solver-0.1 problem object");
  auto version = root->getString("version");
  if (!version || *version != kProblemSchema)
    return fail(error, "validator requires a joint-solver-0.1 problem");
  if (!getInteger(*root, "min_ii", problem.minII) ||
      !getInteger(*root, "max_ii", problem.maxII) ||
      !getInteger(*root, "smem_budget", problem.smemBudget) ||
      !getInteger(*root, "tmem_col_limit", problem.tmemColLimit) ||
      problem.minII <= 0 || problem.maxII < problem.minII ||
      problem.smemBudget < 0 || problem.tmemColLimit < 0)
    return fail(error, "invalid joint-solver-0.1 budgets or II range");

  if (root->get("time_limit_s")) {
    auto value = root->getNumber("time_limit_s");
    if (!value || !std::isfinite(*value) || *value <= 0.0)
      return fail(error, "invalid time_limit_s");
  }
  problem.streamingVL = false;
  if (root->get("streaming_vl")) {
    auto value = root->getBoolean("streaming_vl");
    if (!value)
      return fail(error, "streaming_vl must be Boolean");
    problem.streamingVL = *value;
  }

  auto *nodeValues = root->getArray("nodes");
  auto *edgeValues = root->getArray("edges");
  auto *bufferValues = root->getArray("buffers");
  if (!nodeValues || !edgeValues || !bufferValues)
    return fail(error, "joint-solver-0.1 problem is missing a required array");

  std::map<int64_t, size_t> nodeIndices;
  problem.nodes.reserve(nodeValues->size());
  for (const llvm::json::Value &value : *nodeValues) {
    auto *object = value.getAsObject();
    int64_t id = 0;
    int64_t duration = 0;
    std::string pipeline;
    if (!object || !getInteger(*object, "id", id) ||
        !getInteger(*object, "duration", duration) || duration < 0 ||
        !getString(*object, "pipeline", pipeline) || nodeIndices.count(id))
      return fail(error, "invalid node in joint-solver-0.1 problem");
    bool streaming = false;
    if (object->get("streaming")) {
      auto parsedStreaming = object->getBoolean("streaming");
      if (!parsedStreaming)
        return fail(error, "node streaming flag must be Boolean");
      streaming = *parsedStreaming;
    }
    nodeIndices.emplace(id, problem.nodes.size());
    problem.nodes.push_back(Node{id, std::move(pipeline), duration, streaming});
  }

  if (root->get("canonical_root")) {
    int64_t rootId = 0;
    if (!getInteger(*root, "canonical_root", rootId))
      return fail(error, "canonical_root must be an integer");
    auto rootNode = nodeIndices.find(rootId);
    if (rootNode == nodeIndices.end())
      return fail(error, "canonical_root names an unknown node");
    problem.canonicalRoot = rootNode->second;
  }

  problem.edges.reserve(edgeValues->size());
  for (const llvm::json::Value &value : *edgeValues) {
    auto *object = value.getAsObject();
    int64_t srcId = 0;
    int64_t dstId = 0;
    int64_t latency = 0;
    int64_t distance = 0;
    if (!object || !getInteger(*object, "src", srcId) ||
        !getInteger(*object, "dst", dstId) ||
        !getInteger(*object, "latency", latency) ||
        !getInteger(*object, "distance", distance) || latency < 0 ||
        distance < 0)
      return fail(error, "invalid edge in joint-solver-0.1 problem");
    auto src = nodeIndices.find(srcId);
    auto dst = nodeIndices.find(dstId);
    if (src == nodeIndices.end() || dst == nodeIndices.end())
      return fail(error, "edge names an unknown node");
    problem.edges.push_back(Edge{src->second, dst->second, latency, distance});
  }

  problem.buffers.reserve(bufferValues->size());
  for (const llvm::json::Value &value : *bufferValues) {
    auto *object = value.getAsObject();
    int64_t allocId = 0;
    int64_t sizeBytes = 0;
    int64_t tmemCols = 0;
    std::string kindName;
    auto *consumers = object ? object->getArray("consumers") : nullptr;
    if (!object || !getInteger(*object, "alloc_node", allocId) ||
        !getString(*object, "kind", kindName) || !consumers ||
        !getInteger(*object, "size_bytes", sizeBytes) ||
        !getInteger(*object, "tmem_cols", tmemCols) || sizeBytes < 0 ||
        tmemCols < 0)
      return fail(error, "invalid buffer in joint-solver-0.1 problem");
    auto alloc = nodeIndices.find(allocId);
    if (alloc == nodeIndices.end())
      return fail(error, "buffer allocation names an unknown node");
    BufferKind kind;
    if (kindName == "smem")
      kind = BufferKind::Smem;
    else if (kindName == "tmem")
      kind = BufferKind::Tmem;
    else
      return fail(error, "unknown joint-solver-0.1 buffer kind");
    int64_t id = allocId;
    if (object->get("id") && !getInteger(*object, "id", id))
      return fail(error, "buffer id must be an integer");
    Buffer buffer{id, alloc->second, kind, sizeBytes, tmemCols, {}};
    buffer.consumers.reserve(consumers->size());
    for (const llvm::json::Value &consumerValue : *consumers) {
      auto consumerId = consumerValue.getAsInteger();
      if (!consumerId)
        return fail(error, "buffer consumer ID must be an integer");
      auto consumer = nodeIndices.find(*consumerId);
      if (consumer == nodeIndices.end())
        return fail(error, "buffer consumer names an unknown node");
      buffer.consumers.push_back(consumer->second);
    }
    problem.buffers.push_back(std::move(buffer));
  }
  return true;
}

static bool parseSolution(llvm::StringRef json, const Problem &problem,
                          Solution &solution, std::string &error) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return fail(error, "solution JSON parse failed: " +
                           llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root)
    return fail(error, "solution must be a JSON object");
  auto version = root->getString("version");
  auto status = root->getString("status");
  if (!version || *version != kProblemSchema || !status || *status != "ok")
    return fail(error, "solution must be a successful joint-solver-0.1 result");
  if (!getInteger(*root, "ii", solution.ii) || solution.ii <= 0)
    return fail(error, "invalid solution II");
  auto objective = root->getNumber("objective");
  if (!objective || !std::isfinite(*objective))
    return fail(error, "solution objective must be a finite number");
  auto *stats = root->getObject("stats");
  if (!stats)
    return fail(error, "solution stats must be an object");

  auto parseIIStats = [&](llvm::StringRef key,
                          std::set<int64_t> &values) -> bool {
    auto *array = stats->getArray(key);
    if (!array)
      return false;
    std::optional<int64_t> previous;
    for (const llvm::json::Value &value : *array) {
      auto ii = value.getAsInteger();
      if (!ii || *ii < problem.minII || *ii > problem.maxII ||
          !values.insert(*ii).second || (previous && *ii <= *previous))
        return false;
      previous = *ii;
    }
    return true;
  };
  std::set<int64_t> tried;
  std::set<int64_t> unsat;
  std::set<int64_t> unknown;
  if (!parseIIStats("iis_tried", tried) || !parseIIStats("unsat_iis", unsat) ||
      !parseIIStats("unknown_iis", unknown))
    return fail(error, "solution stats contains an invalid II list");
  if (!tried.count(solution.ii))
    return fail(error, "solution stats omits the selected II");
  for (int64_t ii : unsat)
    if (!tried.count(ii))
      return fail(error, "solution stats reports an untried UNSAT II");
  if (!unknown.empty())
    return fail(error, "successful solution stats contains an unknown II");

  std::map<int64_t, size_t> nodeIndices;
  for (size_t index = 0; index < problem.nodes.size(); ++index)
    nodeIndices.emplace(problem.nodes[index].id, index);
  if (!parseAssignments(*root, "cycles", nodeIndices, solution.cycles, true,
                        error))
    return false;

  auto *depths = root->getObject("buffer_depths");
  if (!depths)
    return fail(error, "buffer_depths must be an object");
  std::set<int64_t> expectedDepths;
  for (const Buffer &buffer : problem.buffers)
    if (buffer.kind == BufferKind::Smem)
      expectedDepths.insert(buffer.id);
  for (const auto &entry : *depths) {
    int64_t id = 0;
    llvm::StringRef key(entry.first);
    auto depth = entry.second.getAsInteger();
    if (key.getAsInteger(10, id) || !depth || *depth <= 0 ||
        !expectedDepths.erase(id))
      return fail(error, "buffer_depths contains an invalid entry");
    solution.bufferDepths.emplace(id, *depth);
  }
  if (!expectedDepths.empty())
    return fail(error, "buffer_depths is incomplete");
  return true;
}

static int64_t effectiveLatency(const Problem &problem, const Edge &edge) {
  if (problem.streamingVL && problem.nodes[edge.src].streaming)
    return 0;
  return edge.latency;
}

static std::optional<int64_t> computeHorizon(const Problem &problem,
                                             int64_t ii) {
  if (problem.nodes.empty())
    return ii;
  __int128 maxAdvance = 1;
  for (const Edge &edge : problem.edges) {
    __int128 residual = static_cast<__int128>(effectiveLatency(problem, edge)) -
                        static_cast<__int128>(edge.distance) * ii;
    __int128 advance = (residual + static_cast<__int128>(2) * ii - 2) / ii;
    maxAdvance = std::max(maxAdvance, advance);
  }
  __int128 maxStages =
      1 + static_cast<__int128>(problem.nodes.size() - 1) * maxAdvance;
  maxStages = std::max<__int128>(maxStages, 4);
  __int128 horizon = maxStages * ii;
  if (horizon <= 0 || horizon > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(horizon);
}

static void validateSchedule(const Problem &problem, const Solution &solution,
                             ValidationResult &result) {
  if (solution.ii < problem.minII || solution.ii > problem.maxII)
    addViolation(result, "solution II is outside the requested range");
  auto horizon = computeHorizon(problem, solution.ii);
  if (!horizon) {
    addViolation(result, "joint-solver-0.1 horizon overflows int64");
  } else {
    for (size_t index = 0; index < problem.nodes.size(); ++index)
      if (solution.cycles[index] >= *horizon)
        addViolation(result, "node " + std::to_string(problem.nodes[index].id) +
                                 " is outside the solver horizon");
  }

  if (problem.canonicalRoot) {
    if (solution.cycles[*problem.canonicalRoot] != 0)
      addViolation(result, "canonical root is not scheduled at cycle zero");
  } else if (!solution.cycles.empty() &&
             std::find(solution.cycles.begin(), solution.cycles.end(), 0) ==
                 solution.cycles.end()) {
    addViolation(result, "schedule has no node at cycle zero");
  }

  for (const Edge &edge : problem.edges) {
    const Node &src = problem.nodes[edge.src];
    const Node &dst = problem.nodes[edge.dst];
    __int128 required = static_cast<__int128>(solution.cycles[edge.src]) +
                        effectiveLatency(problem, edge) -
                        static_cast<__int128>(edge.distance) * solution.ii;
    if (solution.cycles[edge.dst] < required)
      addViolation(result, "dependence N" + std::to_string(src.id) + " -> N" +
                               std::to_string(dst.id) + " is violated");
    if (edge.distance == 0 && solution.cycles[edge.src] / solution.ii >
                                  solution.cycles[edge.dst] / solution.ii)
      addViolation(result, "dependence N" + std::to_string(src.id) + " -> N" +
                               std::to_string(dst.id) +
                               " reverses stage order");
  }

  std::map<std::string, std::vector<size_t>> nodesByPipeline;
  for (size_t index = 0; index < problem.nodes.size(); ++index) {
    const Node &node = problem.nodes[index];
    if (node.pipeline == "NONE")
      continue;
    int64_t duration = std::max<int64_t>(node.duration, 1);
    if (duration > solution.ii)
      addViolation(result, "node " + std::to_string(node.id) +
                               " occupies more than one II");
    nodesByPipeline[node.pipeline].push_back(index);
  }
  for (const auto &entry : nodesByPipeline) {
    const std::vector<size_t> &members = entry.second;
    for (size_t leftIndex = 0; leftIndex < members.size(); ++leftIndex) {
      size_t left = members[leftIndex];
      Issue leftIssue{solution.cycles[left],
                      std::max<int64_t>(problem.nodes[left].duration, 1),
                      entry.first,
                      std::nullopt,
                      {}};
      for (size_t rightIndex = leftIndex + 1; rightIndex < members.size();
           ++rightIndex) {
        size_t right = members[rightIndex];
        Issue rightIssue{solution.cycles[right],
                         std::max<int64_t>(problem.nodes[right].duration, 1),
                         entry.first,
                         std::nullopt,
                         {}};
        if (cyclicOverlap(leftIssue, rightIssue, solution.ii))
          addViolation(result,
                       "nodes " + std::to_string(problem.nodes[left].id) +
                           " and " + std::to_string(problem.nodes[right].id) +
                           " overlap on pipeline " + entry.first);
      }
    }
  }
}

static void validateMemory(const Problem &problem, const Solution &solution,
                           ValidationResult &result) {
  struct TmemLifetime {
    __int128 start;
    __int128 end;
    int64_t columns;
  };
  __int128 smem = 0;
  std::vector<TmemLifetime> tmemLifetimes;
  for (const Buffer &buffer : problem.buffers) {
    if (buffer.kind == BufferKind::Smem) {
      int64_t allocStage = solution.cycles[buffer.alloc] / solution.ii;
      int64_t lastStage = allocStage;
      for (size_t consumer : buffer.consumers)
        lastStage =
            std::max(lastStage, solution.cycles[consumer] / solution.ii);
      int64_t depth = lastStage - allocStage + 1;
      auto actual = solution.bufferDepths.find(buffer.id);
      if (actual == solution.bufferDepths.end() || actual->second != depth)
        addViolation(result, "SMEM buffer " + std::to_string(buffer.id) +
                                 " reports the wrong depth");
      smem += static_cast<__int128>(buffer.sizeBytes) * depth;
      continue;
    }

    int64_t lastCycle = solution.cycles[buffer.alloc];
    for (size_t consumer : buffer.consumers)
      lastCycle = std::max(lastCycle, solution.cycles[consumer]);
    tmemLifetimes.push_back(TmemLifetime{solution.cycles[buffer.alloc],
                                         static_cast<__int128>(lastCycle) + 1,
                                         buffer.tmemCols});
  }
  if (smem > problem.smemBudget)
    addViolation(result, "SMEM requirement " + wideToString(smem) +
                             " exceeds budget " +
                             std::to_string(problem.smemBudget));

  __int128 peakTmem = 0;
  for (const TmemLifetime &checkpoint : tmemLifetimes) {
    __int128 active = 0;
    for (const TmemLifetime &lifetime : tmemLifetimes)
      if (lifetime.start <= checkpoint.start && checkpoint.start < lifetime.end)
        active += lifetime.columns;
    peakTmem = std::max(peakTmem, active);
  }
  if (peakTmem > problem.tmemColLimit)
    addViolation(result, "TMEM requirement " + wideToString(peakTmem) +
                             " exceeds column limit " +
                             std::to_string(problem.tmemColLimit));
}

static void validate(const Problem &problem, const Solution &solution,
                     ValidationResult &result) {
  validateSchedule(problem, solution, result);
  validateMemory(problem, solution, result);
}

} // namespace v01

static bool readProblemSchema(llvm::StringRef json, std::string &schema,
                              std::string &error) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return fail(error, "problem JSON parse failed: " +
                           llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root)
    return fail(error, "problem must be a JSON object");
  auto version = root->getString("version");
  if (!version)
    return fail(error, "problem version must be a string");
  schema = version->str();
  return true;
}

static std::string serializeResult(ValidationResult result) {
  std::vector<std::string> normalized;
  std::set<std::string> seen;
  for (std::string &violation : result.violations)
    if (seen.insert(violation).second)
      normalized.push_back(std::move(violation));
  bool valid = normalized.empty();
  llvm::json::Array violations;
  for (std::string &violation : normalized)
    violations.push_back(std::move(violation));
  llvm::json::Object output{
      {"schema", kValidationSchema},
      {"valid", valid},
      {"message", valid ? "solution satisfies solver constraints"
                        : "solution violates solver constraints"},
      {"violations", std::move(violations)},
  };
  std::string json;
  llvm::raw_string_ostream stream(json);
  stream << llvm::json::Value(std::move(output));
  stream.flush();
  return json;
}

} // namespace

std::string validateZ3JointSolution(llvm::StringRef problemJson,
                                    llvm::StringRef solutionJson) {
  ValidationResult result;
  std::string schema;
  std::string error;
  if (!readProblemSchema(problemJson, schema, error)) {
    addViolation(result, std::move(error));
    return serializeResult(std::move(result));
  }
  if (schema == v01::kProblemSchema) {
    v01::Problem problem;
    v01::Solution solution;
    if (!v01::parseProblem(problemJson, problem, error)) {
      addViolation(result, std::move(error));
      return serializeResult(std::move(result));
    }
    if (!v01::parseSolution(solutionJson, problem, solution, error)) {
      addViolation(result, std::move(error));
      return serializeResult(std::move(result));
    }
    v01::validate(problem, solution, result);
    return serializeResult(std::move(result));
  }
  if (schema != kProblemSchema) {
    addViolation(result, "validator requires a joint-solver-0.1 or "
                         "joint-solver-0.2 problem");
    return serializeResult(std::move(result));
  }
  Problem problem;
  Solution solution;
  if (!parseProblem(problemJson, problem, error)) {
    addViolation(result, std::move(error));
    return serializeResult(std::move(result));
  }
  if (!parseSolution(solutionJson, problem, solution, error)) {
    addViolation(result, std::move(error));
    return serializeResult(std::move(result));
  }

  validateSchedule(problem, solution, result);
  auto expectedEvents = buildExpectedEvents(problem, solution, result);
  validateLoweringPlan(problem, solution, expectedEvents, result);
  validateIssueOccupancy(problem, solution, expectedEvents, result);
  validateMemory(problem, solution, result);
  validateRegisters(problem, solution, result);
  return serializeResult(std::move(result));
}

} // namespace mlir::triton::gpu
