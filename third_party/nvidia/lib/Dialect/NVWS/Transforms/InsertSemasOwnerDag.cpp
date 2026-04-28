// OWNER analysis; see sema-docs/insert-semas/owner-dag.md.
#include "InsertSemas.h"

namespace mlir::triton::nvws_semas {
static Node *wrapChainWithBrackets(GroupDag &g, Node *regionNode, Node *head) {
  Node *enter = g.newNode(Node::Enter, nullptr, regionNode);
  Node *exit = g.newNode(Node::Exit, nullptr, regionNode);
  if (head) {
    enter->next = head;
    head->prev = enter;
    Node *tail = head;
    while (tail->next)
      tail = tail->next;
    tail->next = exit;
    exit->prev = tail;
  } else {
    enter->next = exit;
    exit->prev = enter;
  }
  return enter;
}
static void spliceEnterExit(GroupDag &g, Node *chainHead) {
  for (Node *n = chainHead; n; n = n->next) {
    if (!n->isRegion())
      continue;
    if (n->kind == Node::If)
      assert(n->children.size() == 2 && "If node carries then+else slots");
    for (Node *&child : n->children) {
      if (child)
        spliceEnterExit(g, child);
      child = wrapChainWithBrackets(g, n, child);
    }
  }
}
static bool toucherContribution(GroupDag &g, Node *n, PieceId piece, Owner &out) {
  if (n->kind == Node::Access) {
    if (!touchesPiece(g, n, piece))
      return false;
    out = n->owner;
    return true;
  }
  if (n->isRegion()) {
    auto it = n->pieceInfo.find(piece);
    if (it != n->pieceInfo.end()) {
      bool sealed = n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op);
      out = sealed ? Owner(std::nullopt) : it->second.owner;
      return true;
    }
  }
  return false; // Enter/Exit and non-touching rows contribute nothing.
}
static bool findOwner(GroupDag &g, Node *start, PieceId piece, Owner &out, bool forward) {
  for (Node *n = start; n; n = forward ? n->next : n->prev)
    if (toucherContribution(g, n, piece, out))
      return true;
  return false;
}
static Node *chainTail(Node *head) {
  Node *tail = head;
  while (tail && tail->next)
    tail = tail->next;
  return tail;
}
static void chainEffectFootprint(GroupDag &g, Node *head, DenseMap<PieceId, Effect> &out) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access) {
      forEachTouchedPiece(g, n, [&](PieceId p, Effect e) { mergeEffect(out, p, e); });
      continue;
    }
    if (n->isRegion())
      for (const auto &[piece, info] : n->pieceInfo)
        mergeEffect(out, piece, info.effect);
  }
}

static LogicalResult assignOwners(GroupDag &g, Node *chainHead) {
  for (Node *n = chainHead; n; n = n->next) {
    if (!n->isRegion())
      continue;
    for (Node *childHead : n->children)
      if (failed(assignOwners(g, childHead)))
        return failure();
    for (auto [p, _] : sortedPieceInfo(n)) {
      Owner owner;
      bool found = false;
      if (n->kind == Node::For) {
        found = findOwner(g, n->children[0], p, owner, true);
      } else {
        found = findOwner(g, n->prev, p, owner, false) || findOwner(g, n->children[0], p, owner, true) ||
                findOwner(g, n->children[1], p, owner, true);
      }
      if (!found)
        return semaError(n->op) << "no toucher resolves the owner for a piece in this region's "
                  "summary (stage-1/stage-2 inconsistency)";
      n->pieceInfo[p].owner = owner;
    }
    for (Node *childHead : n->children) {
      DenseMap<PieceId, Effect> fp;
      chainEffectFootprint(g, childHead, fp);
      DenseMap<PieceId, PieceInfo> rec;
      for (const auto &kv : fp) {
        auto it = n->pieceInfo.find(kv.first);
        assert(it != n->pieceInfo.end() && "branch footprint exceeds region summary");
        rec.try_emplace(kv.first, PieceInfo{it->second.owner, kv.second});
      }
      childHead->pieceInfo = rec;             // Enter
      chainTail(childHead)->pieceInfo = rec;  // Exit
    }
  }
  return success();
}
static LogicalResult verifyOwnerDag(GroupDag &g, Node *chainHead) {
  return forEachNodeChecked(chainHead, [&](Node *n) -> LogicalResult {
    if (!n->isRegion())
      return success();
    for (Node *childHead : n->children) {
      DenseMap<PieceId, Effect> fp;
      chainEffectFootprint(g, childHead, fp);
      for (Node *bracket : {childHead, chainTail(childHead)}) {
        if (bracket->pieceInfo.size() != fp.size())
          return semaError(n->op) << "bracket footprint size mismatch";
        for (const auto &kv : bracket->pieceInfo) {
          auto fpIt = fp.find(kv.first);
          auto regIt = n->pieceInfo.find(kv.first);
          if (fpIt == fp.end() || regIt == n->pieceInfo.end() || kv.second.effect != fpIt->second ||
              !sameOwner(kv.second.owner, regIt->second.owner))
            return semaError(n->op) << "bracket record violates the restriction rule";
        }
      }
    }
    return success();
  });
}

LogicalResult buildOwnerDag(GroupDag &g) {
  if (g.root->children.empty())
    return success();
  spliceEnterExit(g, g.root->children[0]);
  if (failed(assignOwners(g, g.root->children[0])))
    return failure();
  return verifyOwnerDag(g, g.root->children[0]);
}
void printPieceRecord(llvm::raw_ostream &os, const Node *n, Operation *anchor) {
  if (n->pieceInfo.empty())
    return;
  os << " pieces{";
  bool first = true;
  for (const auto &[p, info] : sortedPieceInfo(n)) {
    if (!first)
      os << ",";
    first = false;
    os << "P" << p << ":" << (info.effect == Effect::W ? "W" : "R") << ":" << ownerStr(anchor, info.owner);
  }
  os << "}";
}

void dumpGroupOwnerDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "OWNER-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  dumpDagTree(g, DumpStage::Owner);
}
} // namespace mlir::triton::nvws_semas
