#include "test/include/Analysis/TestProgramIdAddress.h"

namespace mlir {
namespace test {
void registerTestProgramIdAddressPass() {
  PassRegistration<TestProgramIdAddressPass>();
}
} // namespace test
} // namespace mlir
