#include "third_party/nvidia/hopper/lib/Transforms/ModuloScheduling/Z3ConstraintEncoder.h"

#include <gtest/gtest.h>

#ifdef TRITON_ENABLE_Z3_JOINT_SOLVER

namespace mlir::triton::gpu {
namespace {

TEST(Z3ConstraintEncoderTest, EncodesIntegerConstraints) {
  Z3_config config = Z3_mk_config();
  ASSERT_NE(config, nullptr);
  Z3_context context = Z3_mk_context(config);
  Z3_del_config(config);
  ASSERT_NE(context, nullptr);

  Z3_solver solver = Z3_mk_solver(context);
  ASSERT_NE(solver, nullptr);
  Z3_solver_inc_ref(context, solver);

  Z3ConstraintEncoder encoder(context, solver);
  Z3_ast x = encoder.variable("x");
  Z3_ast y = encoder.variable("y");
  encoder.assertFormula(Z3_mk_ge(context, x, encoder.intValue(0)));
  encoder.assertFormula(
      Z3_mk_eq(context, y, encoder.add(x, encoder.intValue(3))));
  encoder.assertFormula(
      Z3_mk_eq(context, encoder.maximum(x, y), encoder.intValue(5)));

  ASSERT_EQ(Z3_solver_check(context, solver), Z3_L_TRUE);
  Z3_model model = Z3_solver_get_model(context, solver);
  ASSERT_NE(model, nullptr);
  Z3_model_inc_ref(context, model);
  Z3_ast value = nullptr;
  int64_t integer = 0;
  ASSERT_TRUE(Z3_model_eval(context, model, x, true, &value));
  ASSERT_TRUE(Z3_get_numeral_int64(context, value, &integer));
  EXPECT_EQ(integer, 2);

  Z3_model_dec_ref(context, model);
  Z3_solver_dec_ref(context, solver);
  Z3_del_context(context);
}

} // namespace
} // namespace mlir::triton::gpu

#endif // TRITON_ENABLE_Z3_JOINT_SOLVER
