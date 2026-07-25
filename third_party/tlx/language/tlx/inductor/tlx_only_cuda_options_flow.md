# tlx_only_cuda_options Flow in Inductor TLX Templates

This document shows how `tlx_only_cuda_options` (e.g., `ctas_per_cga`) flows from the TLX template heuristic configuration to the Triton kernel decorator.

---

## Complete Data Flow

```mermaid
sequenceDiagram
    participant Registry as tlx/inductor/registry.py
    participant Utils as utils.py
    participant Choices as choices.py
    participant KTC as KernelTemplateChoice
    participant Template as TritonTemplate
    participant Caller as TritonTemplateCaller
    participant Kernel as TritonTemplateKernel
    participant Heuristics as triton_heuristics.py
    participant Triton as Triton Compiler

    Registry->>Registry: Define tlx_only_cuda_options = ["ctas_per_cga"]
    Registry->>Registry: TLX2CTAConfigMixin.get_extra_kwargs() returns {ctas_per_cga: (2,1,1)}

    Note over Choices: Template selection begins
    Choices->>Utils: tlx_only_cuda_options()
    Utils->>Registry: Import tlx_only_cuda_options
    Registry-->>Utils: ["ctas_per_cga"]
    Utils-->>Choices: ["ctas_per_cga"]

    Choices->>Registry: heuristic.get_extra_kwargs()
    Registry-->>Choices: {ctas_per_cga: (2,1,1)}

    Choices->>KTC: make_ktc_generator(extra_kwargs)
    KTC->>Template: choice_or_none(**extra_kwargs)
    Template->>Caller: TritonTemplateCaller(kwargs with ctas_per_cga)

    Note over Kernel: Code generation begins
    Caller->>Kernel: TritonTemplateKernel(meta=kwargs)
    Kernel->>Kernel: self.meta = meta

    Kernel->>Utils: tlx_only_cuda_options()
    Utils-->>Kernel: ["ctas_per_cga"]
    Kernel->>Kernel: for k in tlx_only_cuda_options(): add to template_args

    Kernel->>Heuristics: @triton_heuristics.template(ctas_per_cga=(2,1,1))
    Heuristics->>Heuristics: triton.Config(ctas_per_cga=(2,1,1))
    Heuristics->>Heuristics: _create_compile_meta() extracts ctas_per_cga
    Heuristics->>Heuristics: _create_compile_options() sets cluster_dims
    Heuristics->>Triton: Compile with cluster_dims=(2,1,1)
    Triton->>Triton: CTA Cluster launch on Blackwell GPU
```

---

## Adding a New tlx_only_cuda_option

To add a new FB-only option (e.g., `my_new_option`):

1. **Add to registry.py**:
   ```python
   tlx_only_cuda_options = ["ctas_per_cga", "my_new_option"]
   ```

2. **Return from get_extra_kwargs()**:
   ```python
   def get_extra_kwargs(self, kernel_inputs, op_name):
       return {"ctas_per_cga": (2, 1, 1), "my_new_option": value}
   ```
