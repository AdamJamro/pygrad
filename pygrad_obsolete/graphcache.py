# TODO
""" "
A lazy caching mechanism for computation graphs, especially useful for working with higher order derivatives
"""


class GraphCache:
    def __init__(self):
        self.compiled_cache = {}  # Our main cache: {key: compiled_graph_function}

    def execute_forward(self, model, *inputs, **kwargs):
        # 1. Derive the key from the *current* state. No global flags.
        structural_hash = self.compute_structural_hash(model)
        execution_context = (
            model.training,
            tuple(t.shape for t in inputs),
            tuple(t.dtype for t in inputs),
            # ... add device, kwargs, etc.
        )
        cache_key = (structural_hash, execution_context)

        # 2. Check the cache (The "Lazy" part)
        if cache_key in self.compiled_cache:
            # CACHE HIT
            compiled_fn = self.compiled_cache[cache_key]
            return compiled_fn(*inputs, **kwargs)
        else:
            # CACHE MISS: Recompute (compile) the graph
            print("GraphCache: Recomputing graph...")

            # We "trace" or "compile" the model's forward method
            # in its *current* state (train or eval).
            compiled_fn = self.jit_compile(model.forward, execution_context)

            # Store the newly compiled graph
            self.compiled_cache[cache_key] = compiled_fn

            return compiled_fn(*inputs, **kwargs)

    def jit_compile(self, fn, context):
        pass

    def compute_structural_hash(self, model) -> int:
        # Hashes module hierarchy, static params, and fwd method bytecode
        return hash(42)
