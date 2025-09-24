import timeit

import jax
import jax.numpy as jnp
import jax.random as random
from flax import nnx
from graphviz import Digraph

import src

master_key = random.key(0)

generator_key, rand_gen_key = random.split(master_key)

# Tensor shape is (B, H, W, C)
input = random.normal(rand_gen_key, (32, 256, 256, 1))
gen = src.Generator(key=generator_key, in_features=1, out_features=1, len_condition_params=10)
gen_jitted = jax.jit(gen)
# jaxpr = jax.make_jaxpr(gen)(input, random.normal(rand_gen_key, (1, 10)))


# def jaxpr_to_dot(jaxpr):
#     dot = Digraph()
#     for eqn in jaxpr.eqns:
#         dot.node(str(id(eqn)), eqn.primitive.name)
#         for var in eqn.invars:
#             dot.edge(str(var), str(id(eqn)))
#         for var in eqn.outvars:
#             dot.edge(str(id(eqn)), str(var))
#     return dot


# dot = jaxpr_to_dot(jaxpr)
# dot.render("model_graph", format="png")
output = gen_jitted(input, random.normal(rand_gen_key, (1, 10)))

output = timeit.timeit(lambda: gen_jitted(input, random.normal(rand_gen_key, (1, 10))), number=10)
output2 = timeit.timeit(lambda: gen(input, random.normal(rand_gen_key, (1, 10))), number=10)

print("Jitted")
print(output)
print("Normal")
print(output2)

print(f"Output of Gen is {output}")
print(f"Shape of Output is {output.shape}")
