# Mini-Inference

LLM serving and inferencing is a big challenge. In this demo project , we will build a mini inference engine project stage by stage adding popular inference features. By the end of this project you will learn how to add handle inference workloads, know how to built featues that might help with contributing to popular inference engines like vLLM , SGLang etc , understand why CUDA/Triton kernels are needed and how they improve performance. 

# Motivation

Why do we need vLLM or SGLang ? The conventional LLM models are not built for inference.When heavy workloads come in often it results in a lot of GPU capabilties being idle.

# The Setup

First we set up a very naive model from huggingface and profile them on concurrent queries. We use a FastAPI based server to interact.

# Evaluation Bench 



# Quantization

# KV cache 

KV Caching is a saves a lot of memory . But it adds to the VRAM for caching it. Let us analyze how KV Caching influences the performance of the .Try reading my write up on KV Caching to get a better insight.

# Paged KV Caching

# Continous batching

# Chunked Prefill

# Speculative Decoding

# MoE



