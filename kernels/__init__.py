"""
kernels - custom GPU kernels. Each operation has its own folder with a plain-PyTorch
reference (the ground truth tests compare against) and the optimized version:

    fused_moe/   all experts of a Mixture-of-Experts layer in two Triton kernels
"""