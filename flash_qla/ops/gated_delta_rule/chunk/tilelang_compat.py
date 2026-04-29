import tilelang.language as T

if not hasattr(T, "gemm_v1"):
    T.gemm_v1 = T.gemm
