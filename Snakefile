import platform
import subprocess

configfile: "config.yaml"

def supports_variant(variant):
    if platform.system() == "Darwin":
        return any(line.endswith(": 1") for line in subprocess.run(["sysctl", "-a"], stdout=subprocess.PIPE).stdout.decode().splitlines() if variant in line) or variant == "base"
    else:
        return subprocess.run(["grep", variant, "/proc/cpuinfo"], stdout=subprocess.PIPE).returncode == 0 or variant == "base"

KERNELS = [
    "matmul"
]

EXO_VARIANTS = [
    "base", # baseline naive implementations
    "avx2", # x86 AVX2
    "neon" # ARM neon
]

SUPPORTED_EXO_VARIANTS = [
    variant for variant in EXO_VARIANTS if supports_variant(variant)
]

print("Supported variants: ", SUPPORTED_EXO_VARIANTS)

rule cc_assemble:
    input:
        "build/benchmarks/{compiler}/{kernel}/{variant}.S"
    output:
        "build/benchmarks/{compiler}/{kernel}/{variant}.o"
    # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        asflags=config["asflags"]
    shell:
        "{params.cc} {params.asflags} -c -o {output} {input}"

rule benchmark_compile:
    input:
        "benchmarks/benchmark.c",
    output:
        "build/benchmarks/benchmark.o"
    params:
        cc=config["cc"],
        cflags=config["cflags"]
    shell:
        "{params.cc} -Ibenchmarks {params.cflags} -c -o {output} {input}"

rule exocc_compile:
    input:
        "benchmarks/{kernel}/{variant}.py"
    output:
        "build/benchmarks/exocc/{kernel}/{variant}.c",
        "build/benchmarks/exocc/{kernel}/{variant}.h"
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    shell:
        "exocc -o build/benchmarks/exocc/{wildcards.kernel} --stem {wildcards.variant} {input}"

rule exocc_cc_compile:
    input:
        c="build/benchmarks/exocc/{kernel}/{variant}.c",
        h="build/benchmarks/exocc/{kernel}/{variant}.h"
    output:
        "build/benchmarks/exocc/{kernel}/{variant}.S"
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        include=lambda wildcards: "build/benchmarks/exocc/{wildcards.kernel}",
        cflags=config["cflags"],
        vflags=lambda wildcards: config["vflags"][wildcards.variant]
    shell:
        "{params.cc} -I{params.include} {params.cflags} {params.vflags} -S -o {output} {input.c}"


rule exocc_cc_compile_main:
    input:
        "benchmarks/{kernel}/exocc.c",
        "build/benchmarks/benchmark.o",
        "build/benchmarks/exocc/{kernel}/{variant}.o"
    output:
        "build/benchmarks/exocc/{kernel}/{variant}.x"
    # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        cflags=config["cflags"],
        ldflags=config["ldflags"]
    shell:
        "{params.cc} -Ibenchmarks -I$(dirname {output}) {params.cflags} -o {output} {input} {params.ldflags} -D__TARGET_{wildcards.variant}"

rule exocc_benchmark:
    input:
        "build/benchmarks/exocc/{kernel}/{variant}.x"
    output:
        "build/benchmarks/exocc/{kernel}/{variant}.csv"
    shell:
        "{input} > {output}"

rule exomlir_compile:
    input:
        "benchmarks/{kernel}/{variant}.py"
    output:
        "build/benchmarks/exomlir/{kernel}/{variant}.mlir",
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    shell:
        "exo-mlir -o build/benchmarks/exomlir/{wildcards.kernel} {input}"

rule exomlir_lower_mlir:
    input:
        "build/benchmarks/exomlir/{kernel}/{variant}.mlir"
    output:
        "build/benchmarks/exomlir/{kernel}/{variant}.mlir.lowered"
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    shell:
        "mlir-opt {input} -convert-scf-to-cf -canonicalize -cse | xdsl-opt -p convert-memref-to-ptr{{lower_func=true}},convert-ptr-to-llvm | mlir-opt -convert-func-to-llvm -convert-arith-to-llvm -convert-index-to-llvm > {output}"

rule exomlir_lower_llvmir:
    input:
        "build/benchmarks/exomlir/{kernel}/{variant}.mlir.lowered",
    output:
        "build/benchmarks/exomlir/{kernel}/{variant}.ll",
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    shell:
        "mlir-translate {input} -mlir-to-llvmir > {output}"

rule exomlir_assemble:
    input:
        "build/benchmarks/exomlir/{kernel}/{variant}.ll"
    output:
        "build/benchmarks/exomlir/{kernel}/{variant}.o"
    # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        asflags=config["asflags"]
    shell:
        "{params.cc} {params.asflags} -c -o {output} {input}"

rule exomlir_cc_compile_main:
    input:
        "benchmarks/{kernel}/exomlir.c",
        "build/benchmarks/benchmark.o",
        "build/benchmarks/exomlir/{kernel}/{variant}.o"
    output:
        "build/benchmarks/exomlir/{kernel}/{variant}.x"
    # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        cflags=config["cflags"],
        ldflags=config["ldflags"]
    shell:
        "{params.cc} -Ibenchmarks -I$(dirname {output}) {params.cflags} -o {output} {input} {params.ldflags} -D__TARGET_{wildcards.variant}"

rule exomlir_benchmark:
    input:
        "build/benchmarks/exomlir/{kernel}/{variant}.x"
    output:
        "build/benchmarks/exomlir/{kernel}/{variant}.csv"
    shell:
        "{input} > {output}"

rule exocc:
    input:
        expand("build/benchmarks/exocc/{kernel}/{variant}.csv", kernel=KERNELS, variant=SUPPORTED_EXO_VARIANTS)

rule exomlir:
    input:
        expand("build/benchmarks/exomlir/{kernel}/{variant}.csv", kernel=KERNELS, variant=SUPPORTED_EXO_VARIANTS)

rule all:
    input:
        expand("build/benchmarks/exocc/{kernel}/{variant}.csv", kernel=KERNELS, variant=SUPPORTED_EXO_VARIANTS),
        expand("build/benchmarks/exomlir/{kernel}/{variant}.csv", kernel=KERNELS, variant=SUPPORTED_EXO_VARIANTS)
