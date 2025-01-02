import platform
import subprocess

configfile: "config.yaml"

def supports_variant(variant):
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

rule exocc_compile:
    input:
        "benchmarks/{kernel}/{variant}.py"
    output:
        "build/benchmarks/{kernel}/{variant}.c",
        "build/benchmarks/{kernel}/{variant}.h"
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    shell:
        "exocc -o build/benchmarks/{wildcards.kernel} --stem {wildcards.variant} {input}"

rule cc_compile:
    input:
        c="build/benchmarks/{kernel}/{variant}.c",
        h="build/benchmarks/{kernel}/{variant}.h"
    output:
        "build/benchmarks/{kernel}/{variant}.S"
     # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        include=lambda wildcards: "build/benchmarks/{wildcards.kernel}",
        cflags=config["cflags"],
        vflags=lambda wildcards: config["vflags"][wildcards.variant]
    shell:
        "{params.cc} -I{params.include} {params.cflags} {params.vflags} -S -o {output} {input.c}"

rule cc_assemble:
    input:
        "build/benchmarks/{kernel}/{variant}.S"
    output:
        "build/benchmarks/{kernel}/{variant}.o"
    # exclude main
    wildcard_constraints:
        variant="|".join(EXO_VARIANTS)
    params:
        cc=config["cc"],
        asflags=config["asflags"]
    shell:
        "{params.cc} {params.asflags} -c -o {output} {input}"

rule cc_compile_benchmark:
    input:
        c="benchmarks/benchmark.c",
        h="benchmarks/benchmark.h"
    output:
        "build/benchmarks/benchmark.o"
    params:
        cc=config["cc"],
        cflags=config["cflags"],
        ldflags=config["ldflags"]
    shell:
        "{params.cc} -Ibenchmarks {params.cflags} -c -o {output} {input.c} {params.ldflags}"

rule cc_compile_main:
    input:
        "benchmarks/{kernel}/main.c",
    output:
        "build/benchmarks/{kernel}/main.o"
    params:
        cc=config["cc"],
        cflags=config["cflags"]
    shell:
        "{params.cc} -Ibenchmarks -I$(dirname {input})  {params.cflags} -c -o {output} {input}"

rule cc_link:
    input:
        "build/benchmarks/{kernel}/{variant}.o",
        "build/benchmarks/{kernel}/main.o",
        "build/benchmarks/benchmark.o"
    output:
        "build/benchmarks/{kernel}/{variant}.x"
    params:
        cc=config["cc"]
    shell:
        "{params.cc} -o {output} {input}"

rule benchmark:
    input:
        "build/benchmarks/{kernel}/{variant}.x"
    output:
        "build/benchmarks/{kernel}/{variant}.csv"
    shell:
        "{input} > {output}"

rule all:
    input:
        expand("build/benchmarks/{kernel}/{variant}.csv", kernel=KERNELS, variant=SUPPORTED_EXO_VARIANTS)
