import platform
import subprocess

configfile: "config.yaml"

rule all:
    input:
        "bin/benchmark"

rule clean:
    shell:
        "rm -rf build submodules/benchmark/build"

rule env:
    shell:
        """
        uv venv
        uv sync --all-extras
        source .venv/bin/activate
        """

rule submodules:
    output:
        directory("submodules")
    shell:
        "git submodule update --init --recursive"

rule compile_submodule_benchmark:
    input:
        "submodules/benchmark"
    output:
        directory("submodules/benchmark/build")
    shell:
        """
        cmake -S submodules/benchmark -B submodules/benchmark/build -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF
        cmake --build submodules/benchmark/build --config Release
        """

rule mkdirs:
    shell:
        f"mkdir -p build/benchmarks/bin"
    output:
        directory("build/benchmarks")
        directory("bin")

rule compile_exo_all_procedures:
    input:
        "benchmarks/{proc}/{stem}.py"
    output:
        "build/benchmarks/{proc}/{stem}.c"
        "build/benchmarks/{proc}/{stem}.h"
    shell:
        """
        mkdir -p build/benchmarks/{wildcards.proc}
        exocc -o build//benchmarks/{wildcards.proc} --stem {wildcards.stem} benchmarks/{wildcards.proc}/{wildcards.stem}.py
        """

rule compile_cc_ext_procedures:
    input:
        "benchmarks/{proc}/{stem}-{ext}.c"
        "benchmarks/{proc}/{stem}-{ext}.h"
    output:
        "build/benchmarks/{proc}/{stem}-{ext}.o"
    params:
        cc: config["cc"]
        cflags: config["cflags"]
        ext_flags: config["ext_flags"][wildcards.ext]
    shell:
        """
        {params.cc} {params.cflags} {params.ext_flags} -c -o build/benchmarks/{wildcards.proc}/{wildcards.stem}-{wildcards.ext}.o benchmarks/{wildcards.proc}/{wildcards.stem}-{wildcards.arch}.c
        """

def supports_ext(ext: str):
    return subprocess.run(["grep", "/proc/cpuinfo", ext], stdout=subprocess.PIPE).returncode == 0


# rule to build all compatible procedures 
rule compile_cc_all_procedures:
