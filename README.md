# exo-mlir

An experimental MLIR backend for the Exo programming language.

## Requirements

The compiler itself uses the `uv` package manager to manage dependencies. You can install it by running:

```bash
curl -sSfL https://get.uv.tools | sh
```

To build the benchmarks, you need the following dependencies:

-   `clang`
-   `ccache`
-   `lld`
-   `ninja`

Running `make` will build the required LLVM dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
