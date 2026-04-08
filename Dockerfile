# Dockerfile for reproducing smelt-ml paper benchmarks.
#
# Build:  docker build -t smelt-replication .
# Run:    docker run --rm smelt-replication
# Shell:  docker run --rm -it smelt-replication bash
#
# Reproduces all benchmark tables and figures from the paper.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (stable)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Python ML libraries (pinned versions matching paper)
RUN python3 -m pip install --break-system-packages \
    numpy scipy matplotlib \
    scikit-learn==1.8.0 \
    xgboost==3.1.1 \
    lightgbm==4.6.0 \
    catboost==1.2.8

# Copy project
WORKDIR /smelt
COPY . /smelt/

# Build Rust benchmarks (release + LTO)
RUN RUSTFLAGS="-C target-cpu=native" cargo build --release \
    --example benchmark_large \
    --example benchmark_prediction \
    --example ablation_study \
    --example accuracy_validation \
    --example case_study_king_county

# Verify tests pass
RUN cargo test --lib --tests 2>&1 | tail -3

# Default: run the full replication
CMD ["bash", "-c", "\
    echo '=== smelt-ml Paper Replication ===' && \
    echo '' && \
    echo '1. Rust benchmarks...' && \
    ./target/release/examples/benchmark_large && \
    echo '' && \
    echo '2. C++ benchmarks...' && \
    python3 paper/replication/benchmark_cpp.py && \
    echo '' && \
    echo '3. Comparison tables...' && \
    python3 paper/replication/compare_results.py && \
    echo '' && \
    echo '4. Statistical analysis...' && \
    python3 paper/replication/statistical_analysis.py && \
    echo '' && \
    echo '5. Prediction benchmark...' && \
    ./target/release/examples/benchmark_prediction && \
    echo '' && \
    echo '6. Accuracy validation...' && \
    ./target/release/examples/accuracy_validation && \
    echo '' && \
    echo '7. Case study (King County)...' && \
    ./target/release/examples/case_study_king_county && \
    echo '' && \
    echo '=== Replication complete ===' \
"]
