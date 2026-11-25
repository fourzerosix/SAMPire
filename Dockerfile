# syntax=docker/dockerfile:1
# SAMPire image (amd64, arm64), non-root.
#
#   docker build -t sampire .
#   docker run -it --rm sampire                                   # built-in demo, silent, no audio device needed
#   docker run -it --rm -v "$PWD:/work" -w /work sampire --render loop.wav --out /work/out.wav   # headless pipeline
#   docker run -it --rm -v "$XDG_RUNTIME_DIR/pulse:/run/pulse" -e PULSE_SERVER=unix:/run/pulse/native sampire --demo
#
# Real-time audio through a container depends on the host audio server; run natively for the lowest latency.

FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml setup.py setup.cfg README.md LICENSE MANIFEST.in ./
COPY sampire ./sampire
RUN pip install --no-cache-dir --upgrade pip build \
 && python -m build --wheel --outdir /dist

FROM python:3.12-slim
LABEL org.opencontainers.image.title="SAMPire" \
      org.opencontainers.image.description="Terminal sampler, chopper and drum machine with an autonomous creative engine" \
      org.opencontainers.image.source="https://github.com/fourzerosix/SAMPire" \
      org.opencontainers.image.licenses="GPL-3.0-only"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libportaudio2 libsndfile1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 sampire

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir "$(ls /tmp/*.whl)[audio]" && rm -f /tmp/*.whl

USER sampire
ENV LANG=C.UTF-8 TERM=xterm-256color
ENTRYPOINT ["sampire"]
CMD ["--demo", "--no-output"]
