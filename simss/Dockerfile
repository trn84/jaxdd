# SimSS 5.36 (SIMsalabim, https://github.com/kostergroup/SIMsalabim, LGPL-3.0), the reference
# jaxdd is validated against, built as two binaries:
#
#   simss              the release, unchanged
#   simss_fixed_ions   the release with fixed_ions.patch: with `fixIons`, Poisson's equation
#                      leaves the frozen ions alone (see README.md here)
#
#   docker build -t jaxdd/simss:5.36 simss
#   docker run --rm -v "$PWD/run:/work" jaxdd/simss:5.36 simulation_setup.txt
#   docker run --rm -v "$PWD/run:/work" --entrypoint simss_fixed_ions jaxdd/simss:5.36 simulation_setup.txt
#
# The tag is pinned: a different SIMsalabim version is a different reference.

FROM debian:bookworm-slim AS build
ARG SIMSALABIM_TAG=v5.36
RUN apt-get update \
 && apt-get install -y --no-install-recommends fpc git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "$SIMSALABIM_TAG" \
        https://github.com/kostergroup/SIMsalabim.git /opt/SIMsalabim
WORKDIR /opt/SIMsalabim/SimSS
RUN fpc -O3 -Fu../Units simss.pas
COPY fixed_ions.patch /opt/fixed_ions.patch
WORKDIR /opt
RUN cp -r SIMsalabim SIMsalabim-fixed-ions \
 && git -C SIMsalabim-fixed-ions apply /opt/fixed_ions.patch \
 && cd SIMsalabim-fixed-ions/SimSS \
 && fpc -O3 -Fu../Units simss.pas

FROM debian:bookworm-slim
ARG SIMSALABIM_TAG=v5.36
LABEL org.opencontainers.image.source="https://github.com/kostergroup/SIMsalabim" \
      org.opencontainers.image.version="$SIMSALABIM_TAG" \
      org.opencontainers.image.licenses="LGPL-3.0-or-later"
COPY --from=build /opt/SIMsalabim/SimSS/simss /usr/local/bin/simss
COPY --from=build /opt/SIMsalabim-fixed-ions/SimSS/simss /usr/local/bin/simss_fixed_ions
COPY --from=build /opt/SIMsalabim/COPYING.txt /opt/SIMsalabim/COPYING.LESSER.txt /usr/share/doc/simsalabim/
WORKDIR /work
ENTRYPOINT ["simss"]
