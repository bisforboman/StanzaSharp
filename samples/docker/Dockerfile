# A small StanzaSharp app in a Linux x64 container: English text on stdin, CoNLL-U on stdout.
#
#   docker build -t stanzasharp-sample samples/docker
#   docker run -i --rm -v /path/to/models/stanza/en:/models:ro stanzasharp-sample < input.txt
#
# The models (about 600 MB) are mounted read-only at /models, not baked in, so the image stays the same
# when they change. Download them once with the stanzasharp tool, ModelDownloader or `StanzaSharp.Cli download DIR`:
#   dotnet tool install -g StanzaSharp.Tool && stanzasharp download /path/to/models/stanza/en
# To bake them in instead, download in a stage on the SDK image and copy them over (the tool needs no libtorch):
#   RUN dotnet tool install --tool-path /tools StanzaSharp.Tool && /tools/stanzasharp download /models
#   COPY --from=build /models /models
#
# libtorch is built against glibc and needs only glibc, libstdc++ and libgcc_s (it ships its own libgomp),
# so any glibc base image works. Alpine (musl) does not: see "Supported platforms" in the README.
# BASE can be the chiseled image too, which has those three libraries and no shell:
#   docker build --build-arg BASE=mcr.microsoft.com/dotnet/runtime:10.0-noble-chiseled ...

ARG BASE=mcr.microsoft.com/dotnet/runtime:10.0

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG STANZASHARP_VERSION=*
WORKDIR /src
COPY . .
# -r linux-x64 copies only that runtime's native files; the platform package brings no others anyway.
RUN dotnet publish -c Release -r linux-x64 --no-self-contained -p:StanzaSharpVersion=$STANZASHARP_VERSION -o /app

FROM $BASE
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "StanzaSharpDocker.dll", "/models"]
