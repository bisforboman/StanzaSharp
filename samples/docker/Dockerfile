# A small StanzaSharp app in a Linux container: English text on stdin, CoNLL-U on stdout.
#
#   docker build -t stanzasharp-sample samples/docker
#   docker run -i --rm -v /path/to/models/stanza/en:/models:ro stanzasharp-sample < input.txt
#
# The models (about 600 MB) are mounted read-only at /models, not baked in, so the image stays the same
# when they change. Download them once with the stanzasharp tool, ModelDownloader or `StanzaSharp.Cli download DIR`:
#   dotnet tool install -g StanzaSharp.Tool && stanzasharp download /path/to/models/stanza/en
# To bake them in instead, download in a stage on the SDK image and copy them over:
#   RUN dotnet tool install --tool-path /tools StanzaSharp.Tool && /tools/stanzasharp download /models
#   COPY --from=build /models /models
#
# StanzaSharp's default backend is plain .NET code: the image carries no libtorch and needs nothing but the .NET
# runtime, so any runtime image works. BASE can be the chiseled image (no shell) or Alpine (musl; set RID too):
#   docker build --build-arg BASE=mcr.microsoft.com/dotnet/runtime:10.0-noble-chiseled ...
#   docker build --build-arg BASE=mcr.microsoft.com/dotnet/runtime:10.0-alpine --build-arg RID=linux-musl-x64 ...

ARG BASE=mcr.microsoft.com/dotnet/runtime:10.0

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
ARG STANZASHARP_VERSION=*
ARG RID=linux-x64
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -r $RID --no-self-contained -p:StanzaSharpVersion=$STANZASHARP_VERSION -o /app

FROM $BASE
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "StanzaSharpDocker.dll", "/models"]
