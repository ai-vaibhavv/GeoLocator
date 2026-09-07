---
title: GeoLocator
emoji: 🗺️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GeoLocator

Upload a street level photo taken in Europe. The model works out where it was
probably taken and drops a pin on the map.

**Scope: 12 European countries** — Belarus, Finland, France, Germany, Iceland,
Italy, Norway, Poland, Spain, Sweden, Turkey and the United Kingdom, about 700
training photos each. The model has no class for anywhere else, so a photo from
outside these will still be assigned one of them.

Under the hood there is a HGNetV2-B0 backbone (4.5 M parameters) that predicts
over two k-means geo-cell partitions at the same time, 64 coarse cells and 256
fine ones, blended per image by a learned gate, plus a head that nudges the
result off the cell centre. At inference the prediction is snapped to the most
likely country and re-anchored on that country's three best cells. The
dashboard lets you switch between that and the other methods, including a
lookalike search that matches your photo against the training set. Each one is
labelled with both a nickname and the name it goes by in the report.

On the held out test set, half of all photos land within **99 km** of the truth,
62% within 200 km, and 87% within 750 km. The ring drawn around the pin is that
99 km radius, so read the answer as a neighbourhood rather than an address.

Built with Plotly Dash. It runs on CPU and a single photo takes well under a
second.
