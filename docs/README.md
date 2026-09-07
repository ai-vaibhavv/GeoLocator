# GitHub Pages landing page

A static page that embeds the Hugging Face Space, so the URL you share is your
own and loads instantly even while the Space is waking up.

## Enable it

Repository **Settings → Pages → Source: Deploy from a branch → `main` / `/docs`**.
The page then serves at `https://ai-vaibhavv.github.io/GeoLocator/`.

## Point it at your Space

`index.html` embeds `https://ai-vaibhavv-geolocator.hf.space`, which is the
direct app URL for the Space `ai-vaibhavv/GeoLocator`. The pattern is
`https://<user>-<space>.hf.space`, lowercased with `/` replaced by `-`. Change
the `iframe src` and the two links if your Space is named differently.

## Keeping the numbers honest

The method table and the 611 km / 42 km figures are copied from
`dashboard/model_assets/protocol_metrics.json` and `calibration.json`. If you
retrain, re-run `dashboard/bench.py` and `dashboard/calibrate.py` and update
them here too, or the page will quietly misreport the model.

`.nojekyll` stops GitHub Pages running Jekyll over these files.
