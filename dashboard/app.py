"""Plotly Dash dashboard: upload a photo, get a coordinate and a pin on the map.

    python dashboard/app.py            # http://127.0.0.1:8050
    gunicorn -b 0.0.0.0:7860 app:server  # how the Space serves it
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import sys

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inference
from theme import BODY, CREAM, CSS, DEEP, FONTS, MINT, MUTED_TEXT

# Per-protocol accuracy, measured on splits/test.csv by bench/protocol_metrics.
# Each protocol gets its own ring, because they are not equally accurate: the
# repo's test_summary.json reports only the `anchor` rule, and quoting that one
# number for all four would flatter the weakest and libel the strongest.
METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_assets",
    "protocol_metrics.json",
)
try:
    with open(METRICS_PATH) as _f:
        METRICS = json.load(_f)
except OSError:
    METRICS = {}

FALLBACK_KM = 99.0

REPO = "https://github.com/ai-vaibhavv/GeoLocator"

# Matches the caution box's border, so a widened ring and the warning text read
# as the same signal.
CAUTION_LINE = "#B08428"

def headline_km(predictor):
    """The masthead figure for one protocol.

    It tracks the dropdown rather than showing the best protocol's number: a
    fixed 37 km over a caption reading 75 km is an overclaim by 2x.

    Args:
        predictor: the protocol key.

    Returns:
        A display string.
    """
    m = METRICS.get(predictor)
    return f"{m['median_km']:,.0f} km" if m else "—"

# Dash's own placeholders are {%name%}, so the page is assembled with
# str.replace rather than any %- or brace-formatting.
INDEX = """<!DOCTYPE html>
<html lang="en">
  <head>
    {%metas%}
    <title>GeoLocator</title>
    {%favicon%}
    <link rel="icon" href="/assets/favicon.ico" sizes="any">
    <link rel="icon" type="image/png" href="/assets/favicon.png">
    <link rel="apple-touch-icon" href="/assets/apple-touch-icon.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="__FONTS__">
    <style>__CSS__</style>
    {%css%}
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""


def circle(lat, lng, km, n=90):
    """Points of a circle of radius `km` around a coordinate.

    Args:
        lat, lng: centre in degrees.
        km: radius in kilometres.
        n: number of vertices.

    Returns:
        (lats, lons) lists that close back on the first point.
    """
    d = km / 6371.0088
    p, l = math.radians(lat), math.radians(lng)
    lats, lons = [], []
    for i in range(n + 1):
        b = 2 * math.pi * i / n
        p2 = math.asin(math.sin(p) * math.cos(d) + math.cos(p) * math.sin(d) * math.cos(b))
        l2 = l + math.atan2(
            math.sin(b) * math.sin(d) * math.cos(p),
            math.cos(d) - math.sin(p) * math.sin(p2),
        )
        lats.append(math.degrees(p2))
        lons.append((math.degrees(l2) + 540) % 360 - 180)
    return lats, lons


def base_figure(traces, center, zoom, revision):
    """A map figure with our chrome, whatever is drawn on it.

    Args:
        traces: the traces to draw.
        center: (lat, lon) the map opens on.
        zoom: initial zoom level.
        revision: uirevision key. Plotly keeps the user's pan and zoom while
            this is unchanged, so it has to move with the prediction: a
            constant would pin the map at the world view it opened on.

    Returns:
        A plotly Figure.
    """
    fig = go.Figure(traces)
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center[0], lon=center[1]),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
        showlegend=False,
        paper_bgcolor=CREAM,
        font=dict(family=BODY, color=DEEP),
        hoverlabel=dict(bgcolor="#fff", bordercolor=MINT, font=dict(family=BODY)),
        uirevision=revision,
    )
    return fig


def empty_figure():
    """The world, before anything has been predicted.

    The trace is empty but must exist: a `map=` layout with no map trace makes
    Plotly fall back to a cartesian grid, which is not a map at all.
    """
    return base_figure(
        [go.Scattermap(lat=[], lon=[], mode="markers", hoverinfo="skip")],
        (54.0, 15.0),
        2.4,
        "empty",
    )


def median_km(predictor):
    """The measured median error for one protocol, in km.

    Args:
        predictor: the protocol key.

    Returns:
        A float; FALLBACK_KM when the metrics file has not been built.
    """
    return METRICS.get(predictor, {}).get("median_km", FALLBACK_KM)


def accuracy_note(predictor):
    """The caption under the controls, honest about the selected protocol.

    Args:
        predictor: the protocol key.

    Returns:
        A list of children for the hint div.
    """
    m = METRICS.get(predictor)
    if not m:
        return [
            "Accuracy has not been measured for this build. Run ",
            html.B("dashboard/bench.py"),
            " to fill in the per-method numbers.",
        ]
    return [
        "On the held out test set this method puts half of all photos within ",
        html.B(f"{m['median_km']:,.0f} km"),
        f" of the truth, and {m['frac_200km']:.0%} within 200 km. The ring "
        f"around the pin is that median radius, so treat the answer as a "
        f"neighbourhood rather than an address. Half of all photos land outside "
        f"it, and the model is not always able to tell you which half it is in.",
    ]


def map_figure(p):
    """Build the map: the median-error ring, then the pin.

    Args:
        p: an inference.Prediction.

    Returns:
        A plotly Figure.
    """
    radius = median_km(p.predictor)
    shaky = is_low_confidence(p)
    if shaky:
        # A solid median-width ring would be asserting "the truth is in here"
        # exactly where the caution says it probably is not.
        radius *= 2.0
    clat, clon = circle(p.lat, p.lng, radius)
    return base_figure(
        [
            go.Scattermap(
                lat=clat,
                lon=clon,
                mode="lines",
                fill="toself",
                # Scattermap's line supports only colour and width; there is no
                # dash property on a map trace. Low confidence is signalled by
                # the amber of the caution box plus the widened radius.
                fillcolor="rgba(217, 185, 120, 0.16)" if shaky
                else "rgba(102, 163, 191, 0.20)",
                line=dict(color=CAUTION_LINE if shaky else DEEP, width=1.5),
                hoverinfo="skip",
                name=(
                    f"~{radius:.0f} km, widened for low confidence"
                    if shaky
                    else f"~{radius:.0f} km"
                ),
            ),
            go.Scattermap(
                lat=[p.lat],
                lon=[p.lng],
                mode="markers",
                marker=dict(size=17, color=DEEP),
                text=[f"{p.country}<br>{p.lat:.4f}, {p.lng:.4f}"],
                hoverinfo="text",
                name="prediction",
            ),
        ],
        (p.lat, p.lng),
        4.4 if shaky else 5.2,
        f"{p.lat:.4f},{p.lng:.4f}",
    )


def pct(x):
    """Format a probability without ever claiming certainty.

    A model with a ~100 km median error must not print "100%", so the top of
    the range is shown as an inequality instead of being rounded up to it.

    Args:
        x: a probability in [0, 1].

    Returns:
        A display string.
    """
    if x >= 0.995:
        return ">99%"
    if x < 0.01:
        return "<1%"
    return f"{x:.0%}"


def tile(key, value, wide=False, minor=False, word=False):
    """One stat tile.

    Args:
        key: the small uppercase label.
        value: the figure or phrase shown under it.
        wide: span the whole row, for the headline answer.
        minor: render as a diagnostic rather than part of the answer.
        word: the value is a word, so set it in the body face, not mono.

    Returns:
        A Div.
    """
    classes = "tile" + (" wide" if wide else "") + (" minor" if minor else "")
    return html.Div(
        [
            html.Div(key, className="k"),
            html.Div(value, className="v word" if word else "v"),
        ],
        className=classes,
    )


def tiles(p):
    """The numeric side of a prediction.

    Args:
        p: an inference.Prediction.

    Returns:
        A list of tile Divs.
    """
    headline = [f"{p.country} · {pct(p.country_prob)}"]
    if is_low_confidence(p):
        # Without this the banner asserts a country in the same confident blue
        # it uses for a >99% answer, and the amber warning sits four blocks below.
        headline.append(html.Span("low confidence", className="flag"))
    out = [
        tile("Most likely country", headline, wide=True),
        tile("Cell confidence", pct(p.cell_prob), minor=True),
        tile("Fine / coarse gate", f"{p.gate:.2f}", minor=True),
        tile("Method", p.predictor_label, minor=True, word=True),
    ]
    if p.neighbour_km is not None:
        # Only Mitron has a neighbour, and how far it sits from the model's own
        # guess says whether the two approaches agree about this photo.
        out.append(
            tile("Match vs raw guess", f"{p.neighbour_km:,.0f} km", minor=True)
        )
    return out


EMPTY_TILES = [tile("Status", "Waiting for a photo", wide=True)]
REJECTED_TILES = [tile("Status", "That photo could not be read", wide=True)]

# Calibrated, not guessed: dashboard/calibrate.py scored 500 test photos and
# measured which of the model's own signals separate bad answers from good ones.
#
#   country_prob <= 0.81  ->  median error 611 km, against 42 km for the rest
#   cell_prob    <= 0.51  ->  median error 643 km, against 47 km for the rest
#
# The gate was also tested and is NOT a warning signal: it runs the other way,
# with low-gate photos scoring BETTER (35 km vs 93 km), so folding it in would
# have flagged the model's most accurate answers.
LOW_COUNTRY_PROB = 0.81
LOW_CELL_PROB = 0.51

# What being flagged is worth, from the same run, for the caution to quote.
FLAGGED_MEDIAN_KM = 611
UNFLAGGED_MEDIAN_KM = 42


def is_low_confidence(p):
    """Whether the model's own signals say this particular answer is shaky.

    Args:
        p: the current Prediction.

    Returns:
        True when any calibrated signal is below its threshold.
    """
    return (
        p.country_prob < LOW_COUNTRY_PROB
        or p.cell_prob < LOW_CELL_PROB
    )


def caution(p):
    """A per-photo warning when the model's own confidence is low.

    The accuracy caption is a global median; it cannot tell anyone that *this*
    answer is shaky. These two signals can.

    Args:
        p: the current Prediction.

    Returns:
        A Div, or None when the model is reasonably confident.
    """
    if not is_low_confidence(p):
        return None
    reasons = []
    if p.country_prob < LOW_COUNTRY_PROB:
        reasons.append(f"it is only {pct(p.country_prob)} sure of the country")
    if p.cell_prob < LOW_CELL_PROB:
        reasons.append(f"no single region stands out ({pct(p.cell_prob)})")
    if not reasons:
        return None
    return html.Div(
        [
            html.B("Low confidence on this photo. "),
            "The model says " + " and ".join(reasons) + ". On the test set, "
            f"photos it was this unsure about missed by about "
            f"{FLAGGED_MEDIAN_KM} km, against {UNFLAGGED_MEDIAN_KM} km for the "
            "rest, so treat this pin as a rough direction. The ring is widened "
            "to match.",
        ],
        className="caution",
    )


def coords_block(p, previous=None):
    """The coordinate readout, with how far this method moved the pin.

    Args:
        p: the current Prediction.
        previous: the (lat, lng) the previous method gave for the SAME photo,
            or None. Comparing across photos would be meaningless.

    Returns:
        A Div for the prediction card.
    """
    ns = "N" if p.lat >= 0 else "S"
    ew = "E" if p.lng >= 0 else "W"
    # Four decimals is about 11 m. Six would claim 10 cm against a ~100 km
    # median error, which the caption right below would immediately contradict.
    children = [f"{abs(p.lat):.4f}° {ns}, {abs(p.lng):.4f}° {ew}"]
    if previous:
        moved = inference.haversine_km(p.lat, p.lng, previous[0], previous[1])
        if moved >= 0.05:
            children.append(
                html.Span(
                    f"{moved:,.0f} km from the previous method"
                    if moved >= 1
                    else "less than 1 km from the previous method",
                    className="moved",
                )
            )
    return html.Div(children, className="coords")


def first_clause(blurb):
    """The opening sentence of a blurb, for the dropdown's second line.

    Args:
        blurb: the full description.

    Returns:
        A short string.
    """
    head = blurb.split(". ")[0].strip()
    return head if head.endswith(".") else head + "."


def decode(contents):
    """Turn a dcc.Upload data URL into a PIL image.

    Args:
        contents: the "data:image/...;base64,..." string dcc.Upload gives.

    Returns:
        A PIL image in RGB.
    """
    _, payload = contents.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def layout():
    """The whole page."""
    return html.Div(
        className="wrap",
        children=[
            html.Div(
                className="masthead",
                children=[
                    html.Div(
                        [
                            html.H1("GeoLocator"),
                            html.P(
                                "Upload a street level photo taken in Europe. The "
                                "model works out where it was probably taken and "
                                "drops a pin on the map, along with its best guess "
                                "at the country."
                            ),
                        ],
                        className="masthead-copy",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(id="headline-km", className="n"),
                                    html.Div("Median error", className="l"),
                                ],
                                className="masthead-stat",
                            ),
                            html.Div(
                                [
                                    html.Div("4.5 M", className="n"),
                                    html.Div("Parameters", className="l"),
                                ],
                                className="masthead-stat",
                            ),
                        ],
                        className="masthead-stats",
                    ),
                ],
            ),
            html.Main(
                className="grid",
                children=[
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Photo"),
                                    dcc.Upload(
                                        id="upload",
                                        multiple=False,
                                        accept="image/*",
                                        children=html.Div(
                                            [
                                                html.Div(
                                                    "Drop a photo here",
                                                    className="big",
                                                ),
                                                html.Div(
                                                    "or click to browse. JPG or PNG.",
                                                    className="small",
                                                ),
                                            ],
                                            className="drop",
                                        ),
                                    ),
                                    html.Div(id="preview"),
                                    html.Div(
                                        [
                                            html.Div(
                                                "How to locate it",
                                                className="field-label",
                                                id="predictor-label",
                                            ),
                                            # dcc.Dropdown takes no ARIA
                                            # kwargs, so the label is associated
                                            # on a wrapper instead.
                                            html.Div(
                                                dcc.Dropdown(
                                                    id="predictor",
                                                    options=PREDICTOR_OPTIONS,
                                                    value=inference.DEFAULT_PREDICTOR,
                                                    clearable=False,
                                                    searchable=False,
                                                ),
                                                role="group",
                                                **{
                                                    "aria-labelledby":
                                                        "predictor-label"
                                                },
                                            ),
                                            html.Div(id="predictor-note", className="field-note"),
                                        ],
                                        className="field",
                                    ),
                                    html.Button(
                                        "Locate again",
                                        id="locate",
                                        className="btn secondary",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Div(id="btn-note", className="btn-note"),
                                    html.Div(
                                        id="error",
                                        className="err",
                                        role="alert",
                                        **{"aria-live": "polite"},
                                    ),
                                    html.Div(id="accuracy-note", className="hint"),
                                    html.Div(
                                        [
                                            html.B("Europe only. "),
                                            "The model was trained on "
                                            f"{len(COUNTRIES)} countries and has no "
                                            "class for anywhere else, so a photo "
                                            "from outside them will still return "
                                            "one of these, confidently and wrongly.",
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        f"See the {len(COUNTRIES)} "
                                                        "countries"
                                                    ),
                                                    html.Div(
                                                        ", ".join(COUNTRIES) + ".",
                                                        className="coverage-list",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="hint coverage",
                                    ),
                                ],
                                className="card",
                            )
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Loading(
                                    dcc.Graph(
                                        id="map",
                                        figure=empty_figure(),
                                        config={
                                            "displayModeBar": False,
                                            # Wheel-zoom on a page-embedded map
                                            # hijacks page scrolling: the cursor
                                            # passes over the map, the map zooms
                                            # instead, and uirevision then keeps
                                            # that accidental zoom so the pin is
                                            # stranded off-screen. Drag still
                                            # pans; double-click still zooms.
                                            "scrollZoom": False,
                                            "doubleClick": "reset",
                                            "responsive": True,
                                        },
                                        style={
                                            "height": "430px",
                                            "borderRadius": "14px",
                                            "overflow": "hidden",
                                        },
                                    ),
                                    # Dim the old map rather than blanking it:
                                    # inference takes seconds on CPU and a white
                                    # card reads as a crash.
                                    overlay_style={
                                        "visibility": "visible",
                                        "opacity": 0.45,
                                        "filter": "blur(1px)",
                                    },
                                    custom_spinner=html.Div(
                                        "Locating… this takes a few seconds",
                                        className="loading-pill",
                                    ),
                                ),
                                className="card map-shell",
                            ),
                            html.Div(
                                [
                                    html.H2("Prediction"),
                                    html.Div(
                                        EMPTY_TILES, id="tiles", className="tiles"
                                    ),
                                    html.Div(id="coords"),
                                    html.Div(id="caution"),
                                ],
                                className="card",
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Store(id="last"),
            dcc.Store(id="map-nudge"),
            html.Footer(
                [
                    html.Span("GeoLocator"),
                    html.Span("·", className="dot"),
                    html.A(
                        "ai-vaibhavv/GeoLocator",
                        href=REPO,
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                    html.Span("·", className="dot"),
                    html.Span("Map data © OpenStreetMap contributors"),
                ],
                className="footer",
            ),
        ],
    )


def option_desc(key, blurb):
    """The dropdown's second line: what the method does, and how accurate it is.

    The median is what actually separates these four, so it belongs at the point
    of choice rather than only after one has been picked.

    Args:
        key: the protocol key.
        blurb: the method's full description.

    Returns:
        A list of children.
    """
    out = [first_clause(blurb)]
    m = METRICS.get(key)
    if m:
        out += [" ", html.Span(f"{m['median_km']:,.0f} km median", className="km")]
    return out


# The training set is 12 European countries, ~700 photos each. The model has no
# class for anywhere else, so a photo from outside Europe cannot return a
# correct answer, only a confident wrong one. The UI has to say so.
COUNTRIES = [c.replace("_", " ") for c in inference.load()["countries"]]


PREDICTOR_OPTIONS = [
    {
        "label": html.Div(
            [
                html.Div(label, className="opt-name"),
                html.Div(option_desc(key, blurb), className="opt-desc"),
            ]
        ),
        "value": key,
    }
    for key, (label, blurb) in inference.available_predictors().items()
]


app = Dash(__name__, title="GeoLocator", update_title=None)
app.index_string = INDEX.replace("__FONTS__", FONTS).replace("__CSS__", CSS)
app.layout = layout
server = app.server  # what gunicorn serves


# Plotly hands MapLibre the requested centre but anchors it to the TOP of the
# map instead of the middle: measured, the pin projected to y = -215 in a 430px
# canvas, exactly half the viewport too high, and jumpTo put it back at y = 215.
# The error is a fixed pixel offset, so it is invisible near the equator (the
# pin is still on screen) and strands the pin entirely at high latitudes, where
# the same offset covers far more ground. Re-centring the underlying map after
# each figure update frames the pin correctly at any latitude.
app.clientside_callback(
    """
    function (figure) {
        const nu = window.dash_clientside.no_update;
        if (!figure || !figure.data || figure.data.length < 2) { return nu; }
        const pin = figure.data[1];
        if (!pin.lat || !pin.lat.length) { return nu; }
        const zoom = (figure.layout && figure.layout.map && figure.layout.map.zoom) || 5.2;
        const target = [pin.lon[0], pin.lat[0]];
        let tries = 0;
        const nudge = function () {
            // #map is Dash's wrapper; the Plotly graph is the child that
            // actually carries _fullLayout.
            const gd = document.querySelector('#map .js-plotly-plot');
            const sub = gd && gd._fullLayout && gd._fullLayout.map
                && gd._fullLayout.map._subplot;
            const drawn = gd && gd.data && gd.data.length > 1;
            if (sub && sub.map && drawn) {
                sub.map.jumpTo({center: target, zoom: zoom});
                return;
            }
            if (tries++ < 40) { setTimeout(nudge, 150); }
        };
        setTimeout(nudge, 200);
        return nu;
    }
    """,
    Output("map-nudge", "data"),
    Input("map", "figure"),
)


@app.callback(
    Output("preview", "children"),
    Output("locate", "disabled"),
    Output("btn-note", "children"),
    Input("upload", "contents"),
)
def show_preview(contents):
    """Echo a readable upload back as a thumbnail and enable the re-run button.

    A file the decoder rejects gets no preview at all: the browser's broken
    image glyph sitting above "this file is unreadable" is just noise.
    """
    if not contents:
        return None, True, "Upload a photo to enable this."
    try:
        decode(contents)
    except Exception:
        # The error banner already explains what went wrong; repeating "upload a
        # photo" at someone who just uploaded one only contradicts it.
        return None, True, ""
    return (
        html.Img(src=contents, className="preview", alt="The photo you uploaded"),
        False,
        "",
    )


@app.callback(
    Output("predictor-note", "children"),
    Output("accuracy-note", "children"),
    Output("headline-km", "children"),
    Input("predictor", "value"),
)
def describe_predictor(predictor):
    """The selected rule's blurb and its measured accuracy.

    Args:
        predictor: the protocol key.

    Returns:
        (blurb children, accuracy caption children, masthead figure).
    """
    if not predictor:
        return "", "", "—"
    blurb = [
        inference.PREDICTORS[predictor][1],
        " Protocol: ",
        html.B(predictor),
        ".",
    ]
    return blurb, accuracy_note(predictor), headline_km(predictor)


@app.callback(
    Output("map", "figure"),
    Output("tiles", "children"),
    Output("tiles", "className"),
    Output("coords", "children"),
    Output("caution", "children"),
    Output("error", "children"),
    Output("last", "data"),
    Input("upload", "contents"),
    Input("locate", "n_clicks"),
    Input("predictor", "value"),
    State("upload", "contents"),
    State("last", "data"),
    prevent_initial_call=True,
)
def locate(contents, n_clicks, predictor, held, last):
    """Run the model on whatever is currently uploaded.

    Fires on a new upload, on the button, and whenever the rule changes, so the
    first result arrives without a click and switching rules re-scores the same
    photo in place.
    """
    contents = contents or held
    if not contents:
        return empty_figure(), EMPTY_TILES, "tiles", None, None, None, None
    # A new photo resets the comparison: "1,702 km from the previous method" is
    # a lie when what actually changed was the photo.
    digest = hashlib.sha1(contents.encode()).hexdigest()[:16]
    same_photo = bool(last) and last.get("digest") == digest
    previous = last.get("latlng") if same_photo and last.get("predictor") != predictor else None
    try:
        p = inference.locate(decode(contents), predictor=predictor)
    except Exception as exc:  # a corrupt upload should not take the page down
        # The traceback goes to the log, never to the page: a user cannot act on
        # a Python repr, and a memory address on screen is just alarming.
        print(f"upload failed: {exc!r}", file=sys.stderr, flush=True)
        return (
            empty_figure(),
            REJECTED_TILES,
            "tiles",
            None,
            None,
            "That file is not a readable JPG or PNG. Try another photo.",
            None,
        )

    return (
        map_figure(p),
        tiles(p),
        # Re-adding the class restarts the flash, so a method change is visible
        # even when it only moves the pin a few km.
        f"tiles updated {predictor}",
        coords_block(p, previous),
        caution(p),
        None,
        {"digest": digest, "latlng": [p.lat, p.lng], "predictor": predictor},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8050)))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    inference.load()  # fail fast on a missing checkpoint, and warm the weights
    app.run(host=a.host, port=a.port, debug=a.debug)
