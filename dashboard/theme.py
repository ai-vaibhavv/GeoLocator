"""Palette, fonts and the CSS the Dash page is served with.

Palette: colorhunt.co/palette/3368a066a3bfc8dfdbf2efe7
Type: Josefin Sans for display, Inter for UI text, IBM Plex Mono for figures.

Contrast note: MUTED and MINT are light enough that they fail WCAG AA on the
cream and mint grounds, so the darker tints below exist purely to carry text.
Text over the masthead is checked against GRADIENT_END, the lightest pixel it
can sit on, not against DEEP: a gradient's dark end proves nothing about its
light end.
"""

DEEP = "#3368A0"
MID = "#66A3BF"
MINT = "#C8DFDB"
CREAM = "#F2EFE7"
INK = "#1B3348"
MUTED = "#5B7186"

# Text-safe tints, used only where the base palette colour fails AA.
MUTED_TEXT = "#4A5F73"  # ~5.5:1 on CREAM
DEEP_TEXT = "#255183"  # ~4.7:1 on MINT
MINT_TEXT = "#E4F0EE"  # ~4.9:1 on DEEP
# Measured, not guessed: 3.32:1 on cream, 3.81:1 on white, 3.50:1 on the tile
# fill. The obvious mint tints all land near 2:1, which is why this is so dark.
BORDER = "#5F8B84"
GRADIENT_END = "#4177A8"  # stops short of MID so masthead text clears AA

FONTS = (
    "https://fonts.googleapis.com/css2"
    "?family=Josefin+Sans:wght@400;500;600;700"
    "&family=Inter:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@500;600"
    "&display=swap"
)

DISPLAY = "'Josefin Sans', 'Trebuchet MS', sans-serif"
BODY = "'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace"

CSS = f"""
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    background: {CREAM};
    color: {INK};
    font-family: {BODY};
    font-size: 15px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 26px 22px 40px; }}

.masthead {{
    background: linear-gradient(118deg, {DEEP} 0%, {GRADIENT_END} 100%);
    border-radius: 20px;
    padding: 30px 34px;
    color: #fff;
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: center;
    justify-content: space-between;
}}
.masthead-copy {{ max-width: 60ch; }}
.masthead h1 {{
    font-family: {DISPLAY};
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: 0.02em;
    margin: 0;
    line-height: 1;
}}
.masthead p {{ margin: 10px 0 0; font-size: 0.96rem; line-height: 1.45; }}
.masthead-stats {{ display: flex; gap: 26px; }}
.masthead-stat .n {{
    font-family: {DISPLAY};
    font-weight: 700;
    font-size: 1.9rem;
    line-height: 1;
}}
.masthead-stat .l {{
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-top: 6px;
    color: #fff;
}}

.grid {{
    display: grid;
    grid-template-columns: minmax(320px, 5fr) 7fr;
    gap: 18px;
    margin-top: 18px;
    align-items: start;
}}
@media (max-width: 1020px) {{
    .grid {{ grid-template-columns: 1fr; }}
    .masthead-stats {{ gap: 20px; }}
}}

.card {{
    background: #fff;
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 18px;
}}
.card + .card {{ margin-top: 16px; }}
.card h2 {{
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 1.02rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {DEEP};
    margin: 0 0 12px;
}}

.drop {{
    border: 2px dashed {DEEP};
    border-radius: 14px;
    background: {CREAM};
    padding: 30px 16px;
    text-align: center;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
}}
.drop:hover {{ background: {MINT}; }}
#upload:focus-within .drop {{
    outline: 2px solid {DEEP};
    outline-offset: 2px;
    background: {MINT};
}}
.drop .big {{
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 1.25rem;
    color: {DEEP};
}}
.drop .small {{ color: {MUTED_TEXT}; font-size: 0.85rem; margin-top: 4px; }}

.preview {{
    width: 100%;
    max-height: 230px;
    object-fit: cover;
    border-radius: 12px;
    margin-top: 14px;
    border: 1px solid {BORDER};
}}

.btn {{
    width: 100%;
    margin-top: 14px;
    padding: 13px 16px;
    border: none;
    border-radius: 12px;
    background: {DEEP};
    color: #fff;
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background 0.15s;
}}
.btn:hover {{ background: {MID}; }}
.btn:focus-visible {{ outline: 2px solid {INK}; outline-offset: 2px; }}
/* No opacity here: fading the fill also fades the label out of legibility. */
.btn:disabled {{
    background: #E3E9EE;
    color: #55697D;
    cursor: not-allowed;
}}
.btn:disabled:hover {{ background: #E3E9EE; }}
.btn.secondary {{
    background: transparent;
    color: {DEEP};
    border: 1px solid {DEEP};
    font-size: 0.9rem;
    padding: 10px 16px;
}}
.btn.secondary:hover {{ background: {MINT}; }}
.btn.secondary:disabled {{
    background: transparent;
    border-color: #7E93A6;  /* 3.18:1 on white */
    color: #55697D;  /* 4.63:1 */
}}
.btn-note {{
    color: {MUTED_TEXT};
    font-size: 0.78rem;
    text-align: center;
    margin-top: 6px;
}}

.tiles {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}}
@media (min-width: 1100px) {{
    .tiles {{ grid-template-columns: repeat(4, 1fr); }}
}}
/* A last row with one tile left over stretches it rather than leaving a void.
   The wide answer tile is child 1, so a stranded minor tile is an EVEN child in
   the 2-column layout. Counting from the wrong parity fires on exactly the
   cases that were already fine. */
.tiles .tile.minor:last-child:nth-child(even) {{ grid-column: 1 / -1; }}
@media (min-width: 1100px) {{
    .tiles .tile.minor:last-child:nth-child(even) {{ grid-column: auto; }}
    /* 4 columns, wide tile spans the row: 3 minors leave one empty column, so
       the last of three stretches across the gap. */
    .tiles .tile.minor:last-child:nth-child(4) {{ grid-column: span 2; }}
}}
.tile {{ background: {MINT}; border-radius: 14px; padding: 13px 15px; }}
.tile .k {{
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 0.74rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {DEEP_TEXT};
}}
.tile .v {{
    font-family: {MONO};
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 3px;
    line-height: 1.3;
}}
/* Proper nouns are words, not figures; mono is reserved for the numbers. */
.tile .v.word {{ font-family: {BODY}; font-weight: 600; }}
.tile.wide {{ grid-column: 1 / -1; background: {DEEP}; }}
.tile.wide .k {{ color: {MINT_TEXT}; }}
.tile.wide .v {{
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 1.6rem;
    color: #fff;
    letter-spacing: 0.01em;
}}
/* Diagnostics, not the answer: same grid, quieter surface. */
.tile.minor {{ background: #F7F5F0; border: 1px solid {BORDER}; }}
.tile.minor .k {{ color: {MUTED_TEXT}; }}
.tile.minor .v {{ font-size: 1rem; color: {MUTED_TEXT}; }}

@keyframes tile-update {{
    0% {{ box-shadow: 0 0 0 3px rgba(102, 163, 191, 0.55); }}
    100% {{ box-shadow: 0 0 0 3px rgba(102, 163, 191, 0); }}
}}
.tiles.updated .tile {{ animation: tile-update 700ms ease-out; }}

.map-shell {{ position: relative; }}
.map-shell .dash-loading, .map-shell [data-dash-is-loading="true"] {{ opacity: 1; }}
.loading-pill {{
    font-family: {DISPLAY};
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.8rem;
    color: {CREAM};
    background: {DEEP};
    padding: 10px 18px;
    border-radius: 999px;
    box-shadow: 0 6px 18px rgba(27, 51, 72, 0.18);
}}

.coords {{
    font-family: {MONO};
    font-size: 1.05rem;
    color: {INK};
    background: {CREAM};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 11px 14px;
    margin-top: 12px;
    text-align: center;
    letter-spacing: 0.02em;
}}
.coords .moved {{
    display: block;
    font-family: {BODY};
    font-size: 0.8rem;
    color: {MUTED_TEXT};
    letter-spacing: 0;
    margin-top: 4px;
}}
.caution {{
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 10px;
    background: #FBF3E4;
    border: 1px solid #D9B978;
    color: #6B4E12;
    font-size: 0.83rem;
    line-height: 1.5;
}}
.hint {{
    color: {MUTED};
    font-size: 0.84rem;
    line-height: 1.55;
    margin-top: 14px;
    max-width: 66ch;
}}
.coverage {{ border-top: 1px solid {BORDER}; padding-top: 12px; }}
.coverage summary {{
    cursor: pointer;
    color: {DEEP};
    font-weight: 500;
    margin-top: 6px;
}}
.coverage-list {{ color: {MUTED_TEXT}; margin-top: 4px; }}
.tile.wide .flag {{
    display: inline-block;
    margin-left: 10px;
    padding: 3px 9px;
    border-radius: 999px;
    background: #FBF3E4;
    color: #6B4E12;
    font-family: {BODY};
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    vertical-align: middle;
}}
.err {{
    color: #8E3227;
    background: #FBEEEC;
    border: 1px solid #E7C4BD;
    border-radius: 10px;
    font-size: 0.88rem;
    margin-top: 12px;
    padding: 10px 12px;
}}
.err:empty {{ display: none; }}

.field {{ margin-top: 16px; }}
.field-label {{
    display: block;
    font-family: {DISPLAY};
    font-weight: 600;
    font-size: 0.74rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {DEEP};
    margin-bottom: 6px;
}}
.field-note {{
    color: {MUTED_TEXT};
    font-size: 0.8rem;
    line-height: 1.45;
    margin-top: 6px;
    max-width: 66ch;
}}
.dash-dropdown-trigger {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {CREAM} !important;
    color: {INK} !important;
    font-family: {BODY};
    font-size: 0.95rem;
    font-weight: 500;
    padding: 9px 12px !important;
}}
.dash-dropdown-trigger:hover {{ border-color: {MID} !important; }}
/* Dash paints its own violet focus ring on the wrapper, not the trigger. */
.dash-dropdown:focus,
.dash-dropdown:focus-visible,
.dash-dropdown:focus-within {{
    outline: 2px solid {DEEP} !important;
    outline-offset: 2px !important;
    border-radius: 12px;
}}
.dash-dropdown-content, .dash-dropdown-options {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: #fff !important;
    font-family: {BODY};
    font-size: 0.95rem;
    width: calc(var(--radix-popover-trigger-width, 100%) - 2px) !important;
    min-width: calc(var(--radix-popover-trigger-width, 100%) - 2px) !important;
    max-width: calc(var(--radix-popover-trigger-width, 100%) - 2px) !important;
    max-height: min(60vh, 360px) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}}
.dash-options-list-option {{ color: {INK} !important; }}
.dash-options-list-option:hover,
.dash-options-list-option.selected {{
    background: {MINT} !important;
    color: {DEEP_TEXT} !important;
}}
.opt-name {{ font-family: {DISPLAY}; font-weight: 600; font-size: 1rem; }}
.opt-desc {{
    font-size: 0.78rem;
    color: {MUTED_TEXT};
    line-height: 1.35;
    max-width: 60ch;
    white-space: normal;
}}
.opt-desc .km {{ color: {DEEP_TEXT}; font-weight: 600; }}
/* The trigger renders the same node as the menu option, so the second line is
   hidden there: it belongs in the list, where methods are compared, not in the
   closed control, where it just clips. */
.dash-dropdown-trigger .opt-desc {{ display: none; }}
.dash-dropdown-trigger .opt-name {{ font-size: 0.95rem; }}

.footer {{
    margin-top: 26px;
    padding-top: 16px;
    border-top: 1px solid {BORDER};
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: center;
    color: {MUTED_TEXT};
    font-size: 0.83rem;
}}
.footer a {{
    color: {DEEP};
    font-weight: 500;
    text-decoration: none;
    border-bottom: 1px solid {BORDER};
}}
.footer a:hover {{ border-bottom-color: {DEEP}; }}
.footer .dot {{ color: {MID}; }}
@media (max-width: 560px) {{
    /* The separators are decoration; wrapping strands them at line ends. */
    .footer .dot {{ display: none; }}
    .footer {{ flex-direction: column; gap: 4px; }}
}}
"""
