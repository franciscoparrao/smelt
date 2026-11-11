"""Draft maps of Geo-XGBoost local feature importance (gxgb official output).

One panel per predictor (lithology one-hot columns summed into one panel for
cations) plus a 'dominant predictor' panel. North polar stereographic; input
coordinates are EPSG:3995 in km. Run with cartopy if available, else plain axes.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

R = "/home/franciscoparrao/proyectos/smelt/artic/results/"
OUT = os.path.join(R, "figs"); os.makedirs(OUT, exist_ok=True)
try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False
print("cartopy:", HAVE_CARTOPY)

LABEL = {"Imp_Elev": "Elevation", "Imp_SOCC": "Soil organic C", "Imp_PermExt": "Permafrost extent",
         "Imp_ALT": "Active-layer thickness", "Imp_NDVI": "NDVI", "Imp_Temp": "Temperature",
         "Imp_Pre": "Precipitation", "Imp_Month": "Month", "Imp_Geo": "Lithology (all classes)"}
to_ll = Transformer.from_crs(3995, 4326, always_xy=True)

def panels_for(name):
    li = pd.read_csv(R + f"{name}_local_importance_gxgb.csv")
    imp = li.drop(columns=["x_km", "y_km"])
    geo = [c for c in imp.columns if c.startswith("Imp_Geo_")]
    if geo:
        imp["Imp_Geo"] = imp[geo].sum(axis=1); imp = imp.drop(columns=geo)
    imp = imp.div(imp.sum(axis=1).replace(0, np.nan), axis=0)  # share per site
    lon, lat = to_ll.transform(li.x_km.values * 1000, li.y_km.values * 1000)
    return lon, lat, imp

def make_axes(n):
    ncol = 3; nrow = int(np.ceil(n / ncol))
    if HAVE_CARTOPY:
        proj = ccrs.NorthPolarStereo(central_longitude=-45)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), subplot_kw=dict(projection=proj))
        import matplotlib.path as mpath
        r = abs(proj.transform_point(0, 57, ccrs.PlateCarree())[1])
        circle = mpath.Path.circle((0, 0), r)
        for ax in axes.ravel():
            ax.set_xlim(-r, r); ax.set_ylim(-r, r)
            ax.set_boundary(circle, transform=proj)
            ax.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
            ax.coastlines(linewidth=0.4, color="#666666")
            ax.gridlines(linewidth=0.3, color="#bbbbbb")
        return fig, axes.ravel(), dict(transform=ccrs.PlateCarree())
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    return fig, axes.ravel(), {}

for name, title in (("cation", "Major cations (sum, meq/L, log)"), ("doc", "DOC (mg/L, log)")):
    lon, lat, imp = panels_for(name)
    cols = [c for c in LABEL if c in imp.columns]
    fig, axes, kw = make_axes(len(cols) + 1)
    for ax, c in zip(axes, cols):
        s = ax.scatter(lon, lat, c=imp[c], s=4, cmap="viridis", vmin=0, vmax=np.nanpercentile(imp[c], 98), **kw)
        ax.set_title(LABEL[c], fontsize=10)
        plt.colorbar(s, ax=ax, shrink=0.6, pad=0.02).set_label("local importance share", fontsize=7)
    ax = axes[len(cols)]
    dom = imp[cols].idxmax(axis=1)
    cats = sorted(dom.unique()); cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats):
        m = dom == c
        ax.scatter(lon[m], lat[m], s=4, color=cmap(i), label=LABEL[c], **kw)
    ax.set_title("Dominant predictor (local gain)", fontsize=10)
    ax.legend(fontsize=6, markerscale=3, loc="lower left")
    for ax in axes[len(cols) + 1:]:
        ax.set_visible(False)
    fig.suptitle(f"Geo-XGBoost local feature importance, {title}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(OUT, f"{name}_local_importance_maps.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    print("saved", p, "| dominant predictor counts:", dom.value_counts().to_dict())
