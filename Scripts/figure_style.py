"""
QUICK COPY
===
---
Create figure:
---
(single panel)
>>> fig, ax = plt.subplots(figsize=[6,5])

(two panels)
>>> fig, [ax1,ax2] = plt.subplots(1, 2, figsize=[10,5])

(four panels)
>>> fig, axes = plt.subplots(2, 2, figsize=[10,8])

(more settings)
>>> sharex=True, sharey=True
>>> gridspec_kw={"width_ratios": [2, 1]}
>>> gridspec_kw={"height_ratios":[1,.7]}

Raster plot:
---
>>> qgraph.raster_plot(nd.dynamics.spike_times, ax=ax, colors="k", marker=".", ms=3, mec="none")

Legend, shrink size:
---
>>> box = ax.get_position()
... ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

Legend, move outside of frame:
---
>>> ax.legend(loc="center left", bbox_to_anchor=(1,0.5))

Legend, label multiple plots as one:
---
>>> lin1a, = ax.plot([],[], "ko")
... lin1b, = ax.plot([],[], "r-")
... lin2a, = ax.plot([],[], "k^")
... lin2b, = ax.plot([],[], "r--")
... ax.legend([(lin1b, lin1a), (lin2b, lin2a)], ["label 1", "label 2"], title=r"")

Figures in grid:
---
>>> from mpl_toolkits.axes_grid1 import make_axes_locatable
... fig = plt.figure(figsize=[15,15])
... gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[4,1,1], width_ratios=[1,1,1,1], hspace=.5)
... ax1a = fig.add_subplot(gs[:,0])
... div1 = make_axes_locatable(ax1a)
... ax1b = div1.append_axes("bottom", "40%", pad=0, sharex=ax1a)
... ax1c = div1.append_axes("bottom", "40%", pad=0, sharex=ax1a)
... ax1d = div1.append_axes("bottom", "60%", pad=.2, sharex=ax1a)
... axes1 = [ax1a, ax1b, ax1c, ax1d]

Hide ticks, but not labels:
---
>>> ax.tick_params(axis="x", which="both",length=0)

Show minor ticks in log-scale axis:
---
>>> locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(.2, .4, .6, .8),numticks=12)
>>> locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(.1,.2,.3,.4,.5,.6,.7,.8,.9),numticks=12)
... ax.yaxis.set_minor_locator(locmin)
... ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

Major ticks spacing
---
>>> ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

Set axis ticks in multiples of Pi
---
>>> ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
... ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
... ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

"""

import matplotlib.pyplot as plt
import matplotlib

myFontSize0 = 28
myFontSize1 = 20
myFontSize2 = 16
myFontSize3 = 14
myFontSize4 = 10

myMarkerSize0 = 8


# Figure settings
# matplotlib.rcParams['lines.markersize'] = 10
plt.rcParams["font.size"] = myFontSize1
plt.rcParams["xtick.labelsize"] = myFontSize2
plt.rcParams["ytick.labelsize"] = myFontSize2
matplotlib.rc("font", family="serif", serif="cmr10")
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rc("text", usetex=True)

# Legend settings
plt.rcParams["legend.title_fontsize"] = myFontSize2
plt.rcParams["legend.fontsize"] = myFontSize3
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.facecolor"] = (1, 1, 1)
plt.rcParams["legend.edgecolor"] = (0, 0, 0)
plt.rcParams["legend.framealpha"] = .95
plt.rcParams['patch.linewidth'] = .75
plt.rcParams["legend.borderpad"] = .4
plt.rcParams["legend.markerscale"] = 1
plt.rcParams["legend.columnspacing"] = 1.5
