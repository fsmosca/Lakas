"""
plotter.py

Plots in html file the data from optimizer.


Needed:
    Python >= 3.85

Requirements:
    nevergrad:
        pip install nevergrad
    hiplot:
        pip install hiplot

References:
    nevergrad:
        https://github.com/facebookresearch/nevergrad
    hiplot:
        # https://github.com/facebookresearch/hiplot
        # https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/
"""


import nevergrad as ng

# Output from lakas optimizer.
plot_data_fn = 'plot_data.txt'

nevergrad_logger = ng.callbacks.ParametersLogger(plot_data_fn)

exp = nevergrad_logger.to_hiplot_experiment()
exp.to_html(f'{plot_data_fn}.html')
