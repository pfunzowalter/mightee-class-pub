import numpy as np
import pandas as pd # import panda package, and we call it pd. (saves the data as a data frame fromat)
import matplotlib.pyplot as plt # this package will be use to draw some usful graphs (like bar graphs)
import scipy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

#dont touch the code
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs): # https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


#dont touch the code (PROF)
def get_coorelationCoffe (x, y):
     return (np.corrcoef( [np.array(x).flatten(), np.array(y).flatten()] )[0][1] )
    
def error (coeff, x): # I need some explnation here, is this the SDevation error
    sd = np.sqrt( (1-(coeff**2))/(len(x)) )
    return sd
def p_value (coeff, sd): # I need some explnation here, is this name "p_value" correct ?
    z_score = coeff/sd
    p = scipy.stats.norm.sf(abs(z_score))*2
    return p

#%-------------------third-----------------------------
def drawGraph(xFer, yFer, xNF, yNF, xaxis, yaxis, sol, xl, yl, text, loc='lower right'):
    # Set fixed figure size (width, height in inches)
    fig, ax_nstd = plt.subplots(figsize=(10, 8))
    
    # --- Plot data (same as before) ---
    ax_nstd.scatter(np.array(xNF).flatten(), np.array(yNF).flatten(), s=8, color='blue', label='AGN', alpha=1)
    confidence_ellipse(np.array(xNF).flatten(), np.array(yNF).flatten(), ax_nstd, n_std=2, lw=2, edgecolor='b', linestyle='--')
    ax_nstd.scatter(np.array(xFer).flatten(), np.array(yFer).flatten(), s=18, facecolors='none', edgecolors='r', label='SFG', alpha=0.5)
    confidence_ellipse(np.array(xFer).flatten(), np.array(yFer).flatten(), ax_nstd, n_std=2, lw=2, edgecolor='r', linestyle='-')

    # --- Force consistent axis limits ---
    ax_nstd.set_xlim(xl)  # xl = (xmin, xmax) from input
    ax_nstd.set_ylim(yl)  # yl = (ymin, ymax) from input

    # --- Remove last tick (from previous solution) ---
    xticks = ax_nstd.get_xticks()
    yticks = ax_nstd.get_yticks()
    ax_nstd.set_xticks(xticks[:-1])
    ax_nstd.set_yticks(yticks[:-1])

    # --- Standardize margins ---
    plt.subplots_adjust(
        left=0.12,    # Left margin (increase if y-labels are cut off)
        right=0.95, #95default/cor3 98/   # Right margin
        bottom=0.12,  # Bottom margin (increase if x-labels are cut off)
        top=0.95 #0.95/0.92 tsne      # Top margin
    )

    # --- Labels, legend, text (same as before) ---
    ax_nstd.set_xlabel(xaxis, fontsize=28, fontweight='bold')
    ax_nstd.set_ylabel(yaxis, fontsize=28, fontweight='bold')
    ax_nstd.tick_params(axis='both', labelsize=25)
    ax_nstd.legend(prop={"size":18}, loc=loc)
    # ax_nstd.text(0.07, 0.97, text, transform=ax_nstd.transAxes, fontsize=18,
    #             bbox=dict(facecolor='white', alpha=0.5, edgecolor='grey', boxstyle='round'),
    #             verticalalignment='top', horizontalalignment='right')
    ax_nstd.text(
    0.08, 0.97, text, 
    transform=ax_nstd.transAxes,
    fontsize=25,  # Match default legend fontsize (10pt)
    bbox=dict(
        facecolor='white',
        edgecolor='0.8',  # Default legend frame color (light gray)
        alpha=1.0,       # Fully opaque (default)
        boxstyle='round',  # Matches default legend rounding
        linewidth=0.8,    # Default legend frame linewidth
    ),
    verticalalignment='top',
    horizontalalignment='right'
    )

    # --- Save with controlled padding ---
    plt.show()
    plt.savefig(
        # f'plots/correlation/final{sol}.pdf',
        f'normalised/corrs/final{sol}.pdf',
        dpi=300,
        bbox_inches='tight',  # Optional: Can replace with `pad_inches=0.1` if needed
        pad_inches=0.1         # Adds padding (in inches) if bbox_inches='tight' is used
    )
    plt.close()  # Prevents figure from displaying in notebooks
#%-------------------end-----------------------------
    #### Drawing the graphs ends here, now with the calculation, here we are just using the functions above. thanks
    print()
    print("SFGs")
    fer_coeff = get_coorelationCoffe (xFer, yFer)
    fer_error = error(fer_coeff, xFer)
    fer_p = p_value(fer_coeff, fer_error)
    
    print( "correlation coefficient:" + str(fer_coeff) )
    print("error: "+str(fer_error))
    print("p: " +str(fer_p))
    
    
    print()
    print("AGNs")
    nf_coeff = get_coorelationCoffe (xNF, yNF)
    nf_error = error(nf_coeff, xNF)
    nf_p = p_value(nf_coeff, nf_error)
    
    print( "correlation coefficient:" + str(nf_coeff) )
    print("error: "+str(nf_error))
    print("p: " +str(nf_p))

    

