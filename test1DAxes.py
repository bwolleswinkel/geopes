"""This is a working implementation of a 1D axes class for matplotlib"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Axes1D(Axes):
    """A custom 1D axis class that inherits from matplotlib Axes.
    
    Features:
    - No vertical axes (completely hidden)
    - Horizontal axis centered at y=0
    - Prevents upward/downward scrolling/panning
    - Override plot method to accept calls without y-arguments (defaulting y to 0)

    # FROM: GitHub Copilot, Claude Sonnet 4 | 2026/01/07
    
    """
    
    def __init__(self, fig, *args, fixed_ylim=(-1, 1), **kwargs):
        """Initialize the Axes1D instance.
        
        Parameters:
        -----------
        fig : Figure
            The parent figure
        *args : tuple
            Subplot positioning arguments (like 111 or (2,1,1))
        fixed_ylim : tuple, optional
            Fixed y-limits for the axis (default: (-0.1, 0.1))
        **kwargs : dict
            Additional arguments passed to parent Axes class
        
        """
        super().__init__(fig, *args, **kwargs)
        self.fixed_ylim = fixed_ylim
        self._updating_ylim = False  # Flag to prevent recursion
        # Setup the 1D appearance
        self._setup_1d_appearance()
        # Setup y-limit control
        self._setup_ylimit_control()
        self.name = '1d'
    
    def _setup_1d_appearance(self):
        """Configure the axes to appear as a 1D plot"""
        # Move bottom x-axis to center, passing through (0,0)
        self.spines['bottom'].set_position('center')
        # Completely eliminate ALL y-axis elements
        self.spines['left'].set_color('none')     # Hide left y-axis line
        self.spines['right'].set_color('none')    # Hide right y-axis line
        self.spines['top'].set_color('none')      # Hide top x-axis line
        # Remove y-axis ticks and labels completely
        self.yaxis.set_ticks_position('none')     # Remove tick marks
        self.set_yticks([])                       # Remove tick labels
        # Show ticks only on bottom x-axis
        self.xaxis.set_ticks_position('bottom')
        # Set initial y-limits
        self.set_ylim(self.fixed_ylim)
    
    def _setup_ylimit_control(self):
        """Setup callback to prevent y-axis movement"""

        def on_ylims_change(ax):
            if not self._updating_ylim:  # Prevent recursion
                self._updating_ylim = True
                ax.set_ylim(self.fixed_ylim)
                self._updating_ylim = False
        
        self.callbacks.connect('ylim_changed', on_ylims_change)
    
    def plot(self, *args, **kwargs):
        """Override plot method to handle 1D plotting. If only x-values are provided, y-values default to 0.
        Supports all standard matplotlib plot arguments.
        
        """
        # Check if `linewidth` is provided in kwargs, else set to 2
        if 'linewidth' not in kwargs and 'lw' not in kwargs:
            kwargs['linewidth'] = 2
        # Handle case where only x-values are provided
        if len(args) == 1 and not isinstance(args[0], str):
            # Single argument that's not a format string - assume it's x-data
            x_data = args[0]
            y_data = np.zeros_like(x_data)
            args = (x_data, y_data)
        elif len(args) >= 2:
            # Standard case - x and y data provided
            pass
        # Call parent plot method
        return super().plot(*args, **kwargs)
    
    def scatter(self, x, y=None, **kwargs):
        """Override scatter method to handle 1D scatter plots. If y is not provided, defaults to 0.
        
        """
        if 's' not in kwargs:
            kwargs['s'] = 5
        if y is None:
            y = np.zeros_like(x)
        return super().scatter(x, y, **kwargs)
    
    def set_ylim(self, *args, **kwargs):
        """Override set_ylim to maintain fixed limits"""
        if not self._updating_ylim:
            # Only allow setting ylim during initialization or internal updates
            super().set_ylim(self.fixed_ylim)
        else:
            super().set_ylim(*args, **kwargs)
    
    def set_fixed_ylim(self, ylim):
        """Change the fixed y-limits.
        
        Parameters:
        -----------
        ylim : tuple
            New fixed y-limits (ymin, ymax)
        """
        self.fixed_ylim = ylim
        self._updating_ylim = True
        super().set_ylim(ylim)
        self._updating_ylim = False


def add_1d_subplot(fig, *args, fixed_ylim=(-0.1, 0.1), **kwargs):
    """Create and add an Axes1D subplot to a figure.
    
    Parameters:
    -----------
    fig : Figure
        The figure to add the subplot to
    *args : tuple
        Position arguments (like subplot(111) or subplot(2,1,1))
    fixed_ylim : tuple, optional
        Fixed y-limits for the 1D axis (default: (-0.1, 0.1))
    **kwargs : dict
        Additional keyword arguments passed to subplot creation
    
    Returns:
    --------
    Axes1D
        The created 1D axes instance
    
    """
    # Create Axes1D directly using subplot positioning
    # Default to 111 if no args provided
    if not args:
        args = (111,)
    
    ax = Axes1D(fig, *args, fixed_ylim=fixed_ylim, **kwargs)
    fig.add_axes(ax)
    
    return ax


# Run the demonstration
if __name__ == "__main__":
    fig = plt.figure()
    ax = add_1d_subplot(fig)
    ax.plot([0, 1], label=r"$P_{1}$")
    ax.plot([9, 10], label=r"$P_{2}$")
    ax.scatter([0, 1])
    ax.scatter([9, 10])
    ax.legend(loc='upper left')
    plt.show()