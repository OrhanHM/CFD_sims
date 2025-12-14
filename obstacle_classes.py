import numpy as np
import scipy as sc
import matplotlib as mpl

from helper_functions import five_point_stencil, parameter_scaling, generate_NACA4


class Wing:
    def __init__(self, upper_height_params=None, lower_height_params=None, naca_code=None, naca_points=50, inverted=False,
                 chord_length=0.7, x_start=0.6, thickness=0.3, scale=True,
                 interpolation_method='pchip', n_points=10000, boundary_condition=None):
        """Define the upper and lower surfaces of a wing object from a set of control points that we interpolate
        through. Specify upper and lower height parameters directly or input a four digit NACA4 code to automatically
        build a wing.
        Possible interpolation methods are pchip ('pchip' or 'p') and cubic spline ('cubic_spline' or 'c')
        If cubic spline is used, specific boundary conditions on edge derivatives can be specified with a two-tuple
        matching scipy.interpolate.CubicSpline syntax."""

        self.thickness = thickness
        self.x_start = x_start
        self.boundary_condition = boundary_condition
        self.scale = scale
        self.interpolation_method = interpolation_method
        self.naca_code = naca_code
        self.inverted = inverted
        self.upper_height_params = upper_height_params
        self.lower_height_params = lower_height_params

        if upper_height_params is None or lower_height_params is None:
            if naca_code is None: # No definition given
                raise ValueError('Must specify upper and lower control points directly, or a NACA code')
            else: # defined with naca code
                self.interpolate_x_upper, self.upper_heights, self.interpolate_x_lower, self.lower_heights = generate_NACA4(naca_code, n_points=naca_points)

        else: # Defined with control points
            assert (upper_height_params.ndim == 1) and (lower_height_params.ndim == 1), 'Height arrays are not one dimensional'
            assert upper_height_params.shape[0] == lower_height_params.shape[0], 'Shapes of upper and lower height parameters do not match'

            self.upper_heights = np.concatenate([[0.0], upper_height_params, [0.0]])
            self.lower_heights = np.concatenate([[0.0], lower_height_params, [0.0]])

            self.n_control_points = len(self.upper_heights)
            self.interpolate_x_upper = np.linspace(0, 1, self.n_control_points)
            self.interpolate_x_lower = np.linspace(0, 1, self.n_control_points)

        if scale:
            self.upper_heights[1:-1] = parameter_scaling(self.upper_heights[1:-1])
            self.lower_heights[1:-1] = parameter_scaling(self.lower_heights[1:-1])

        if inverted:
            self.upper_heights, self.lower_heights = self.lower_heights, self.upper_heights


        if interpolation_method in ('pchip', 'p'):
            upper_interpolator = sc.interpolate.PchipInterpolator(self.interpolate_x_upper, self.upper_heights)
            lower_interpolator = sc.interpolate.PchipInterpolator(self.interpolate_x_lower, self.lower_heights)
        elif interpolation_method in ('cubic_spline', 'c'):
            if boundary_condition is None:
                upper_interpolator = sc.interpolate.CubicSpline(self.interpolate_x_upper, self.upper_heights)
                lower_interpolator = sc.interpolate.CubicSpline(self.interpolate_x_lower, self.lower_heights)
            else:
                upper_interpolator = sc.interpolate.CubicSpline(self.interpolate_x_upper, self.upper_heights, bc_type=boundary_condition[0])
                lower_interpolator = sc.interpolate.CubicSpline(self.interpolate_x_lower, self.lower_heights, bc_type=boundary_condition[1])
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

        x_fine = np.linspace(0, 1, n_points)
        self.x_real = x_start + x_fine * chord_length

        y_upper = upper_interpolator(x_fine)
        y_lower = -lower_interpolator(x_fine)

        # Ensure equal max cross-section for all wings
        max_width = np.max(y_upper - y_lower)
        y_upper *= thickness / max_width
        y_lower *= thickness / max_width

        self.upper_surface = np.column_stack((self.x_real, y_upper))
        self.lower_surface = np.column_stack((self.x_real, y_lower))

        self.all_points = np.vstack((self.upper_surface, self.lower_surface[::-1][1:-1]))


    def interior_mask(self, X, Y):
        """Takes a wing object and two arrays representing a numpy meshgrid.
         Returns a mask that is True at all grid point locations contained within the wing"""

        path = mpl.path.Path(np.vstack((self.all_points, [self.x_start, 0.0])))
        points = np.column_stack([X.ravel(), Y.ravel()])
        mask_flat = path.contains_points(points)

        return mask_flat.reshape(X.shape)


    def grid_boundary(self, X, Y):
        """Takes a wing object and two arrays representing a numpy meshgrid.
        Returns all a mask that is True at all grid points that are boundary points to the wing
        (defined as having an interior point as an immediate neighbor)"""

        int_mask = self.interior_mask(X, Y)
        xmin, xplus, ymin, yplus = five_point_stencil(int_mask)

        return (xmin | xplus | ymin | yplus) ^ int_mask


    def influence_sections(self, X, Y):
        """Determines influence section for each grid boundary point, defined as all wing points closest to said
        boundary point. Returns length of linear-approximated influence section (dS) and unit inward normal vectors"""

        boundary_mask = self.grid_boundary(X, Y)
        boundary_points = np.column_stack((X[boundary_mask], Y[boundary_mask]))

        distances = sc.spatial.distance.cdist(self.all_points, boundary_points)
        closest_grid_pt_idxs = np.argmin(distances, axis=1)

        points_of_influence = [[] for _ in boundary_points] # each list contains indices of wing points closest to boundary point

        for wing_idx, grid_idx in enumerate(closest_grid_pt_idxs):
            points_of_influence[grid_idx].append(wing_idx)


        for i, ls in enumerate(points_of_influence):
            points_of_influence[i] = np.asarray(ls)
            # Unless section of influence wraps around front of wing, points should compose the entirety of a simple interval
            if 0 not in ls:
                if not np.allclose(np.diff(points_of_influence[i]), 1):
                    raise ValueError(f'Non-continuous region of influence for a boundary point {i}')

        dS = np.zeros(len(boundary_points))
        dS_vecs = np.zeros((len(boundary_points), 2))
        inward_normals = np.zeros((len(boundary_points), 2))

        for i, ls in enumerate(points_of_influence):
            if len(ls) == 0:
                dS[i] = 0
                dS_vecs[i] = np.zeros(2)
                inward_normals[i] = np.array([1, 0]) # Default, unimportant because multiplied by dS=0 in force calcs

            else:
                if 0 in ls and len(self.all_points)-1 in ls: # wraparound case
                    low_mask = ls < int(len(self.all_points) / 2)
                    lower_inds = ls[low_mask]
                    higher_inds = ls[~low_mask]

                    start_ind = np.min(higher_inds) # lowest of the end is the start point
                    end_ind = np.max(lower_inds) # highest of the starting section is the end point

                else:
                    start_ind = np.min(ls)
                    end_ind = np.max(ls)

                # edges of section of influence is midpoint of start (end) point with the preceding (following) point
                start_point = (self.all_points[start_ind] + self.all_points[start_ind-1])/2

                if end_ind == len(self.all_points)-1:
                    end_point = (self.all_points[-1] + self.all_points[0]) / 2
                else:
                    end_point = (self.all_points[end_ind] + self.all_points[end_ind+1]) / 2

                dS_vec = end_point-start_point
                dS[i] = np.linalg.norm(dS_vec)
                dS_vecs[i] = dS_vec

                # dS vectors traverse curve CW so 90° CW rotation transforms to inward normal
                inward_normals[i] = [dS_vec[1], -dS_vec[0]]

        unit_normals = (inward_normals.T / np.linalg.norm(inward_normals.T, axis=0)) # Shape (2, n_wing_points)

        assert np.allclose(np.linalg.norm(unit_normals, axis=0), 1), "Inward normals are not properly normalized"

        return dS, unit_normals

