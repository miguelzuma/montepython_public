from montepython.likelihood_class import Likelihood
import numpy as np


class Planck_compressed(Likelihood):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.

    def loglkl(self, cosmo, data):

        derived_params = cosmo.get_current_derived_parameters(['z_rec', 'rs_rec'])

        #z_star = cosmo.z_optical_depth_unity()  # z at which the optical depth is unity
        z_star = derived_params['z_rec']  # z at which the optical depth is unity
        # comoving angular diameter distance
        D_A_star = cosmo.angular_distance(z_star) * (z_star + 1)
        #rs_star = cosmo.rs_at_z(z_star)
        rs_star = derived_params['rs_rec']

        R = np.sqrt(cosmo.Omega_m()) * cosmo.Hubble(0) * D_A_star
        l_a = np.pi * D_A_star / rs_star
        omega_b = cosmo.omega_b()
        n_s = cosmo.n_s()

        # Using self in order to be accesible by data_mp_likelihoods.py
        # work-in-class module.
        # https://github.com/miguelzuma/work-in-class &&
        # https://github.com/ardok-m/work-in-class &&
        self.values = np.array([R, l_a, omega_b, n_s])

        # loglkl = -1/2 * (x_i - x_mean_i) * 1/sigma_i * corr_mat_ij^-1 *
        #                 1/sigma_j * (x_j - x_mean_j)

        differences_over_sigma = (self.values - np.array(self.planck_values_mean)) / self.planck_values_sigma

        loglkl = -0.5 * np.dot(differences_over_sigma,
                               np.dot(self.inverse_correlation_matrix,
                                      differences_over_sigma))

        return loglkl
