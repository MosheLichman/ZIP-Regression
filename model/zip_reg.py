"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from os.path import join

from commons import objectives
from commons import log_utils as log
from commons import file_utils as flu
from commons import time_measure as tm

from model import gd_commons, pois_reg


class ZipRegression(object):
    def __init__(self, N, M, eta_0=None, eta_u=None, eta_gd_lamb=0, eta_gd_batch_size=10000, eta_gd_max_iter=200,
                 eta_gd_step_size=0.01, eta_gd_tol=0.001, em_num_iter=30, em_tol=0.01,
                 eta_0_prior=0, eta_u_prior=0, interleave=True, min_gd_iter=100, gd_num_dec=np.inf,
                 eta_gd_ll_iters=10, em_ll_iters=1, min_em_iter=2,
                 **kwargs):
        self.N, self.M = N, M
        self.eta_0, self.eta_u = eta_0, eta_u
        self.eta_0_prior, self.eta_u_prior = eta_0_prior, eta_u_prior
        self.trained = False
        
        # EM hyper parameters
        self.em_num_iter = em_num_iter
        self.em_tol = em_tol
        self.em_ll_iters = em_ll_iters
        
        # GD hyperparameters
        self.gd_lamb = eta_gd_lamb
        self.gd_max_iter = eta_gd_max_iter
        self.gd_tol = eta_gd_tol
        self.gd_batch_size = eta_gd_batch_size
        self.gd_step_size = eta_gd_step_size
        self.gd_ll_iters = eta_gd_ll_iters
        self.min_gd_iter = min_gd_iter
        self.min_em_iter = min_em_iter
        self.interleave = interleave
        self.gd_num_dec = gd_num_dec

        # The poisson regression code
        self.pos_model = pois_reg.PoissonRegression(N, M, gd_num_dec=gd_num_dec, min_gd_iter=min_gd_iter, **kwargs)

    def sigmoid_func(self, users, items, user_feat):
        """Computes the sigmoid/logistic function to estimate the pies.

         Args
        ------
            1. users:       <(D, ) int>      user ids
            2. items:       <(D, ) int>      item ids
            3. user_feat:   <(D, f) float>   user features values

         Returns
        ---------
            1. sig:    <(D, ) float>   estimated pies (using the sigmoid/logistic function).
        """
        z = gd_commons.mul_feat_coeff(users, items, user_feat, self.eta_0, self.eta_u)
        sig = 1.0 / (1 + np.exp(-1.0 * z))

        return sig

    def eta_likelihood(self, users, items, user_feat, w_ijt):
        """ Computes the likelihood conditioned on eta.

        This is the logistic likelihood function.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. user_feat:   <(D, f) float>  user features values
            4. w_ijt:       <(D, ) int>     target response values

         Returns
        ---------
            1. ll:       <float>  likelihood
        """
        point = tm.get_point('eta_logistic_likelihood')
        sig = self.sigmoid_func(users, items, user_feat)

        # For robustness making sure no one is totally 1 or totally.
        tmp = np.where(sig == 1)[0]
        sig[tmp] -= 1E-24

        tmp = np.where(sig == 0)[0]
        sig[tmp] += 1E-24

        ll = np.mean(w_ijt * np.log(sig) + (1 - w_ijt) * (np.log(1 - sig)))

        point.collect()

        return ll

    def _eta_derivative_vals(self, users, items, user_feat, w_ijt):
        """Computes the derivations for each element in the matrix.

        Note that this is not where the gradient is computed, but just where each element in the feature table is
        derived. This also includes the two intercept. If there's a prior, it is derived as well.

        The reason for the separation is because of the fixed-regression in which we have a fixed effect for population,
        each individual and each item so it is easier to first compute the derivation at each point using mat operations
        and later compute the different gradients separately.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. user_feat:   <(D, f) float>  user features values
            4. w_ijt:       <(D, ) int>     target response values

         Returns
        ---------
            1. d_features:       <(D, f + 1) float>  derivative of ALL features
            2. d_0_prior:        <float>             derivative of the global intercept prior
            3. d_u_prior:        <(N, f) float>      derivative of the user \eta (including intercept)
        """
        sig = self.sigmoid_func(users, items, user_feat)

        # Adding two columns of ones for the 'non-user' feat. This is done to make the computation easier using matrix
        # operations.
        f_const = np.hstack([np.ones([user_feat.shape[0], 1]), user_feat])
        d_features = np.atleast_2d(w_ijt - sig).T * f_const
        
        d_0_prior = self.gd_lamb * (self.eta_0 - self.eta_0_prior)
        d_u_prior = self.gd_lamb * (self.eta_u - self.eta_u_prior)
        
        return d_features, d_0_prior, d_u_prior
    
    def _initialize_eta(self, f):
        """Makes sure all \eta's are initialized. If not sample from 0 mean normal distribution.

         Args
        ------
            1. f: <int> number of data-driven (non-intercepts) features + the user intercept.
        """
        if self.eta_u is None:
            self.eta_u = np.random.normal(0, 0.1, [self.N, f])
        if self.eta_0 is None:
            self.eta_0 = np.random.normal(0, 0.1, 1)[0]
    
    def learn_eta(self, users, items, user_feat, w_ijt):
        """Performs the e-step of the EM algorithm to estimate the response values w_ijt.

         Args
        ------
            1. users:       <(D, ) int>      user ids
            2. items:       <(D, ) int>      item ids
            3. user_feat:   <(D, f) float>   user features values
            4. w_ijt:       <(D, ) int>      target response values
        """
        self._initialize_eta(user_feat.shape[1])

        # Number of times the likelihood went down. Used to prevent overfitting and parameter explosion.
        num_down = 0

        prev_ll = curr_ll = -np.inf
        reached_conv = False

        for i in range(1, self.gd_max_iter + 1):
            # Sampling a mini-batch
            samp = gd_commons.fast_sample(user_feat.shape[0], self.gd_batch_size)

            eta_sgd_point = tm.get_point('eta_sgd_iter')  # Taking this time point after the sample.

            d_features, d_0_prior, d_u_prior = self._eta_derivative_vals(users[samp], items[samp], user_feat[samp],
                                                                         w_ijt[samp])

            # ADAM initial values
            adam_vals_u = {'mean': np.zeros(self.eta_u.shape), 'var': np.zeros(self.eta_u.shape), 't': 0}
            adam_vals_0 = {'mean': 0, 'var': 0, 't': 0}
            
            g_grad = gd_commons.grad_for_global(d_features[:, 0], d_0_prior)
            u_grad = gd_commons.grad_for_user(users[samp], d_features[:, 1:], d_u_prior)
            
            # These operations are safe because if the user or item were not in the sample the grad for them will be
            # zero.
            self.eta_0 += gd_commons.get_adam_update(self.gd_step_size, g_grad, adam_vals_0)
            self.eta_u += gd_commons.get_adam_update(self.gd_step_size, u_grad, adam_vals_u)

            eta_sgd_point.collect()

            # Checking for convergence - using only the data likelihood.
            if i >= self.min_gd_iter and i % self.gd_ll_iters == 0:
                curr_ll = self.eta_likelihood(users, items, user_feat, w_ijt)

                if curr_ll < prev_ll:
                    num_down += 1

                log.info('ZipRegression.learn_eta: Data log like after %d iterations [%.5f --> %.5f]' % (
                    i, prev_ll, curr_ll))

                if np.abs(curr_ll - prev_ll) <= self.gd_tol or num_down >= self.gd_num_dec:
                    log.info('ZipRegression.learn_eta: Reached convergance after %d iterations' % i)
                    reached_conv = True
                    break

                prev_ll = curr_ll

        if not reached_conv:
            log.info('ZipRegression.learn_eta: Did not reach convergance after %d iterations' % self.gd_max_iter)

        log.info('ZipRegression.learn_eta: Train data log like %.3f' % curr_ll)

    def _e_step(self, users, items, user_feat, target, pie, rate):
        """Performs the e-step of the EM algorithm to estimate the response values w_ijt.

         Args
        ------
            1. users:       <(D, ) int>      user ids
            2. items:       <(D, ) int>      item ids
            3. user_feat:   <(D, f) float>   user features values
            4. target:      <(D, ) int>      target rates
            5. pie:         <(D, ) float>    estimated mixing weights
            6. rate:        <(D, ) float>    estimated rate parameter

         Returns
        ---------
            1. w_ijt:    <(D, ) float>   estimated response values.
        """
        point = tm.get_point('_e_step')

        zero_mask = np.where(target == 0)[0]
        pois_prob = np.exp(objectives.pois_log_prob(target, rate))
        prob_from_rate = pie[zero_mask] * pois_prob[zero_mask]

        # Only need to update the w_ijt at the zero_mask, for the rest it has to come from the rate process so we can
        # leave it as 1.
        w_ijt = np.ones(user_feat.shape[0])
        w_ijt[zero_mask] = prob_from_rate / (prob_from_rate + 1 - pie[zero_mask])

        point.collect()

        return w_ijt

    def _m_step(self, users, items, user_feat, target, w_ijt, curr_rate):
        """Performs the m-step of the EM algorithm to update the model parameters.

        This is where the gradient descent algorithm takes place.

         Args
        ------
            1. users:       <(D, ) int>      user ids
            2. items:       <(D, ) int>      item ids
            3. user_feat:   <(D, f) float>   user features values
            4. target:      <(D, ) int>      target rates
            5. w_ijt:       <(D, ) float>    estimated response values
            6. curr_rate:   <(D, ) float>    current rate parameter

         Returns
        ---------
            1. pie:    <(D, ) float>   expected mixing weights
            2. rate:   <(D, ) float>   new rate parameters
        """
        # M step eta
        self.learn_eta(users, items, user_feat, w_ijt)

        # Computing the new pie values after updating eta
        pie = self.sigmoid_func(users, items, user_feat)

        if self.interleave:
            # This means that there's only one gradient descent algorithm between each e_step. I found it to work better
            # in terms of convergence, so you might want to set this flag True.
            w_ijt = self._e_step(users, items, user_feat, target, pie, curr_rate)

        # M step beta
        self.pos_model._learn_beta(users, items, user_feat, target, w_ijt)

        # Computing the new rate parameter after updating beta
        rate = self.pos_model.get_est_lambda(users, items, user_feat)

        return pie, rate

    def _em(self, users, items, data_feat, target):
        """Runs the EM algorithm to learn both \eta and \beta.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates
        """
        prev_ll = curr_ll = -np.inf
        reached_conv = False

        # Adding the user intercept constant. In my code, the exposure process has different constants than the
        # rate process, so I only pass the data_feat to the methods and deal with the constants separately.
        # The reason I keep the user const in the user_feat is to avoid starting the counts from 1 in the cython code.
        # Other you trust me, or you can go and look at it :)
        user_feat = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])

        # Randomly initializing \eta and \beta
        self._initialize_eta(user_feat.shape[1])
        self.pos_model._initialize_beta(user_feat.shape[1])

        pie = self.sigmoid_func(users, items, user_feat)
        rate = self.pos_model.get_est_lambda(users, items, user_feat)

        # Starting with an ESTEP after randomly initializing eta and beta.
        w_ijt = self._e_step(users, items, user_feat, target, pie, rate)

        # M STEP
        pie, rate = self._m_step(users, items, user_feat, target, w_ijt, rate)

        for em_i in xrange(self.em_num_iter):
            w_ijt = self._e_step(users, items, user_feat, target, pie, rate)
            pie, rate = self._m_step(users, items, user_feat, target, w_ijt, rate)

            # ZIP probability
            if em_i > self.min_em_iter and em_i % self.em_ll_iters == 0:
                curr_ll = np.mean(self.data_log_like(target, rate, pie))
                log.info('ZipRegression._em: Data LL at iteration %d [%.5f --> %.5f]' % (em_i, prev_ll, curr_ll))

                if np.abs(prev_ll - curr_ll) < self.em_tol:
                    log.info('ZipRegression._em: Reached conversion')

                    reached_conv = True
                    break

                prev_ll = curr_ll

        if not reached_conv:
            log.error('ZipRegression._em: Did not reach convergance after %d iterations' % self.em_num_iter)

        log.info('ZipRegression._em: Train data log like %.5f' % curr_ll)

    def learn_model(self, users, items, data_feat, target):
        """Optimized the ZIP model parameter. This includes both the EM and the gradient descent algorithms.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates

         Raise
        -------
            1. ValueError if any of the inputs are not of shape[0] == D
            2. ValueError if data_feat is not a 2d matrix.
            3. Bunch of type and value errors if you did not follow the args input instruction.
        """
        if not users.shape[0] == items.shape[0] == data_feat.shape[0] == target.shape[0]:
            raise ValueError('All inputs must be same size D.')
        if data_feat.ndim != 2:
            raise ValueError('Data driven features must be 2d')

        # Saving the user ids that this model used in training. This is used later in the test and validation to filter
        # out the users we've never seen before. This is crucial for online training when users can "join" the data at
        # later time windows. If in your pre-processing of the data you made sure that all users are in all time windows
        # this will not be of any use to you.
        self.trained_users = np.unique(users)

        self._em(users, items, data_feat, target)

        # Indicating the the model was trained. It's not really necessary, just good practice if you ask me :)
        self.trained = True

    def predict(self, users, items, data_feat):
        """Predicts the expected rate value.

        The expected rate under the ZIP model is \pie * \hat{lambda}. This comes from the fact that at 0 rate we don't
        care because it's not part of the expected value.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features

         Returns
        ---------
            1. est_lambda: <(D, ) float>    expected rate value
        """
        user_feat = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
        lambda_est = self.pos_model.get_est_lambda(users, items, user_feat)

        pis = self.sigmoid_func(users, items, user_feat)
        zip_exp = lambda_est * pis

        return zip_exp

    def data_log_like(self, target, est_lambda, pie):
        """Comptues the data log likelihood of the ZIP model.

         Args
        ------
            1. target:      <(D, ) int>     target rates
            2. est_lambda:  <(D, ) float>   estimated rate parameters
            3. pie:         <(D, ) float>   expected mixing weights.

         Returns
        ---------
            1. zip_log_like:    <(D, ) float>    data log-likelihood for each point
        """
        zero_mask = np.where(target == 0)[0]

        pois_log_like = np.log(pie) + objectives.pois_log_prob(target, est_lambda)
        zero_inf_log_like = np.log(1 - pie[zero_mask])

        # Combining the two
        zip_log_like = pois_log_like
        zip_log_like[zero_mask] = np.logaddexp(zip_log_like[zero_mask], zero_inf_log_like)

        return zip_log_like

    def save_model(self, path):
        model_name = 'zip'
        # Saving the zip_model eta - we can estimate the pies from it
        flu.np_save(path, '%s_eta_0.npy' % model_name, self.eta_0)
        flu.np_save(path, '%s_eta_u.npy' % model_name, self.eta_u)

        # Saving the pos_mode beta - we can estimate the lambda from it
        flu.np_save(path, '%s_beta_0.npy' % model_name, self.pos_model.beta_0)
        flu.np_save(path, '%s_beta_i.npy' % model_name, self.pos_model.beta_i)
        flu.np_save(path, '%s_beta_u.npy' % model_name, self.pos_model.beta_u)

        flu.np_save(path, '%s_trained_users.npy' % model_name, self.trained_users)

    @staticmethod
    def load_model(path):
        model_name = 'zip'
        log.info('ZipRegression.load_model: Loading model %s from path %s' % (model_name, path))

        eta_0 = flu.np_load(join(path, '%s_eta_0.npy' % model_name))
        eta_0 = np.atleast_1d(eta_0)[0]

        eta_u = flu.np_load(join(path, '%s_eta_u.npy' % model_name))

        beta_0 = flu.np_load(join(path, '%s_beta_0.npy' % model_name))
        beta_0 = np.atleast_1d(beta_0)[0]

        beta_i = flu.np_load(join(path, '%s_beta_i.npy' % model_name))
        beta_u = flu.np_load(join(path, '%s_beta_u.npy' % model_name))

        n = beta_u.shape[0]
        m = beta_i.shape[0]

        model = ZipRegression(N=n, M=m, eta_0=eta_0, eta_u=eta_u, beta_0=beta_0, beta_i=beta_i, beta_u=beta_u)
        model.trained_users = flu.np_load(join(path, '%s_trained_users.npy' % model_name))
        model.trained = True

        return model

    """

      Evaluation Metrics

    """

    def test_log_prob(self, users, items, data_feat, target, return_vals=False):
        """Evaluates Log-Likelihood accuracy on test data.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates
            5. return_vals: <bool>          if True returns all the values instead of the mean (default = False)

         Returns
        ---------
            1. <float> average log-likelihood (if return_vals is False)
            2  <(D, ) float> log likelihood for each point (if return_vals is True)
            3. -np.inf if model is not trained.
        """
        if not self.trained:
            return -np.inf

        # At optimization time - it is very likely that some users don't have train data (because they're not active
        # yet). This makes sure that I'm not testing on them.
        test_users = np.unique(users)
        trained_user_mask = np.where(np.in1d(users, self.trained_users, assume_unique=False))
        log.info('Trained on %d out of %d test users' % (self.trained_users.shape[0], test_users.shape[0]))

        user_feat = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
        lambda_est = self.pos_model.get_est_lambda(users[trained_user_mask], items[trained_user_mask],
                                                   user_feat[trained_user_mask])

        pis = self.sigmoid_func(users[trained_user_mask], items[trained_user_mask], user_feat[trained_user_mask])

        vals = self.data_log_like(target[trained_user_mask], lambda_est, pis)

        if return_vals:
            return vals
        else:
            return np.mean(vals)

    def test_mae(self, users, items, data_feat, target, return_vals=False):
        """Evaluates Mean Average Error accuracy on test data.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates
            5. return_vals: <bool>          if True returns all the values instead of the mean (default = False)

         Returns
        ---------
            1. <float> average mae (if return_vals is False)
            2  <(D, ) float> mae for each point (if return_vals is True)
            3. np.inf if model is not trained.
        """
        if not self.trained:
            return np.inf

        # At optimization time - it is very likely that some users don't have train data (because they're not active
        # yet). This makes sure that I'm not testing on them.
        test_users = np.unique(users)
        trained_user_mask = np.where(np.in1d(users, self.trained_users, assume_unique=False))
        log.info('Trained on %d out of %d test users' % (self.trained_users.shape[0], test_users.shape[0]))

        zip_exp = self.predict(users[trained_user_mask], items[trained_user_mask], data_feat[trained_user_mask])
        vals = objectives.mae(target[trained_user_mask], zip_exp)

        if return_vals:
            return vals
        else:
            return np.mean(vals)

    def test_mse(self, users, items, data_feat, target, return_vals=False):
        """Evaluates Mean Squared Error accuracy on test data.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates
            5. return_vals: <bool>          if True returns all the values instead of the mean (default = False)

         Returns
        ---------
            1. <float> average mse (if return_vals is False)
            2  <(D, ) float> mse for each point (if return_vals is True)
            3. np.inf if model is not trained.
        """
        if not self.trained:
            return np.inf

        # At optimization time - it is very likely that some users don't have train data (because they're not active
        # yet). This makes sure that I'm not testing on them.
        test_users = np.unique(users)
        trained_user_mask = np.where(np.in1d(users, self.trained_users, assume_unique=False))
        log.info('Trained on %d out of %d test users' % (self.trained_users.shape[0], test_users.shape[0]))

        zip_exp = self.predict(users[trained_user_mask], items[trained_user_mask], data_feat[trained_user_mask])
        vals = objectives.mse(target[trained_user_mask], zip_exp)

        if return_vals:
            return vals
        else:
            return np.mean(vals)

    def test_f1(self, users, items, data_feat, target, return_vals=False):
        """Evaluates F1 accuracy on test data.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates
            5. return_vals: <bool>          if True returns all the values instead of the mean (default = False)

         Returns
        ---------
            1. <float> average f1 (if return_vals is False)
            2  <(D, ) float> f1 for each point (if return_vals is True)
            3. np.inf if model is not trained.
        """
        if not self.trained:
            return np.inf

        # At optimization time - it is very likely that some users don't have train data (because they're not active
        # yet). This makes sure that I'm not testing on them.
        test_users = np.unique(users)
        trained_user_mask = np.where(np.in1d(users, self.trained_users, assume_unique=False))
        log.info('Trained on %d out of %d test users' % (self.trained_users.shape[0], test_users.shape[0]))

        zip_exp = self.predict(users[trained_user_mask], items[trained_user_mask], data_feat[trained_user_mask])
        vals = objectives.f_measure(target[trained_user_mask], zip_exp)

        if return_vals:
            return vals
        else:
            return np.mean(vals)
