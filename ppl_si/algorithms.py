import numpy as np
from numpy.linalg import pinv
from skglm import WeightedLasso, Lasso
from scipy.stats import norm


def soft_threshold(x, lam):
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    else:
        return 0.0


def inverse_linfty_one_column(Sigma, j_index, mu_k, maxiter=50, threshold=1e-2):
    Sigma = np.asarray(Sigma)
    p = Sigma.shape[0]

    rho = np.max(np.abs(np.delete(Sigma[j_index, :], j_index))) / Sigma[j_index, j_index]
    mu0 = rho / (1 + rho)

    theta = np.zeros(p)

    if mu_k >= mu0:
        theta[j_index] = (1 - mu0) / Sigma[j_index, j_index]
        return {"theta_j": theta, "iter": 0}

    diff_norm2 = 1.0
    last_norm2 = 1.0
    iter_cnt = 1
    iter_old = 1

    theta[j_index] = (1 - mu0) / Sigma[j_index, j_index]
    theta_old = theta.copy()

    Sigma_tilde = Sigma.copy()
    np.fill_diagonal(Sigma_tilde, 0)

    vs = -Sigma_tilde @ theta

    while iter_cnt <= maxiter and diff_norm2 >= threshold * last_norm2:

        for j in range(p):
            oldval = theta[j]
            v = vs[j]

            if j == j_index:
                v += 1.0

            theta[j] = soft_threshold(v, mu_k) / Sigma[j, j]

            if oldval != theta[j]:
                vs += (oldval - theta[j]) * Sigma_tilde[:, j]

        iter_cnt += 1

        if iter_cnt == 2 * iter_old:
            d = theta - theta_old
            diff_norm2 = np.linalg.norm(d)
            last_norm2 = np.linalg.norm(theta)

            iter_old = iter_cnt
            theta_old = theta.copy()

            if iter_cnt > 10:
                vs = -Sigma_tilde @ theta

    return {"theta_j": theta, "iter": iter_cnt}


def inverse_linfty(Sigma, n, resol=1.5, mu_k=None, maxiter=50, threshold=1e-2):
    Sigma = np.asarray(Sigma)
    p = Sigma.shape[0]

    Theta_hat = np.zeros((p, p))

    mu_given = mu_k is not None

    for j in range(p):

        if not mu_given:
            mu_k = (1 / np.sqrt(n)) * norm.ppf(1 - 0.1 / (p ** 2))

        mu_stop = False
        attempt = 1
        increase = 0

        while not mu_stop and attempt < 10:

            last_theta = theta = np.zeros(p)

            out = inverse_linfty_one_column(
                Sigma, j, mu_k,
                maxiter=maxiter, threshold=threshold
            )
            theta = out["theta_j"]
            it = out["iter"]

            if mu_given:
                mu_stop = True

            else:
                if attempt == 1:
                    if it == (maxiter + 1):
                        increase = 1
                        mu_k *= resol
                    else:
                        increase = 0
                        mu_k /= resol
                else:
                    if increase == 1 and it == (maxiter + 1):
                        mu_k *= resol
                    elif increase == 1 and it < (maxiter + 1):
                        mu_stop = True
                    elif increase == 0 and it < (maxiter + 1):
                        mu_k /= resol
                    elif increase == 0 and it == (maxiter + 1):
                        mu_k *= resol
                        theta = last_theta
                        mu_stop = True

            attempt += 1

        Theta_hat[j, :] = theta

    return Theta_hat


def source_estimator(Xk, Yk, lambda_tilde_k):
    nk = Xk.shape[0]
    lasso = Lasso(alpha=lambda_tilde_k, fit_intercept=False, tol=1e-13)
    lasso.fit(Xk, Yk)
    beta_k_hat = lasso.coef_

    Sigma_k_hat = (Xk.T @ Xk) / nk
    Theta_k_hat = inverse_linfty(Sigma_k_hat, nk)
    residuals_k = Yk - Xk @ beta_k_hat
    correction_k = (Theta_k_hat @ (Xk.T @ residuals_k)) / nk

    beta_tilde_k = beta_k_hat + correction_k

    return beta_tilde_k


def DTransFusion(X_tilde, Y_tilde, XK, YK, q_tilde, lambda_tilde, P, n):
    co_training = WeightedLasso(alpha=n / X_tilde.shape[0], fit_intercept=False, tol=1e-13, weights=q_tilde.ravel(), max_iter=int(5e6), max_epochs=int(5e8))
    co_training.fit(X_tilde, Y_tilde)
    theta_hat = co_training.coef_

    w_hat = P @ theta_hat

    debias = Lasso(alpha=lambda_tilde, fit_intercept=False, tol=1e-13, max_iter=int(5e6), max_epochs=int(5e8))
    debias.fit(XK, YK - XK @ w_hat)
    delta_hat = debias.coef_

    betaK_hat = w_hat + delta_hat

    return theta_hat, delta_hat, betaK_hat


def PretrainedLasso(X, Y, XK, YK, w_tilde, lambda_K, rho, n, nK):
    model_sh = WeightedLasso(alpha=1/n, fit_intercept=False, tol=1e-13, weights=w_tilde.ravel())
    model_sh.fit(X, Y)
    beta_sh = model_sh.coef_

    if rho == 0.0:
        w_ = 1e15
    else:
        w_ = lambda_K / rho


    a_tilde = np.where(beta_sh == 0.0, w_, lambda_K).astype(float)
    a_tilde[0] = 0.0 

    residual = YK - (1 - rho) * (XK @ beta_sh)
    model_indiv = WeightedLasso(alpha=1/nK, fit_intercept=False, tol=1e-13, weights=a_tilde)
    model_indiv.fit(XK, residual)
    betaK_indiv = model_indiv.coef_

    betaK = (1 - rho) * beta_sh + betaK_indiv

    return beta_sh, betaK_indiv, betaK


def Pretrain_Lasso(X, Y, lambda_sh):
    model_shared = Lasso(alpha=(lambda_sh / X.shape[0]), fit_intercept=True, tol=1e-13)
    model_shared.fit(X, Y)

    beta_sh = np.concatenate(([model_shared.intercept_], model_shared.coef_))

    return beta_sh

# Pretrained Lasso
def Finetune_Lasso(XK, YK, beta_sh, a_tilde, rho, nK):

    residual = YK - (1 - rho) * (XK @ beta_sh)
    model_indiv = WeightedLasso(alpha=1/ nK, fit_intercept=False, tol=1e-13, weights=a_tilde.ravel())
    model_indiv.fit(XK, residual)

    beta_indiv = model_indiv.coef_

    betaK = (1 - rho) * beta_sh + beta_indiv

    return beta_indiv, betaK
