from lib import *
# Comparative  performance  on  simulated  data  in  the  supervised
dataset = joblib.load('data/all_mutational.pkl.gz')
n_iter = 500
layers = 10
fig_5 = pd.DataFrame()
lst = {}
kf = KFold(n_splits=5, random_state=1, shuffle=True)

for k, v in dataset.items():
    lst["dnmf"] = []
    lst["dnmf_w"] = []
    lst["mu_0"] = []
    lst["mu_1"] = []
    lst["mu_2"] = []
    for train_index, test_index in kf.split(v["V"].T):

        data, n_components, features, samples = util.build_data(
            v["V"], v["W"], v["H"], index=train_index
        )
        # Supervised DNMF using real h
        _, _, dnmf_error = util.train_supervised(
            data, layers, n_iter, L1=True, L2=True)
        lst["dnmf"].append(dnmf_error[-1])

        # Supervised DNMF using real w
        _, _, dnmf_error_w = util.train_supervised_w(
            data, layers, n_iter, L1=True, L2=True)
        lst["dnmf_w"].append(dnmf_error_w[-1])

        # MU
        for lam in range(3):
            L1 = lam
            L2 = lam
            # Train
            mu_super_iter = 500
            w_mu = data.w_init.mat.copy()
            for i in range(mu_super_iter):
                w_mu, _ = util.mu_update(
                    data.v_train.mat, data.w.mat, data.h_train.mat, l_1=L1, l_2=L2, update_H=False
                )
            # inference
            mu_super_iter = 10
            h_mu_test = data.h_0_test.mat.copy()
            for i in range(mu_super_iter):
                _, h_mu_test = util.mu_update(
                    data.v_test.mat, w_mu, h_mu_test, l_1=L1, l_2=L2, update_W=False
                )
            mu_error = ((data.h_test.mat.T - h_mu_test.T) ** 2).sum()

            lst[f"mu_{lam}"].append(mu_error)

    fig_5[f"dnmf_{k}"] = lst["dnmf"]
    fig_5[f"dnmf_w_{k}"] = lst["dnmf_w"]
    fig_5[f"mu_0_{k}"] = lst["mu_0"]
    fig_5[f"mu_1_{k}"] = lst["mu_1"]
    fig_5[f"mu_2_{k}"] = lst["mu_2"]
fig_5.to_csv('data/outputs/fig_5_w.csv', index=False)

fig_5 = pd.read_csv('data/outputs/fig_5_w.csv')
for x in range(1, 13):
    util.plot_box(
        [f"dnmf_{x}", f"mu_0_{x}", f"mu_1_{x}", f"mu_2_{x}"],
        ["DNMF", "$MU_{\lambda_1=\lambda_2=0}$",
            "$MU_{\lambda_1=\lambda_2=1}$", "$MU_{\lambda_1=\lambda_2=2}$"],
        f"fig5/w_supervised_performance_{x}",
        fig_5,
        'Supervised',
        None,
        '$\log({MSE})$',
    )
