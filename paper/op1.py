from lib import *

brca = np.load('data/BRCA_counts.npy')
n_iter = 500
layers = 10
fig_op1_brca_cv = pd.DataFrame()
lst = {}
kf = KFold(n_splits=5, random_state=1, shuffle=True)

for k in range(1, 2):
    lst["dnmf_0"] = []
    lst["dnmf_1"] = []
    lst["dnmf_2"] = []
    lst["op1_0"] = []
    lst["op1_1"] = []
    lst["op1_2"] = []
    lst["mu_0"] = []
    lst["mu_1"] = []
    lst["mu_2"] = []
    V = brca.T
    for train_index, test_index in kf.split(V.T):
        W = abs(np.random.randn(V.shape[0], 12))  # (f,k) normal
        H = abs(np.random.randn(12, V.shape[1]))  # (k,n) normal

        data, n_components, features, samples = util.build_data(
            V, W, H, index=train_index
        )

        for lam in range(3):
            L1 = lam
            L2 = lam
            ##################### unsupervised performance #############################
            _, dnmf_error, _, _ = util.train_unsupervised(
                data, layers, n_iter, n_components, l_1=L1, l_2=L2
            )
            _, op1_error, _, _ = util.train_unsupervised_opt1(
                data, layers, n_iter, n_components, l_1=L1, l_2=L2
            )
            ###### MU ################
            # train
            h_mu = data.h_0_train.mat.copy()  # k*n
            w_mu = data.w_init.mat.copy()  # f*k
            for i in range(n_iter):
                w_mu, h_mu = util.mu_update(
                    data.v_train.mat, w_mu, h_mu, l_1=L1, l_2=L2)
            # test
            mu_test_iter = 10
            h_mu_test = data.h_0_test.mat.copy()
            for i in range(mu_test_iter):
                _, h_mu_test = util.mu_update(
                    data.v_test.mat, w_mu, h_mu_test, update_W=False
                )
            mu_error = util.cost_mat(data.v_train.mat, w_mu, h_mu)

            lst[f"dnmf_{lam}"].append(dnmf_error[-1])
            lst[f"op1_{lam}"].append(op1_error[-1])
            lst[f"mu_{lam}"].append(mu_error)

    fig_op1_brca_cv[f"dnmf_0_{k}"] = lst["dnmf_0"]
    fig_op1_brca_cv[f"dnmf_1_{k}"] = lst["dnmf_1"]
    fig_op1_brca_cv[f"dnmf_2_{k}"] = lst["dnmf_2"]
    fig_op1_brca_cv[f"op1_0_{k}"] = lst["op1_0"]
    fig_op1_brca_cv[f"op1_1_{k}"] = lst["op1_1"]
    fig_op1_brca_cv[f"op1_2_{k}"] = lst["op1_2"]
    fig_op1_brca_cv[f"mu_0_{k}"] = lst["mu_0"]
    fig_op1_brca_cv[f"mu_1_{k}"] = lst["mu_1"]
    fig_op1_brca_cv[f"mu_2_{k}"] = lst["mu_2"]
fig_op1_brca_cv.to_csv('data/outputs/fig_op1_brca_cv.csv', index=False)


def method(row):
    if row.variable.startswith('dnmf'):
        return 'DNMF'
    elif row.variable.startswith('op1'):
        return 'op1'
    else:
        return 'MU'

#%%
fig_op1_brca_cv = pd.read_csv('../data/outputs/fig_op1_brca_cv.csv')
for x in range(1, 2):
    df = fig_op1_brca_cv[[f"dnmf_0_{x}", f"op1_0_{x}", f"mu_0_{x}", f"dnmf_1_{x}",
                           f"op1_1_{x}", f"mu_1_{x}", f"dnmf_2_{x}", f"op1_2_{x}", f"mu_2_{x}"]]
    # df = fig_op1_brca_cv[[f"dnmf_0_{x}",  f"mu_0_{x}", f"dnmf_1_{x}",
    #                        f"mu_1_{x}", f"dnmf_2_{x}",  f"mu_2_{x}"]]
    df = pd.melt(df)
    df['method'] = df.apply(
        lambda row: method(row), axis=1)
    if x > 9:
        df['lambda'] = df.apply(lambda row: row.variable[-4:-3], axis=1)
    else:
        df['lambda'] = df.apply(lambda row: row.variable[-3], axis=1)
    df['value'] = np.log(df['value'].values)
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    sns.boxplot(x='lambda', y='value', data=df, hue='method')
    plt.ylabel("$\log({MSE})$")
    plt.xlabel("${\lambda_1,\lambda_2}$")
    plt.legend([], [], frameon=False)
    plt.savefig(f"opt1.pdf")
    plt.show()
