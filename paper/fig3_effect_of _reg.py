from lib import *
dataset = joblib.load('data/all_mutational.pkl.gz')
n_iter = 500
layers = 10
fig_3 = pd.DataFrame()
lst = {}

kf = KFold(n_splits=5, random_state=1, shuffle=True)

for k, v in dataset.items():
    lst["super_no_reg"] = []
    lst["super_reg"] = []
    for train_index, test_index in kf.split(v["V"].T):

        data, n_components, features, samples = util.build_data(
            v["V"], v["W"], v["H"], index=train_index
        )

        # Supervised
        dnmf_model_noreg, super_dnmf_train_loss_noreg, super_dnmf_test_loss_noreg = util.train_supervised(
            data, layers, n_iter, L1=False, L2=False
        )
        dnmf_model_loss_learn_L1_L2, super_dnmf_train_loss_learn_L1_L2, super_dnmf_test_loss_learn_L1_L2 = util.train_supervised(
            data, layers, n_iter, L1=True, L2=True
        )

        lst["super_no_reg"].append(super_dnmf_test_loss_noreg[-1])
        lst["super_reg"].append(super_dnmf_test_loss_learn_L1_L2[-1])

    fig_3[f"super_reg_{k}"] = lst["super_reg"]
    fig_3[f"super_no_reg{k}"] = lst["super_no_reg"]
fig_3.to_csv('data/outputs/fig_3.csv', index=False)

# %%
fig_3 = pd.read_csv('../data/outputs/fig_3.csv')
for x in range(1, 13):
    util.plot_box(
        [f"super_reg_{x}", f"super_no_reg{x}"],
        ["Regularized", "Not-Regularized"],
        f"fig3/compare_reg_{x}",
        fig_3,
        'Supervised',
        None,
        '$\log({MSE})$',
    )