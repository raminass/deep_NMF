from lib import *

# The  effect  of  number  of  layers  on  algorithm’s  performance.  
# Eachline  corresponds  to  one  of  the  simulated  data  sets.

dataset = joblib.load('data/all_mutational.pkl.gz')
n_iter = 500
fig_4 = pd.DataFrame()
layers_list = [5, 10, 15, 20]
lst = {}
avg = {}
kf = KFold(n_splits=5, random_state=1, shuffle=True)

for layers in layers_list:
      avg[f"super_{layers}"] = []
      avg[f"unsuper_{layers}"] = []

for k, v in dataset.items():

    for layers in layers_list:
      lst[f"super_{layers}"] = []
      lst[f"unsuper_{layers}"] = []
    
    for train_index, test_index in kf.split(v["V"].T):

      data, n_components, features, samples = util.build_data(
          v["V"], v["W"], v["H"], index = train_index
      )

      for layers in layers_list:
        # Supervised
        _, super_train, super_test = util.train_supervised(
        data, layers, n_iter, L1=True, L2=True )
        ##################### unsupervised performance #############################
        _, unsuper_train, unsuper_test, _ = util.train_unsupervised(
            data, layers, n_iter, n_components
        )
        
        lst[f"super_{layers}"].append(super_test[-1])
        lst[f"unsuper_{layers}"].append(unsuper_test[-1])

    for m in layers_list:
      avg[f"super_{m}"].append(round(np.average(lst[f"super_{m}"])))
      avg[f"unsuper_{m}"].append(round(np.average(lst[f"unsuper_{m}"])))

fig_4 = pd.DataFrame.from_dict(avg)
fig_4.to_csv('data/outputs/fig_4.csv', index=False)

fig_4 = pd.read_csv('data/outputs/fig_4.csv')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
fig = plt.figure()
ax = fig.add_subplot(111)
x_axis = ['5', '10', '15', '20']
for i in range(fig_4.shape[0]):
  data1 = np.log(fig_4[['super_5', 'super_10', 'super_15', 'super_20']].loc[i])
  ax.plot(x_axis, data1,'ko-',label='line1')
ax.set_xlabel('$\# layers$')
ax.set_ylabel('$\log({MSE})$')
plt.savefig(f"plots/fig4/supervised.pdf")
plt.show()