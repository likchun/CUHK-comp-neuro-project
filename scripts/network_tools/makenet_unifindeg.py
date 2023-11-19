from libs.mylib3 import *
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture



N = 1000 # number of neurons
p = 0.01 # connection probability
N_ei_ratio = 4 # neuron EI ratio
kin_ei_ratio = 4 # in-degree EI ratio = (num of EXC presynaptic neuron)/(num of INH presynaptic neuron)
gausfit_compo = 1 # number of Gaussian components used in out-degree fitting
rng_seed = 1



np.random.seed(rng_seed)
random.seed(rng_seed)
neurons = np.arange(N,dtype=int)
neutype = np.zeros(N,dtype=int)+1
neutype[:int(N/(1+N_ei_ratio))] = -1
adjmat = np.zeros((N,N))

def network_data():

    exc_pool = neurons[int(N/(1+N_ei_ratio)):]
    inh_pool = neurons[:int(N/(1+N_ei_ratio))]
    draw_exc = int(N*p*kin_ei_ratio/(1+kin_ei_ratio))
    draw_inh = int(N*p/(1+kin_ei_ratio))

    # choose EXC/INH inputs (controlled by "kin_ei_ratio") from EXC/INH neurons (controlled by "N_ei_ratio")
    presyn_inh_neuron_indices = [np.random.choice(np.delete(inh_pool, n), draw_inh, replace=False) if len(inh_pool)>n else np.random.choice(inh_pool, draw_inh, replace=False) for n in range(N)]
    presyn_exc_neuron_indices = [np.random.choice(np.delete(exc_pool, n-len(inh_pool)), draw_exc, replace=False) if len(inh_pool)<n+1 else np.random.choice(exc_pool, draw_exc, replace=False) for n in range(N)]

    return pd_dataframe(
        ["neuron_index", "neuron_type", "presyn_exc_neuron_indices", "presyn_inh_neuron_indices"],
        [neurons, neutype, presyn_exc_neuron_indices, presyn_inh_neuron_indices])

nd = network_data()

for i in range(N):
    for j_e in nd.loc[i]["presyn_exc_neuron_indices"]: adjmat[i][j_e] = +1
    for j_i in nd.loc[i]["presyn_inh_neuron_indices"]: adjmat[i][j_i] = -1


# end result is:
adjmat # <----
# ^^^^

# exit(0)



# plot section

net = NNetwork()
net.adjacency_matrix_from(adjmat)

# plt.imshow(adjmat, cmap="bwr")
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# fig, ax = plt.subplots()
# graphing.scatter_plot(*net.link_ij, c=adjmat.T[np.nonzero(adjmat.T)], ax=ax)
# exit(0)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

graphing.bar_chart_INT(net.in_degree_exc, c="r", a=.5, label="exc\nin-degree", ax=axes[0])
graphing.bar_chart_INT(net.in_degree_inh, c="b", a=.5, label="inh\nin-degree", ax=axes[0])

x, y = graphing.bar_chart_INT(net.out_degree, ax=axes[1])
if gausfit_compo == 1:
    fit_x = np.linspace(0, 20, int(20/0.01))
    mu, sigma = curve_fit(scipy.stats.norm.pdf, x, y/np.sum(y), p0=[np.mean(x),1], maxfev=2000)[0]
    graphing.line_plot(fit_x, scipy.stats.norm.pdf(fit_x, mu, sigma)*np.sum(y), style="--", c="g", label="mean={:.2f}\nS.D.={:.2f}".format(mu,sigma), ax=axes[1])
    axes[1].legend(fontsize=16, title="Gaussian fit", title_fontsize=16, loc="upper right")
elif gausfit_compo > 1:
    gmm = GaussianMixture(n_components=gausfit_compo).fit(net.out_degree.reshape(-1, 1))
    fit_x = net.out_degree.copy().ravel()
    fit_x.sort()
    plot_area_sum = np.dot(np.full(len(y), x[1]-x[0]), y)
    fit_y = [weight*scipy.stats.norm.pdf(fit_x, mean, np.sqrt(covar)).ravel() for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_)]
    [graphing.line_plot(fit_x, fit_y[-(i+1)]*plot_area_sum, style="--", c=c, label="mean{0}={1:.2f}\nS.D.{0}={2:.2f}".format(i+1,float(gmm.means_[1-i]),np.sqrt(float(gmm.covariances_[1-i]))), ax=axes[1]) for i, c in enumerate(["g","darkorange"])]
    axes[1].legend(fontsize=16, title="two Gaussian fit", title_fontsize=16, loc="upper right")
    # graphing.line_plot(fit_x, np.array(a).sum(axis=0)*plot_area_sum, style="--", c="g", label="two Gaussian fit\nmean1={:.2f}\nSD1={:.2f}".format(mu,sigma), ax=axes[1])

axes[0].set(xlabel="incoming degree", ylabel="number of occurrence")
axes[0].set_xlim(0, 10)
axes[0].legend(fontsize=16)
axes[1].set(xlabel="outgoing degree")
axes[1].set_xlim(0, None)
plt.tight_layout()



print("num of neurons:         {}".format(net.size))
print("connection probability: {}".format(net.connection_prob))
print("neuron E-I ratio:       {}".format(N_ei_ratio))
print("in-degree E-I ratio:    {}".format(kin_ei_ratio))
print("num of non-inh neurons: {}".format(len(net.neuron_type[net.neuron_type != "inh"])))
print("num of exc neurons:     {}".format(len(net.neuron_type[net.neuron_type == "exc"])))
print("num of inh neurons:     {}".format(len(net.neuron_type[net.neuron_type == "inh"])))
print("num of exc links:       {}".format(net.num_of_exc_links))
print("num of inh links:       {}".format(net.num_of_inh_links))
print("num of self-loop:       {}".format(len(np.diag(net.adjacency_matrix)[np.diag(net.adjacency_matrix) != 0])))
print("random number seed:     {}".format(rng_seed))


# plt.show()
# fig.savefig("net_deg")

# net.adjacency_matrix_to_file("matrix.txt")
