
using Zygote
using InverseProblem
const IP = InverseProblem
using PyPlot
const plt = PyPlot

Regx = IP.edgepreserving(1e3, :v2)#IP.homogenedgepreserving(1e3, :v2)

niter = 5
nlin = round(Int, niter/2)
ncol = niter - nlin
fig, axs = plt.subplots(nrows=nlin, ncols=ncol)
fig.suptitle("residuals between Zygote and custom gradients, variance = 1")
for i in 1:nlin
    for j in 1:ncol
        x = randn(10,10)
        g = zeros(size(x))
        # axs[i,j].imshow(x, cmap="seismic", aspect="auto", interpolation="none")
        
        IP.call!(Regx, x, g)
        g_zyg = gradient(y -> IP.call(Regx, y), x)[1]
        
        res_g = g - g_zyg
        vext = max(abs(minimum(res_g)), maximum(res_g))
        im = axs[i,j].imshow(res_g, cmap="seismic", aspect="auto", interpolation="none",
                             vmin=-vext, vmax=vext)
        fig.colorbar(im, ax=axs[i,j])
    end
end

gridx = -10:10
gridy = -10:10
mu = 1e0
τ = 1e-3
edgpres = zeros(length(gridx), length(gridy))
hedgpres = zeros(length(gridx), length(gridy))
for i in 1:length(gridx)
    x = gridx[i]
    for j in 1:length(gridy)
        y = gridy[j]
        edgpres[i,j] = mu*(sqrt(x^2 + τ^2) + sqrt(y^2 + τ^2))
        hedgpres[i,j] = mu*(sqrt(x^2 + τ^2*(x^2+y^2)) + sqrt(y^2 + τ^2*(x^2+y^2)))
    end
end
figure()
plt.imshow(edgpres, aspect="auto", interpolation="none")
figure()
plt.imshow(hedgpres, aspect="auto", interpolation="none")
