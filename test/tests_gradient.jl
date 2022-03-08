
using Zygote
using InverseProblem
const IP = InverseProblem

niter = 5
Regx = IP.homogenedgepreserving(1e3, :v2)
for i in 1:niter
    x = randn(10,10)
    g = zeros(size(x))
    vext = max(abs(minimum(x)), maximum(x))
    figure()
    plt.subplot(121)
    plt.imshow(x, cmap="seismic", aspect="auto", interpolation="none")

    IP.call!(Regx, x, g)
    g_zyg = gradient(y -> IP.call(Regx, y), x)[1]

    plt.subplot(122)
    plt.imshow(g - g_zyg, cmap="seismic", aspect="auto", interpolation="none",
               vmin=-vext, vmax=vext)
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
