
using Zygote
using InverseProblem
const IP = InverseProblem
using PyPlot
const plt = PyPlot

Regx = IP.edgepreserving(1e3, :v3)##IP.homogenedgepreserving(1e3, :v3)#IP.homogenedgepreserving(1e3, :v2)

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

# 1D version
stp = 1e-2
gridx = -10:stp:10
mu = 1e0
gridτ = 10.0 .^(-0.5:0.5:2)#10.0 .^(-2.5:0.5:0)#
tik = zeros(length(gridx), length(gridτ))
tv = zeros(length(gridx), length(gridτ))
edgpres = zeros(length(gridx), length(gridτ))
hedgpres = zeros(length(gridx), length(gridτ))
fig, axs = plt.subplots(nrows=2, ncols=3)
for t in 1:length(gridτ)
    ρ = sqrt(1/(length(gridx)))*gridτ[t]
    ρ2 = ρ^2
    α = 2*ρ
    β = 1
    for i in 1:length(gridx)
        # xm = (i == 1 ? gridx[1] - stp : gridx[i-1])
        x = gridx[i]
        norm_x = 10.0#abs(x)
        dx = x #- xm
        tik[i,t] = mu*dx^2
        tv[i,t] = mu*abs(dx)
        edgpres[i,t] = mu*α*(sqrt(dx^2 + ρ2) - ρ)
        hedgpres[i,t] = mu*α*norm_x^β*(sqrt(dx^2 + ρ2*norm_x^2) - ρ*norm_x)
    end
    
    axs[t].set_title("τ = $(gridτ[t])")
    axs[t].plot(gridx, tik[:,t], color="black", linestyle="dashed")
    axs[t].plot(gridx, tv[:,t], color="black", linestyle="dotted")
    axs[t].plot(gridx, edgpres[:,t])
    axs[t].plot(gridx, hedgpres[:,t])

end

figure()
plt.title("Edge-Preserving")
for t in 1:length(gridτ)
    plt.plot(gridx, edgpres[:,t], label="τ = $(gridτ[t])")
end
plt.legend()

figure()
plt.title("Homogeneous Edge-Preserving")
for t in 1:length(gridτ)
    plt.plot(gridx, hedgpres[:,t], label="τ = $(gridτ[t])")
end
plt.legend()

cols = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
figure()
plt.title("Edge-Preserving (dashed) and Homogeneous Edge-Preserving")
for t in 1:length(gridτ)
    plt.plot(gridx, edgpres[:,t], linestyle="dashed", color=cols[t])
    plt.plot(gridx, hedgpres[:,t], label="τ = $(gridτ[t])", color=cols[t])
end
plt.legend()


# 2D version
stp = 1e-2
gridx = -1:stp:1
gridy = -1:stp:1
mu = 1e0
gridτ = 10.0 .^(-5:0)
tik = zeros(length(gridx), length(gridy), length(gridτ))
tv = zeros(length(gridx), length(gridy), length(gridτ))
edgpres = zeros(length(gridx), length(gridy), length(gridτ))
hedgpres = zeros(length(gridx), length(gridy), length(gridτ))
for t in 1:length(gridτ)
    ρ = sqrt(2/(length(gridx)*length(gridy)))*gridτ[t]
    ρ2 = ρ^2
    β = 1
    for i in 1:length(gridx)
        # xm = (i == 1 ? gridx[1] - stp : gridx[i-1])
        x = gridx[i]
        for j in 1:length(gridy)
            # ym = (j == 1 ? gridy[1] - stp : gridy[j-1])
            y = gridy[j]
            norm_xy = sqrt(x^2+y^2)
            dx = x #- xm
            dy = y #- ym
            tik[i,j,t] = mu*(dx^2 + dy^2)
            tv[i,j,t] = mu*abs(dx + dy)
            edgpres[i,j,t] = 2*mu*ρ*norm_xy^β*(sqrt(dx^2 + dy^2 + ρ2) - ρ)
            hedgpres[i,j,t] = 2*mu*ρ*norm_xy^β*(sqrt(dx^2 + dy^2 + ρ2*norm_xy^2) - ρ*norm_xy)
        end
    end
    
    # fig = plt.subplots()
    # plt.subplot(221)
    # plt.imshow(tik[:,:,t], aspect="auto", interpolation="none")
    # plt.subplot(222)
    # plt.imshow(tv[:,:,t], aspect="auto", interpolation="none")
    # plt.subplot(223)
    # plt.imshow(edgpres[:,:,t], aspect="auto", interpolation="none")
    # plt.subplot(224)
    # plt.imshow(hedgpres[:,:,t], aspect="auto", interpolation="none")
    
    figure()
    plt.plot(gridx, tik[11,:,t], color="black", linestyle="dashed")
    plt.plot(gridx, tv[11,:,t], color="black", linestyle="dotted")
    plt.plot(gridx, edgpres[11,:,t])
    plt.plot(gridx, hedgpres[11,:,t])

end

figure()
plt.plot(edgpres[11,:,1])
plt.plot(edgpres[11,:,end])

figure()
plt.plot(hedgpres[11,:,1])
plt.plot(hedgpres[11,:,end])


