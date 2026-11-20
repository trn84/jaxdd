# The reference for jaxdd's ions bounded by their sites (docs/model.md, docs/validation.md):
# ChargeTransport.jl's Ex103_PSC_IVMeasurement cell (TiO2 | MAPI | spiro-OMeTAD, its default
# parameter set) with mobile vacancies in the perovskite bounded by their sites (Fermi–Dirac of
# order −1), changed where jaxdd's model has less: electrons and holes with Boltzmann
# statistics (Ex103: Fermi–Dirac 1/2) and one density of states for both bands of a layer (the
# conduction band's), and the vacancies' mean density and sites set from outside.
#
#     cd validation/chargetransport
#     julia --project=. -e 'using Pkg; Pkg.instantiate()'
#     CA=1e24 NA=1e25 julia --project=. psc_ions.jl OUTDIR eq dark:0.6 light:0.9 ...
#
# CA the vacancies' mean density, NA their sites (m^-3; Ex103's are 1.6e25 and 1e27, where
# jaxdd's steady state does not converge: PLAN.md). Cases: eq (thermodynamic equilibrium),
# dark:V and light:V (uniform generation in the perovskite), V at the acceptor (spiro) side;
# tstep:dark:V a transient after the voltage's step from equilibrium (below; a lit step from the
# dark ChargeTransport does not take).
# Writes OUTDIR/<case>.csv (region, x, psi, n, p, na on each region's nodes, an interface's
# node in both; m, V, m^-3; the first line the terminal current and the vacancies' mean) and
# OUTDIR/params.txt.

using ChargeTransport
using ExtendableGrids
using Printf

const q = ChargeTransport.constants.q
const eV = q

Base.@kwdef struct P
    iphin::Int = 1
    iphip::Int = 2
    iphia::Int = 3
    numberOfCarriers::Int = 3
    regionDonor::Int = 1
    regionIntrinsic::Int = 2
    regionAcceptor::Int = 3
    numberOfRegions::Int = 3
    bregionDonor::Int = 1
    bregionAcceptor::Int = 2
    bregionJ1::Int = 3
    bregionJ2::Int = 4
    numberOfBoundaryRegions::Int = 4
    h::Vector{Float64} = [100.0e-9, 400.0e-9, 200.0e-9]
    T::Float64 = 298.0
    En::Vector{Float64} = [-4.0, -3.7, -3.4] .* eV
    Ep::Vector{Float64} = [-5.8, -5.4, -5.1] .* eV
    Ea::Vector{Float64} = [0.0, -4.437, 0.0] .* eV        # the start of its search
    Nn::Vector{Float64} = [5.0e25, 8.1e24, 5.0e25]
    Np::Vector{Float64} = [5.0e25, 8.1e24, 5.0e25]        # jaxdd: one N_c for both bands
    Na::Vector{Float64} = [0.0, 1.0e27, 0.0]
    μn::Vector{Float64} = [3.89e-4, 6.62e-3, 3.89e-5]
    μp::Vector{Float64} = [3.89e-4, 6.62e-3, 3.89e-5]
    μa::Vector{Float64} = [0.0, 3.93e-16, 0.0]
    ε::Vector{Float64} = [10.0, 24.1, 3.0]
    r0::Vector{Float64} = [6.8e-17, 3.6e-18, 6.3e-17]
    τn::Vector{Float64} = [1.0e100, 3.0e-9, 1.0e100]
    τp::Vector{Float64} = [1.0e100, 3.0e-7, 1.0e100]
    EI::Vector{Float64} = [-5.0, -4.55, -4.1] .* eV
    G::Vector{Float64} = [0.0, 2.5e27, 0.0]
    Cn::Float64 = 1.0e24
    Cp::Float64 = 1.0e24
    Ca::Float64 = 1.6e25
end

"""Nodes refined geometrically towards each interface and contact, from `hmin` there to a
tenth of the half layer in its middle (275 nodes)."""
function coordinates(p::P; hmin = 1.0e-10)
    edges = cumsum([0.0; p.h])
    coord = Float64[]
    for k in 1:3
        a, b = edges[k], edges[k + 1]
        mid = 0.5 * (a + b)
        seg = glue(geomspace(a, mid, hmin, (mid - a) / 10), geomspace(mid, b, (b - mid) / 10, hmin))
        coord = isempty(coord) ? seg : glue(coord, seg)
    end
    return coord
end

function build(p::P)
    coord = coordinates(p)
    grid = simplexgrid(coord)
    edges = cumsum([0.0; p.h])
    cellmask!(grid, [edges[1]], [edges[2]], p.regionDonor, tol = 1.0e-18)
    cellmask!(grid, [edges[2]], [edges[3]], p.regionIntrinsic, tol = 1.0e-18)
    cellmask!(grid, [edges[3]], [edges[4]], p.regionAcceptor, tol = 1.0e-18)
    bfacemask!(grid, [0.0], [0.0], p.bregionDonor, tol = 1.0e-18)
    bfacemask!(grid, [edges[4]], [edges[4]], p.bregionAcceptor, tol = 1.0e-18)
    bfacemask!(grid, [edges[2]], [edges[2]], p.bregionJ1, tol = 1.0e-18)
    bfacemask!(grid, [edges[3]], [edges[3]], p.bregionJ2, tol = 1.0e-18)

    data = Data(grid, p.numberOfCarriers)
    data.modelType = Transient
    data.bulkRecombination = set_bulk_recombination(; iphin = p.iphin, iphip = p.iphip,
        bulk_recomb_Auger = false, bulk_recomb_radiative = true, bulk_recomb_SRH = true)
    data.boundaryType[p.bregionAcceptor] = OhmicContact
    data.boundaryType[p.bregionDonor] = OhmicContact
    # Uniform generation, its rate switched (`lit!`): the equilibrium solve generates whenever
    # the rate is not 0, whatever λ2, and the model set after the system is built gives NaN.
    data.generationModel = GenerationUniform
    enable_ionic_carrier!(data, ionicCarrier = p.iphia, regions = [p.regionIntrinsic])

    params = Params(p.numberOfRegions, p.numberOfBoundaryRegions, p.numberOfCarriers)
    params.temperature = p.T
    params.chargeNumbers[p.iphin] = -1
    params.chargeNumbers[p.iphip] = 1
    params.chargeNumbers[p.iphia] = 1
    for r in 1:p.numberOfRegions
        params.dielectricConstant[r] = p.ε[r] * ChargeTransport.constants.ε_0
        params.densityOfStates[p.iphin, r] = p.Nn[r]
        params.densityOfStates[p.iphip, r] = p.Np[r]
        params.densityOfStates[p.iphia, r] = p.Na[r]
        params.bandEdgeEnergy[p.iphin, r] = p.En[r]
        params.bandEdgeEnergy[p.iphip, r] = p.Ep[r]
        params.bandEdgeEnergy[p.iphia, r] = p.Ea[r]
        params.mobility[p.iphin, r] = p.μn[r]
        params.mobility[p.iphip, r] = p.μp[r]
        params.mobility[p.iphia, r] = p.μa[r]
        params.recombinationRadiative[r] = p.r0[r]
        params.recombinationSRHLifetime[p.iphin, r] = p.τn[r]
        params.recombinationSRHLifetime[p.iphip, r] = p.τp[r]
        params.recombinationSRHTrapDensity[p.iphin, r] = trap_density(p.iphin, r, params, p.EI[r], ChargeTransport.constants)
        params.recombinationSRHTrapDensity[p.iphip, r] = trap_density(p.iphip, r, params, p.EI[r], ChargeTransport.constants)
        params.generationUniform[r] = 0.0  # dark; `lit!`
    end
    params.doping[p.iphin, p.regionDonor] = p.Cn
    params.doping[p.iphip, p.regionAcceptor] = p.Cp
    params.doping[p.iphia, p.regionIntrinsic] = p.Ca
    data.params = params
    return System(grid, data, unknown_storage = :sparse), coord
end

function write_case(dir, name, ctsys, sol, p::P, J, mean)
    grid = ctsys.fvmsys.grid
    ipsi = ctsys.fvmsys.physics.data.index_psi
    open(joinpath(dir, "$(name).csv"), "w") do io
        println(io, "# J = $(J) A/m^2, mean vacancy density = $(mean) m^-3")
        println(io, "region,x,psi,n,p,na")
        for r in 1:p.numberOfRegions
            sub = subgrid(grid, [r])
            x = vec(sub[Coordinates])
            psi = view(sol[ipsi, :], sub)
            n = get_density(sol, r, ctsys, p.iphin)
            pp = get_density(sol, r, ctsys, p.iphip)
            na = r == p.regionIntrinsic ? get_density(sol, r, ctsys, p.iphia) : zeros(length(x))
            for i in eachindex(x)
                @printf(io, "%d,%.10e,%.10e,%.10e,%.10e,%.10e\n", r, x[i], psi[i], n[i], pp[i], na[i])
            end
        end
    end
end

function main(dir, cases)
    mkpath(dir)
    Ca = parse(Float64, get(ENV, "CA", "1.6e25"))
    Na = parse(Float64, get(ENV, "NA", "1e27"))
    p = P(Ca = Ca, Na = [0.0, Na, 0.0])
    run(dir, p, Na, cases)
    print(read(joinpath(dir, "params.txt"), String))
end

"""The generation in the perovskite on (`true`) or off."""
function lit!(ctsys, p::P, on)
    for params in (ctsys.fvmsys.physics.data.params, ctsys.data.params)
        params.generationUniform .= on ? p.G : 0.0 .* p.G
    end
end

"""ChargeTransport's equilibrium, the vacancies' number exact: from Fermi–Dirac 1/2's solution
(from Boltzmann's start the solve does not converge, its homotopy from the linear Poisson
problem), the vacancies' energy refined by secants on ln(mean) to 1e-12 (ChargeTransport rounds
it to 0.1 meV and stops at a relative 1e-4 of the mean; jaxdd holds the number exactly, and
4e-5 of 4e24 m^-3 moved the potential by a millivolt). With the generation on it generates too
(its ramp of λ2): no equilibrium, but a start with the vacancies' number."""
function equilibrium!(ctsys, data, p::P, control, mean_of)
    data.F[p.iphin] = FermiDiracOneHalfTeSCA
    data.F[p.iphip] = FermiDiracOneHalfTeSCA
    sol = equilibrium_solve!(ctsys, control = control, vacancyEnergyCalculation = true)
    data.F[p.iphin] = Boltzmann
    data.F[p.iphip] = Boltzmann
    sol = equilibrium_solve!(ctsys, inival = sol, control = control, vacancyEnergyCalculation = true)
    function solved_at(E)
        data.params.bandEdgeEnergy[p.iphia, p.regionIntrinsic] = E * q
        s = equilibrium_solve!(ctsys, inival = sol, control = control, vacancyEnergyCalculation = false)
        return s, mean_of(s)
    end
    E0 = data.params.bandEdgeEnergy[p.iphia, p.regionIntrinsic] / q
    E1 = E0 + 1.0e-4
    _, m0 = solved_at(E0)
    s1, m1 = solved_at(E1)
    for _ in 1:30
        abs(m1 / p.Ca - 1) < 1.0e-12 && break
        E2 = E1 - (log(m1) - log(p.Ca)) * (E1 - E0) / (log(m1) - log(m0))
        E0, m0, sol = E1, m1, s1
        s1, m1 = solved_at(E2)
        E1 = E2
    end
    return s1
end

function run(dir, p::P, Na, cases)
    ctsys, coord = build(p)
    data = ctsys.fvmsys.physics.data
    control = SolverControl()
    control.max_round = 5
    control.damp_initial = 0.1
    control.damp_growth = 1.21
    mean_of(s) = integrated_density(ctsys, sol = s, icc = p.iphia, ireg = p.regionIntrinsic) / data.regionVolumes[p.regionIntrinsic]

    sol = equilibrium!(ctsys, data, p, control, mean_of)
    open(joinpath(dir, "params.txt"), "w") do io
        println(io, "ChargeTransport $(pkgversion(ChargeTransport)), CA = $(p.Ca), NA = $(Na)")
        println(io, "vacancy energy E_a = $(data.params.bandEdgeEnergy[p.iphia, p.regionIntrinsic] / q) eV")
        println(io, "mean vacancy density = $(mean_of(sol)) m^-3")
        println(io, "nodes = $(length(coord)), smallest interval = $(minimum(diff(coord))) m")
    end
    "eq" in cases && write_case(dir, "eq", ctsys, sol, p, 0.0, mean_of(sol))
    # Dark from equilibrium. Lit, ChargeTransport does not take the generation's step from the
    # dark (NaN at every step size, from 1e-3 of it): its steady states start from its
    # equilibrium solve with the generation on, which ramps it, the vacancies' number held.
    dark = filter(c -> !occursin("light", c), cases)
    lit = filter(c -> occursin("light", c), cases)
    transients(dir, p, ctsys, data, sol, dark, mean_of)
    if !isempty(lit)
        lit!(ctsys, p, true)
        transients(dir, p, ctsys, data, equilibrium!(ctsys, data, p, control, mean_of), lit, mean_of)
    end
end

"""A system's steady states at a voltage and its transients, from equilibrium `eqsol`."""
function transients(dir, p::P, ctsys, data, eqsol, cases, mean_of)
    # Steady states at a voltage, dark or lit: from equilibrium the voltage stepped and held
    # until nothing moves. ChargeTransport has no stationary solve with mobile ions (their
    # quasi-Fermi potential is fixed by their number only in time), so the steady state is the
    # transient's end, at 1e4 s; the ions' mobility raised to 1e-12 m^2/Vs (it sets how fast
    # they settle, not where).
    for case in cases
        case == "eq" && continue
        startswith(case, "tstep") && continue
        data.params.mobility[p.iphia, p.regionIntrinsic] = 1.0e-12
        _, V = split(case, ":")
        set_contact!(ctsys, p.bregionAcceptor, Δu = parse(Float64, V))
        tc = SolverControl()
        tc.max_round = 5
        tc.damp_initial = 0.5
        tc.damp_growth = 1.21
        tc.Δt = 1.0e-12
        tc.Δt_min = 1.0e-16
        tc.Δt_grow = 1.3
        tc.Δt_max = 10.0
        tc.Δu_opt = 100.0   # the step's change unbounded: the voltage's step is the start
        tsol = ChargeTransport.solve(ctsys, inival = eqsol, times = (0.0, 1.0e4), control = tc)
        last, before = tsol.u[end], tsol.u[end - 1]
        J = get_current_val(ctsys, last, before, tsol.t[end] - tsol.t[end - 1])
        change = maximum(abs.(last .- before))
        change < 1.0e-10 || error("$(case): not settled, last step's change $(change)")
        write_case(dir, replace(case, ":" => "_"), ctsys, last, p, J, mean_of(last))
        println(case, ": J = $(J) A/m^2, last step's largest change $(change)")
    end

    # Transients: from equilibrium the voltage stepped at t = 0 and held to TEND (s; 100), the
    # ions' mobility MUA (m^2/Vs; 1e-12, about 6 s to settle over the perovskite). Writes the
    # current at each of ChargeTransport's own times (implicit Euler, its steps grown by 1.3
    # from 1e-12 s, at most 10 s) and the states nearest 1e-6, 1e-3 and 1 s.
    for case in cases
        startswith(case, "tstep") || continue
        _, _, V = split(case, ":")
        data.params.mobility[p.iphia, p.regionIntrinsic] = parse(Float64, get(ENV, "MUA", "1e-12"))
        tend = parse(Float64, get(ENV, "TEND", "100"))
        set_contact!(ctsys, p.bregionAcceptor, Δu = parse(Float64, V))
        tc = SolverControl()
        tc.max_round = 5
        tc.damp_initial = 0.5
        tc.damp_growth = 1.21
        tc.Δt = 1.0e-12
        tc.Δt_min = 1.0e-16
        tc.Δt_grow = 1.3
        tc.Δt_max = 10.0
        tc.Δu_opt = 100.0
        tsol = ChargeTransport.solve(ctsys, inival = eqsol, times = (0.0, tend), control = tc)
        name = replace(case, ":" => "_")
        open(joinpath(dir, "$(name)_J.csv"), "w") do io
            println(io, "t,J,mean")
            for k in 2:length(tsol.t)
                J = get_current_val(ctsys, tsol.u[k], tsol.u[k - 1], tsol.t[k] - tsol.t[k - 1])
                @printf(io, "%.17e,%.12e,%.12e\n", tsol.t[k], J, mean_of(tsol.u[k]))
            end
        end
        for (j, at) in enumerate((1.0e-6, 1.0e-3, 1.0))
            at > tend && continue
            k = argmin(abs.(log.(max.(tsol.t, 1.0e-300)) .- log(at)))
            J = get_current_val(ctsys, tsol.u[k], tsol.u[k - 1], tsol.t[k] - tsol.t[k - 1])
            write_case(dir, "$(name)_at$(j)", ctsys, tsol.u[k], p, J, mean_of(tsol.u[k]))
            open(joinpath(dir, "$(name)_at$(j)_t.txt"), "w") do io
                @printf(io, "%.17e\n", tsol.t[k])
            end
        end
        println(case, ": ", length(tsol.t), " times to ", tsol.t[end], " s")
    end
end

main(ARGS[1], length(ARGS) > 1 ? ARGS[2:end] : ["eq"])
