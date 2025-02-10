Multimessenger follow-up as an MILP problem
===========================================

Here we document our representation of the multimessenger follow-up problem as a mixed integer linear program.

Problem 1: Fixed exposure time
------------------------------

We receive a :doc:`HEALPix probability sky map <userguide:tutorial/skymaps>` that describes the probability distribution of the true but unknown position of a target of interest as a function of position on the sky. There is a delay between the time that the event occurred and when we can start observations due to the time it takes to uplink commands to the spacecraft, and there is a deadline by which we must complete our observations.

Our telescope can observe any of a set of :math:`n_J` reference fields at predetermined sky locations in order to tile the sky map. For each reference field that we select, our telescope must visit the reference field at least :math:`n_K` times. We have a cadence requirement: each visit of a given reference field must occur at least a time :math:`\gamma` after the previous visit.

Every visit takes a certain amount of exposure time, and it takes a known amount of time to slew between different reference fields. We may only a visit a reference field when it is within the field of regard, the region that constrains where the telescope may point at any given instant of time.

Data preparation
^^^^^^^^^^^^^^^^

1. Construct a discrete 1D grid of times that stretch from the delayed start of observations up to the deadline.

2. Propagate the orbit of the spacecraft to calculate the position of the spacecraft at each time step.

3. For each reference field and each time step, test whether the reference field is within the instantaneous field of regard, creating an observability bit map.

4. Transform the observability bit map into a list of time segments during which each reference field is observable.

5. Discard segments that are shorter than the exposure time.

6. Discard reference fields that have no observable segments.

7. For each reference field, find the HEALPix pixel indices that are within the field's footprint.

8. Select the top 50 reference fields that contain the greatest probability.

9. Discard pixels that are not contained in any reference field.

10. Calculate the slew times between fields.

MILP problem formulation
^^^^^^^^^^^^^^^^^^^^^^^^

Index sets
""""""""""

- :math:`I = \{0, 1, \dots, n_I - 1\}`: pixels
- :math:`J = \{0, 1, \dots, n_J - 1\}`: reference fields
- :math:`K = \{0, 1, \dots, n_K - 1\}`: visits
- :math:`\left(M_j = \{0, 1, \dots, {n_M}_j\}\right)_{j \in J}`: observable segments for reference field :math:`j`
- :math:`\left(J_i = \{0, 1, \dots, {n_J}_i\}\right)_{i \in I}`: set of indices of fields that contain pixel :math:`i`

Parameters
""""""""""

- :math:`\left(\rho_i\right)_{i \in I}`: probability that source is inside pixel :math:`i`
- :math:`\left(\sigma_{jj^\prime}\right)_{j \in J, j^\prime \in J}`: slew time from reference field :math:`j` to reference field :math:`j^\prime`
- :math:`\left(\alpha_{jm}\right)_{j \in J, m \in M}`: start time of observable segment :math:`m` of reference field :math:`j`
- :math:`\left(\omega_{jm}\right)_{j \in J, m \in M}`: end time of observable segment :math:`m` of reference field :math:`j`
- :math:`\epsilon`: exposure time
- :math:`\gamma`: cadence, time between visits
- :math:`\beta`: delay from event time to start of observations
- :math:`\delta`: deadline from event time to end of last observation

Decision variables
""""""""""""""""""

Binary decision variables:

- :math:`\left(p_i\right)_{i \in I}`: pixel :math:`i` is inside the footprint of one or more selected reference fields
- :math:`\left(r_j\right)_{j \in J}`: reference field :math:`j` is selected for observation
- :math:`\left(s_{jkm}\right)_{j \in J, k \in K, m \in M \mid {n_M}_j > 1}`: whether reference field :math:`j` visit :math:`k` occurs in segment :math:`m`

Continuous decision variables:

- :math:`\left(t_{jk}\right)_{j \in J, k \in K}`: midpoint time of observation :math:`j` visit :math:`k`

Constraints
"""""""""""

**Containment.** Only count pixels that are inside one or more reference fields.

.. math::
    :label: fixed-exptime-constraint-containment

    \forall i :\quad p_i \leq \sum_{j \in J_i} r_j

**Cadence.** If a reference field is selected for observation, then enforce a minimum time between visits.

.. math::
    :label: fixed-exptime-constraint-cadence

    \forall k > 1 ,\; j :\quad t_{jk} - t_{j,k-1} \geq (\epsilon + \gamma) r_j

**No overlap.** Observations cannot overlap in time; they must be separated by at least the exposure time plus the slew time.

.. math::
    :label: fixed-exptime-constraint-no-overlap

    \forall j^\prime > j,\; k ,\; k^\prime :\quad \left|t_{jk} - t_{j^\prime k^\prime}\right|  \geq \left(\sigma_{jj^\prime} + \epsilon\right) \left( r_j + r_{j^\prime} - 1\right)

**Field of regard.** An observation of a reference field can only occur while the coordinates of the reference field are within the field of regard.

For fields that have one observable segment (:math:`{n_M}_j = 1`), this constraint is simply an inequality:

.. math::
    :label: fixed-exptime-constraint-for-one

    \forall j ,\; k \;, m \mid {n_M}_j = 1 :\quad \alpha_{jm} + \epsilon / 2 \leq t_{jk} \leq \omega_{jm} - \epsilon / 2

For fields that have more than one observable segment (:math:`{n_M}_j > 1`), we use the decision variable :math:`s_{jkm}` to determine which inequality is satisfied:

.. math::
    :label: fixed-exptime-constraint-for-many

    \begin{eqnarray}
    \forall j ,\; k \;, m \mid {n_M}_j > 1 :\quad s_{jkm} &=& 1 \;\Rightarrow\; \alpha_{jm} + \epsilon / 2 \leq t_{jk} \leq \omega_{jm} - \epsilon / 2, \\
    \sum_m s_{jkm} &\geq& 1
    \end{eqnarray}

Cuts
""""

**Total exposure time.** Although it is implied by other constraints, the constraint that the total exposure time cannot exceed the total available time is found to speed up the search.

.. math::
    :label: fixed-exptime-cut-total-time

    \sum_{j \in J} r_j \leq \frac{\delta - \beta}{\epsilon n_K}

Objective
"""""""""

Maximize the sum of the probability of all of the pixels that are contained within selected fields:

.. math::
    :label: fixed-exptime-objective

    \sum_{i \in I} \rho_i p_i

Problem 2: Variable exposure time
---------------------------------

In this variation, we have a sky map of the exposure time required to detect the source as a function of its position on the sky. We permit the exposure time to vary for each field. A given pixel counts toward the objective value only if the exposure time of a field that contains that pixel exceeds the pixel's exposure time.

MILP problem formulation
^^^^^^^^^^^^^^^^^^^^^^^^

Additional parameters
"""""""""""""""""""""

- :math:`\left(\epsilon_i\right)_{i \in I}`: minimum exposure time to detect a source in pixel :math:`i`
- :math:`\epsilon_\mathrm{min}`: minimum allowed exposure time
- :math:`\epsilon_\mathrm{max}`: maximum allowed exposure time

Additional decision variables
"""""""""""""""""""""""""""""

Semicontinuous decision variables:

- :math:`\left(e_j\right)_{j \in J}, \forall j \in J : e_j = 0 \textnormal{ or } \epsilon_\mathrm{min} \leq e_j \leq \epsilon_\mathrm{max} \;`: exposure time of field :math:`j`

Constraints
"""""""""""

The constraints are slightly different:

**Depth.** Only count pixels that are observed to sufficient depth.

.. math::
    :label: variable-exptime-constraint-depth

    \forall i \in I :\quad p_\mathrm{i} = 1 \Rightarrow \max_{j \in J_i} e_{j} \geq \epsilon_i

**Exposure time.** If a field's exposure time is nonzero, then it is selected for observation.

.. math::
    :label: variable-exptime-constraint-exptime

    \forall j \in J :\quad \epsilon_\mathrm{max} r_j \geq e_\mathrm{j}

**Cadence.** This is similar to Equation :eq:`fixed-exptime-constraint-cadence`, except that we replace the right-hand side of the inequality.

.. math::
    :label: variable-exptime-constraint-cadence

    \forall k > 1 ,\; j :\quad t_{jk} - t_{j,k-1} \geq \gamma r_j + e_j

**No overlap.** This is also similar to Equation :eq:`fixed-exptime-constraint-no-overlap`, except with a slightly different right-hand side.

.. math::
    :label: variable-exptime-constraint-no-overlap

    \forall j^\prime > j ,\; k ,\; k^\prime :\quad \left|t_{jk} - t_{j^\prime k^\prime}\right|  \geq \sigma_{jj^\prime} \left( r_j + r_{j^\prime} - 1\right) + (e_j + e_\mathrm{j^\prime}) / 2

**Field of regard.** This is similar to Equations :eq:`fixed-exptime-constraint-for-one` and :eq:`fixed-exptime-constraint-for-many`, except that we replace :math:`\epsilon` with :math:`e_j`.

For fields that have one observable segment:

.. math::
    :label: variable-exptime-constraint-for-one

    \forall j ,\; k \;, m \mid {n_M}_j = 1 :\quad \alpha_{jm} + e_j / 2 \leq t_{jk} \leq \omega_{jm} - e_j / 2

For fields that have more than one observable segment:

.. math::
    :label: variable-exptime-constraint-for-many

    \begin{eqnarray}
    \forall j ,\; k \;, m \mid {n_M}_j > 1 :\quad s_{jkm} &=& 1 \;\Rightarrow\; \alpha_{jm} + e_j / 2 \leq t_{jk} \leq \omega_{jm} - e_j / 2, \\
    \sum_m s_{jkm} &\geq& 1
    \end{eqnarray}

Additional cuts
"""""""""""""""

**Total exposure time.** Replace Equation :eq:`fixed-exptime-cut-total-time` with:

.. math::
    :label: variable-exptime-cut-total-time

    \begin{eqnarray}
    \sum_{j \in J} r_j &\leq& \frac{\delta - \beta}{\epsilon_\mathrm{min} n_K} \\
    \sum_{j \in J} e_j &\leq& \frac{\delta - \beta}{n_K}
    \end{eqnarray}

Objective
"""""""""

Same as above.

Problem 3: Variable exposure time with prior distribution of absolute magnitude
-------------------------------------------------------------------------------

In this variation, we don't know the precise absolute magnitude :math:`X` of the source. In the case of kilonovae, our prior knowledge about the absolute magnitude is scant; for the sake of mathematical convenience, we assume that the absolute magnitude has a normal distribution, :math:`X \sim~ \mathcal{N}[\mu_X, \sigma_X]`. We need to compute the distribution of *apparent* magnitudes :math:`x` in order to determine the probability of detection as a function of exposure time for each pixel.

Gravitational-wave sky maps provide the posterior distribution of distance, as a parametric ansatz distribution,

.. math::
    p(r) = \frac{N}{\sqrt{2 \pi}\sigma} \exp\left[-\frac{1}{2}\left(\frac{r - \mu}{\sigma}\right)^2\right] r^2,

with the location parameter :math:`\mu`, scale parameter :math:`\sigma`, and normalization :math:`N` tabulated for each pixel. This is an inconvenient distribution for integration, so instead we construct a log-normal distance distribution with the same mean and standard deviation as the ansatz distribution.

We calculate the mean :math:`m` and standard deviation :math:`s` from :math:`\mu` and :math:`\sigma` using the function :obj:`ligo.skymap.distance.parameters_to_moments`. Then, the location and scale parameters of the log-normal distribution are given by

.. math::
    :label: log-distance-parameters

    \begin{eqnarray}
    \mu_{\ln r} &=& \ln m - \frac{1}{2} \ln \left(1 + \frac{s^2}{m^2}\right) \\
    {\sigma_{\ln r}}^2 &=& \ln \left(1 + \frac{s^2}{m^2}\right).
    \end{eqnarray}

The logarithm of the distance then has the distribution :math:`\ln r \sim \mathcal{N}[\mu_{\ln r}, \sigma_{\ln r}]`. The apparent magnitude is related to the absolute magnitude through :math:`x = X + 5 \log_{10} r + 25`, assuming that :math:`r` is in the units of Mpc. Therefore the apparent magnitude has the distribution :math:`x \sim \mathcal{N}[\mu_x, \sigma_x]`, with

.. math::
    :label: appmag-parameters

    \begin{eqnarray}
    \mu_x &=& \mu_X + \left(\frac{5}{\ln 10}\right) \mu_{\ln r} + 25 \\
    {\sigma_x}^2 &=& {\sigma_{X}}^2 + \left(\frac{5}{\ln 10}\right)^2 {\sigma_{\ln r}}^2.
    \end{eqnarray}

With this Gaussian distribution of apparent magnitudes, we can now calculate the detection efficiency for each pixel: the probability that we detect the source assuming that the source is in that pixel, as a function of exposure time. For the purpose of implementation of this function in a MILP, we approximate it with a piecewise linear function.

.. plot::
    :include-source: False
    :caption: Piecewise linear approximation of the detection efficiency for a given pixel

    from matplotlib import pyplot as plt
    import numpy as np
    from scipy import stats

    approx_color, exact_color, *_ = (props['color'] for props in iter(plt.rcParams['axes.prop_cycle']))

    q = np.pad(np.linspace(0.05, 0.95, 5), (1, 0))
    log_flux = np.linspace(-3, 3)
    p = stats.norm.cdf(log_flux)
    t = np.exp(0.5 * log_flux)
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(t, p, color=exact_color)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)

    tq = np.exp(0.5 * stats.norm.ppf(q))
    ax.plot(tq, q, ":", marker="o", color=approx_color, clip_on=False)
    for n, (x, y) in enumerate(zip(tq, q)):
        kwargs = {}
        if n == 0:
            kwargs["ha"] = 'left'
            kwargs['va'] = 'top'
            kwargs['xytext'] = (0, -4)
        elif n == 1:
            kwargs["ha"] = 'left'
            kwargs['va'] = 'center'
            kwargs['xytext'] = (2, 0)
        else:
            kwargs['ha'] = 'right'
            kwargs['va'] = 'bottom'
            kwargs['xytext'] = (-2, 2)
        ax.annotate(
            rf" $(\epsilon_{{i{n}}}, \xi_{n})$", (x, y),
            textcoords='offset points', color=approx_color,
            **kwargs)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.plot(3, 0, ">k", clip_on=False)
    ax.set_xlabel("Exposure time")
    ax.set_ylabel("Detection efficiency")
    ax.set_xticks([])

Additional data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Use the function :obj:`ligo.skymap.distance.parameters_to_moments` and Equations :eq:`log-distance-parameters` and :eq:`appmag-parameters` to calculate the mean and standard deviation of the apparent magnitude in each pixel.

2. Select desired quantiles for the approximation of the detection efficiency curve: for example, :math:`(0, 0.05 , 0.275, 0.5  , 0.725, 0.95)`. For each pixel, calculate the exposure time required to achieve the specified detection efficiencies.

MILP problem formulation
^^^^^^^^^^^^^^^^^^^^^^^^

Additional index sets
"""""""""""""""""""""

- :math:`N = \{0, 1, \dots, n_N\}`: indices of quantiles of detection efficiency function approximation

Additional parameters
"""""""""""""""""""""

- :math:`\left(\xi_n\right)_{n \in N}`: quantiles for piecewise linear approximation of detection efficiency curve
- :math:`\left(\epsilon_{in}\right)_{i \in I, n \in N}`: exposure time required to achieve a detection efficiency of :math:`\xi_n` in pixel :math:`i`
- :math:`\left(f_i\right)_{i \in I}`: piecewise linear function interpolating between the points :math:`(\xi_n, \epsilon_{in})_{i \in I, n \in N}`

Additional decision variables
"""""""""""""""""""""""""""""

- :math:`\left(p_i\right)_{i \in I}`: change from a binary variable to a continuous variable

Additional constraints
""""""""""""""""""""""

**Depth.** Replace Equation :eq:`variable-exptime-constraint-depth` with:

.. math::
    \forall i \in I :\quad \max_{j \in J_i} e_{j} \geq f_i(p_\mathrm{i})

Objective
"""""""""

Same as above.
