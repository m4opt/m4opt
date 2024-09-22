Multimessenger follow-up as an MILP problem
===========================================

Here we document our representation of the multimessenger follow-up problem as a mixed integer linear program.

Problem 1: Fixed exposure time
------------------------------

Problem statement
^^^^^^^^^^^^^^^^^

We receive a :doc:`HEALPix probability sky map <userguide:tutorial/skymaps>` that describes the probability distribution of the true but unknown position of a target of interest as a function of position on the sky. There is a delay between the time that the event occurred and when we can start observations due to the time it takes to uplink commands to the spacecraft, and there is a deadline by which we must complete our observations.

Our telescope can observe any of a set of :math:`n_r` reference fields at predetermined sky locations in order to tile the sky map. For each reference field that we select, our telescope must visit the reference field at least :math:`n_v` times. We have a cadence requirement: each visit of a given reference field must occur at least a time :math:`\gamma` after the previous visit.

Every visit takes a certain amount of exposure time, and it takes a known amount of time to slew between different reference fields. We may only a visit a reference field when it is within the field of regard, the region that constrains where the telescope may point at any given instant of time.

Data preparation
^^^^^^^^^^^^^^^^^

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

- :math:`I = \{0, 1, \dots, n_p - 1\}`: pixels
- :math:`J = \{0, 1, \dots, n_r - 1\}`: reference fields
- :math:`K = \{0, 1, \dots, n_v - 1\}`: visits
- :math:`\left(M_j\right)_{j \in J}`: observable segments for reference field :math:`j`
- :math:`\left(J_i\right)_{i \in I}`: set of indices of fields that contain pixel :math:`i`

Parameters
""""""""""

- :math:`\left(\rho_i\right)_{i \in I}`: probability that source is inside pixel :math:`i`
- :math:`\left(\sigma_{jj^\prime}\right)_{j \in J, j^\prime \in J}`: slew time from reference field :math:`j` to reference field :math:`j^\prime`
- :math:`\left(\alpha_{jm}\right)_{j \in J, m \in M}`: start time of observable segment :math:`m` of reference field :math:`j`
- :math:`\left(\omega_{jm}\right)_{j \in J, m \in M}`: end time of observable segment :math:`m` of reference field :math:`j`
- :math:`\epsilon`: exposure time
- :math:`\gamma`: cadence, time between visits

Decision variables
""""""""""""""""""

Binary decision variables:

- :math:`\left(p_i\right)_{i \in I}`: pixel :math:`i` is inside the footprint of one or more selected reference fields
- :math:`\left(r_j\right)_{j \in J}`: reference field :math:`j` is selected for observation
- :math:`\left(s_{jkm}\right)_{j \in J, k \in K, m \in M}`: whether reference field :math:`j` visit :math:`k` occurs in segment :math:`m`

Continuous decision variables:

- :math:`\left(t_{jk}\right)_{j \in J, k \in K}`: midpoint time of observation :math:`j` visit :math:`k`

Constraints
"""""""""""

**Containment.** Only count pixels that are inside one or more reference fields.

.. math::

    \forall i :\quad p_i \leq \sum_{j \in J_i} r_j

**Cadence.** If a reference field is selected for observation, then enforce a minimum time between visits.

.. math::

    \forall k > 1 ,\; j :\quad t_{jk} - t_{j,k-1} \geq (\epsilon + \gamma) r_j

**No overlap.** Observations cannot overlap in time; they must be separated by at least the exposure time plus the slew time.

.. math::
    \forall j \ne j^\prime ,\; k ,\; k^\prime :\quad \left|t_{jk} - t_{j^\prime k^\prime}\right|  \geq \left(\sigma_{jj^\prime} + \epsilon\right) \left( r_j + r_{j^\prime} - 1\right)

**Field of regard.** An observation of a reference field can only occur while the coordinates of the reference field are within the field of regard.
Note: for each reference field :math:`j` that has exactly one observable interval, instead of an indicator constraint, simply place lower and upper bounds on :math:`t_{jk}`.

.. math::

    \forall j ,\; k \;, m :\quad s_{jkm} = 1 \;\Rightarrow\; \alpha_{jm} + \epsilon / 2 \leq t_{jk} \leq \omega_{jm} - \epsilon / 2

Objective
"""""""""

Maximize the sum of the probability of all of the pixels that are contained within selected fields:

.. math::

    \sum_{i \in I} \rho_i p_i

Problem 2: Variable exposure time
---------------------------------

Problem statement
^^^^^^^^^^^^^^^^^

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

    \forall i \in I :\quad p_\mathrm{i} = 1 \Rightarrow \max_{j \in J_i} e_{j} \geq \epsilon_i

**Exposure time.** If a field's exposure time is nonzero, then it is selected for observation.

.. math::

    \forall j \in J :\quad \epsilon_\mathrm{max} r_j \geq e_\mathrm{j}

**Cadence.** If a reference field is selected for observation, then enforce a minimum time between visits.

.. math::

    \forall k > 1 ,\; j :\quad t_{jk} - t_{j,k-1} \geq \gamma r_j + e_j

**No overlap.** Observations cannot overlap in time; they must be separated by at least the exposure time plus the slew time.

.. math::
    \forall j \ne j^\prime ,\; k ,\; k^\prime :\quad \left|t_{jk} - t_{j^\prime k^\prime}\right|  \geq \sigma_{jj^\prime} \left( r_j + r_{j^\prime} - 1\right) + (e_j + e_\mathrm{j^\prime}) / 2

**Field of regard.** An observation of a reference field can only occur while the coordinates of the reference field are within the field of regard.
Note: for each reference field :math:`j` that has exactly one observable interval, instead of an indicator constraint, simply place lower and upper bounds on :math:`t_{jk}`.

.. math::

    \forall j ,\; k \;, m :\quad s_{jkm} = 1 \;\Rightarrow\; \alpha_{jm} + e_\mathrm{j}/2 \leq t_{jk} \leq \omega_{jm} - e_\mathrm{j}/2

Objective
"""""""""

Same as above.
