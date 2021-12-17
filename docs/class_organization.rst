Class Organization Rubric
=================================

This is an initial organization of the various classes that will be present in |M4OPT|. 
Over time, this document will be subsumed by documentation for each submodule within the overall module.
For now, this is a place for standardizing and organizing |M4OPT| classes using pseudocode. Please
feel free to modify as needed.

Scheduler
---------

The Scheduler class is the overarching class that will do the optimization, given 
Targets, Observatories, and Constraints.

.. code-block:: python
    class Scheduler
        self.observing_plan = []
        self.observatories = []
        self.targets = []
        self.optimizer = ""

        def optimize(self, **args, **kwargs):
            # do optimization ...
            try
                constraints = [c for c in self.observatories.constraints]
                self.steps_in_optimization(constraints, **args, **kwargs)
                self.observing_plan = optimal_plan
                OUTCOME_CODE = m4opt.SUCCESS
            except
                OUTCOME_CODE = m4opt.FAILURE
                 
            return OUTCOME_CODE
    #end class

    # functional API for setting scheduler data
    # can be moved inside Scheduler class if desired
    def setObservers(scheduler::Scheduler, obser::List(Observer))
    def setTargets(scheduler::Scheduler, targets::List(Target))
    def setOptimizer(scheduler::Scheduler, opt_name::String="gurobi")

    # etc


Constraint
----------

The Constraint class defines constraints for the MILP problem solved by the Scheduler.

.. code-block:: python
    class AbstractConstraint
    class SunSeparationConstraint
    class MoonSeparationConstraint
    class TimeConstraint
    
    # etc etc

Observer
--------

The Observer is the facility that will observe Targets with their given Instruments.

.. code-block:: python
    class Observer
        self.instruments = []
        self.constraints = []
    #end class

    # functional API
    # can be moved inside Observer class
    def setInstrument(obsr::Observer, instr::Instrument)
    def setConstraint(obsr::Observer, constr::Constraint)

Instrument
----------

The Instrument class holds information about the instrument that will do the observations of the Targets.

.. code-block:: python
    class Instrument
        self.DARK_NOISE = None
        self.READ_NOISE = None
        self.PLATE_SCALE = None
        self.GAIN = None
        self.NPIX = None
        self.NBINS = None
        self.APERATURE_CORRECTION = None
        self.bandpass = None

    # functional API
    # can be moved into Instrument class
    def load_instrument(file_loc::String)
    def set_instrument(instrument_properties::Dict)
    def default_instrument(instrument::String)

    # exposure time calculations
    def exposure_time(target::Target, instrument::Instrument, background::Background)


Target
------

The Target class contains information regarding targets of opportunity within the sky.

.. code-block:: python
    class Target
        self.location = None
        self.flux = None
        self.magnitude = None
        self.spectrum = None
        self.dust_extinction = True
        self.priority = None
    
    # functional API; can be moved into class
    def load_targets(file_loc::String)
    def dust_extinction(dust:Boolean)
    def set_priority(priority::Float)
    

Background
----------

The Background class contains information about any sources that may contaminate 
the observation of the Target by the Instrument.

.. code-block:: python
    class AbstractBackground

    class ZodiacalBackground(AbstractBackground)
    class GalacticBackground(AbstractBackground)
    class AirGlowBackground(AbstractBackground)

