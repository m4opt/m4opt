'''
Disclaimer I have only done ZTF ToO scheduling with so if these other
schedules/constraints don't make much sense for the observers my apologies

This is more of a proof of concept for different code design types so if it's a
bit weird, it's not that big a deal
'''

schedule = Observer.fromFile(
    '../telescopes/ZTF.dat'
).scheduleType(
    'too', skymap = 'exampleSkymap.fits'
).cadenceConstraint(
    'allSame',
    time = 30 * u.min,
    visits = 2
).filterConstraint(
    ('g', 'r')
).timeDeadlineConstraint(
    Time(59573.8, format = 'mjd'), Time(59574.2, format = 'mjd')
).solve(
    'gurobi'
)

schedule = Observer.fromFile(
    '../telescopes/SEDM.dat'
).scheduleType(
    'observerList',
    targets = (
        (SkyCoord(30, 30), 420 * u.s),
        (SkyCoord(50, -10), 1800 * u.s),
        (SkyCoord(-15, 45), 600 * u.s)
    )
).timeDeadlineConstraint(
    Time(59573.8, format = 'mjd'), Time(59574.2, format = 'mjd')
).solve(
    'cplex'
)

schedule = Observer.fromFile(
    '../telescopes/TESS.dat'
).scheduleType(
    'observerList',
    targets = ascii.read('exampleTargetList.csv')
).timeDeadlineConstraint(
    Time(59573.8, format = 'mjd'), Time(59574.2, format = 'mjd')
).eclipsingConstraint(
    '''
    Relevant parameters
    '''
).cadenceConstraint(
    'perObserver',
    cadenceList = ascii.read('exampleTargetList.csv')['cadence']
)