'''
Disclaimer I have only done ZTF ToO scheduling with so if these other
schedules/constraints don't make much sense for the observers my apologies

This is more of a proof of concept for different code design types so if it's a
bit weird, it's not that big a deal
'''

observer = Observer.fromFile('../telescopes/ZTF.dat')
observer.setScheduleType('too', skymap = 'exampleSkymap.fits')
observer.addConstaint(CadenceConstraint(
    'allSame', time = 30 * u.min, visits = 2))
observer.addConstaint(FilterConstraint(('g', 'r')))
observer.addConstaint(TimeDeadlineConstraint(
    Time(59573.8, format = 'mjd'), Time(59574.2, format = 'mjd')))
schedule = observer.solve('gurobi')


observer = Observer.fromFile('../telescopes/SEDM.dat')
observer.setScheduleType('observerList',
                            targets = (
                                (SkyCoord(30, 30), 420 * u.s),
                                (SkyCoord(50, -10), 1800 * u.s),
                                (SkyCoord(-15, 45), 600 * u.s)
                            )
                         )
observer.addConstaint(TimeDeadlineConstraint(
    Time(59573.8, format = 'mjd'), Time(59574.2, format = 'mjd')))
schedule = observe.solve('cplex')


observer = Observer.fromFile('../telescopes/TESS.dat')
observer.setScheduleType('observerList',
                         targets = ascii.read('exampleTargetList.csv'))
observer.addConstaint(TimeDeadlineConstraint(
    Time(59573.8, format = 'mjd'), Time(59574.2, format = 'mjd')))
observer.addConstaint(eclipsingConstraint('relevant parameters'))
observer.addConstaint(cadenceConstraint('perObserver',
    cadenceList = ascii.read('exampleTargetList.csv')['cadence']))