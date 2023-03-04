scenario = drivingScenario('SampleTime',0.05);
roadcenters = [5 0; 30 10; 35 25];
lspec = lanespec(2);
road(scenario,roadcenters,'Lanes',lspec);

v = vehicle(scenario,'ClassID',1);
waypoints = [6 2; 18 4; 25 7; 28 10; 31 15; 33 22];
speeds = [30 10 5 5 10 30];
trajectory(v,waypoints,speeds)

plot(scenario,'Waypoints','on','RoadCenters','on')
while advance(scenario)
    pause(0.1)
end