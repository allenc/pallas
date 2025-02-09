## core-cpp

:: Code Organization

pallas/ 
    core-cpp/                 [common lib]
	microservice.h
	shm_spsc.h
	timer.h
    model-cpp/                [model interface lib]
    
    starburstd/               [camera interface daemon]
    starforged/               [camera visualizer developer tools: camera viewer, camera calibrator]
    psystreamd/               [ml vision pipeline executor daemon]
    psyforged/                [ml vision pipeline developer tools: data viewer, model evals]

    joyd/                     [health monitoring daemon]

implementation of a single-producer, single-consumer, lock-free queue