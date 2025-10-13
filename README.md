# NASA Turbofan Engine Degradation Simulation

### DATASET DESCRIPTION
The engine degradation simulation was carried out with a simulator called C-MAPSS, which is NASA’s aircraft engine simulator. The [dataset](https://github.com/aminhm/NASA-Turbofan-A2/tree/main/data) contains four different scenarios, simulated under different combinations and fault modes. Different sensors acquire data to characterize the fault evolution.<br>
The dataset has 26 columns, separated by spaces. Each column represents a different variable. Each row is a snapshot of data taken during a single operational cycle. The variables (columns) are:<br>

    1. Unit number.
    2. Time, in cycles.
    3. Operational Setting 1.
    4. Operational Setting 2.
    5. Operational Setting 3.
    6. Sensor Measurement 1.
    7. Sensor Measurement 2.
    ...
    32.	Sensor	measurement	26

### MODELLING GOAL
#### Level A2: Optimize the data acquisition
When creating a regression model to predict the RUL, you observe that certain sensors have more importance when it comes to the estimation of the RUL. Your task is to reduce the model to the sensors that are important to predicting the RUL, based on model diagnostics.<br>

Note:	You only need the four operational settings datasets, without the testing partition.


## PROJECT CONFIGURATION AND RUNNING

To run the project, first you need to do
```bash
    pip install -r requirements.txt
```

then, run [main](https://github.com/aminhm/NASA-Turbofan-A2/blob/main/RUL_Estimation_Optimization/main.py) file
