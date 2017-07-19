# Validation

## Validate on same months previous year
#### Median
Apply imputation after median is computed.

```
days_to_consider: 14
de SMAPE: 46.81
en SMAPE: 46.61
es SMAPE: 68.13
fr SMAPE: 46.83
ja SMAPE: 45.79
na SMAPE: 53.39
ru SMAPE: 39.27
zh SMAPE: 44.26
SMAPE: 48.89

days_to_consider: 21
de SMAPE: 45.12
en SMAPE: 45.40
es SMAPE: 64.69
fr SMAPE: 45.19
ja SMAPE: 44.88
na SMAPE: 52.83
ru SMAPE: 38.41
zh SMAPE: 44.08
SMAPE: 47.57

days_to_consider: 28
de SMAPE: 44.31
en SMAPE: 44.83
es SMAPE: 58.21
fr SMAPE: 44.57
ja SMAPE: 44.73
na SMAPE: 52.52
ru SMAPE: 38.24
zh SMAPE: 44.13
SMAPE: 46.44

days_to_consider: 35
de SMAPE: 43.80
en SMAPE: 44.84
es SMAPE: 53.37
fr SMAPE: 44.11
ja SMAPE: 44.67
na SMAPE: 52.03
ru SMAPE: 38.46
zh SMAPE: 44.41
SMAPE: 45.71

days_to_consider: 42
de SMAPE: 43.46
en SMAPE: 44.98
es SMAPE: 50.84
fr SMAPE: 44.00
ja SMAPE: 44.44
na SMAPE: 51.90
ru SMAPE: 38.67
zh SMAPE: 44.67
SMAPE: 45.37

days_to_consider: 49
de SMAPE: 43.44
en SMAPE: 45.14
es SMAPE: 49.87
fr SMAPE: 44.36
ja SMAPE: 44.42
na SMAPE: 51.91
ru SMAPE: 38.89
zh SMAPE: 45.15
SMAPE: 45.40

days_to_consider: 56
de SMAPE: 43.42
en SMAPE: 45.19
es SMAPE: 49.24
fr SMAPE: 44.15
ja SMAPE: 44.38
na SMAPE: 51.91
ru SMAPE: 38.93
zh SMAPE: 45.46
SMAPE: 45.33

days_to_consider: 63
de SMAPE: 43.44
en SMAPE: 45.33
es SMAPE: 49.04
fr SMAPE: 44.11
ja SMAPE: 44.54
na SMAPE: 51.98
ru SMAPE: 39.02
zh SMAPE: 45.81
SMAPE: 45.41

days_to_consider: 70
de SMAPE: 43.43
en SMAPE: 45.55
es SMAPE: 48.71
fr SMAPE: 44.07
ja SMAPE: 44.73
na SMAPE: 52.12
ru SMAPE: 39.23
zh SMAPE: 46.16
SMAPE: 45.50

days_to_consider: 77
de SMAPE: 43.48
en SMAPE: 45.69
es SMAPE: 48.57
fr SMAPE: 44.15
ja SMAPE: 44.82
na SMAPE: 52.18
ru SMAPE: 39.53
zh SMAPE: 46.40
SMAPE: 45.60

days_to_consider: 84
de SMAPE: 43.54
en SMAPE: 45.86
es SMAPE: 48.47
fr SMAPE: 44.22
ja SMAPE: 44.88
na SMAPE: 52.36
ru SMAPE: 39.80
zh SMAPE: 46.51
SMAPE: 45.70

days_to_consider: 91
de SMAPE: 43.58
en SMAPE: 46.05
es SMAPE: 48.48
fr SMAPE: 44.31
ja SMAPE: 45.00
na SMAPE: 52.52
ru SMAPE: 40.12
zh SMAPE: 46.66
SMAPE: 45.84

days_to_consider: 98
de SMAPE: 43.71
en SMAPE: 46.29
es SMAPE: 48.55
fr SMAPE: 44.44
ja SMAPE: 44.99
na SMAPE: 52.69
ru SMAPE: 40.41
zh SMAPE: 46.68
SMAPE: 45.97
```

#### Median with weekends separate
Apply separate medians to Saturdays and Sundays
Apply imputation after median is computed.
```
days_to_consider: 14
de SMAPE: 46.93
en SMAPE: 46.93
es SMAPE: 68.15
fr SMAPE: 46.96
ja SMAPE: 46.04
na SMAPE: 53.53
ru SMAPE: 39.25
zh SMAPE: 44.53
SMAPE: 49.04

days_to_consider: 21
de SMAPE: 44.64
en SMAPE: 45.48
es SMAPE: 64.71
fr SMAPE: 45.04
ja SMAPE: 44.67
na SMAPE: 52.66
ru SMAPE: 38.30
zh SMAPE: 44.20
SMAPE: 47.46

days_to_consider: 28
de SMAPE: 43.44
en SMAPE: 44.45
es SMAPE: 57.05
fr SMAPE: 44.15
ja SMAPE: 44.42
na SMAPE: 52.26
ru SMAPE: 37.94
zh SMAPE: 44.15
SMAPE: 45.98

days_to_consider: 35
de SMAPE: 42.87
en SMAPE: 44.29
es SMAPE: 51.36
fr SMAPE: 43.48
ja SMAPE: 44.25
na SMAPE: 51.71
ru SMAPE: 37.98
zh SMAPE: 44.37
SMAPE: 45.04

days_to_consider: 42
de SMAPE: 42.46
en SMAPE: 44.44
es SMAPE: 49.64
fr SMAPE: 43.37
ja SMAPE: 44.01
na SMAPE: 51.61
ru SMAPE: 38.13
zh SMAPE: 44.58
SMAPE: 44.78

days_to_consider: 49
de SMAPE: 42.43
en SMAPE: 44.61
es SMAPE: 48.69
fr SMAPE: 43.70
ja SMAPE: 43.92
na SMAPE: 51.56
ru SMAPE: 38.32
zh SMAPE: 45.06
SMAPE: 44.79

days_to_consider: 56
de SMAPE: 42.40
en SMAPE: 44.72
es SMAPE: 48.07
fr SMAPE: 43.48
ja SMAPE: 43.84
na SMAPE: 51.50
ru SMAPE: 38.39
zh SMAPE: 45.37
SMAPE: 44.72    -> 45.1

days_to_consider: 63
de SMAPE: 42.45
en SMAPE: 44.85
es SMAPE: 47.81
fr SMAPE: 43.38
ja SMAPE: 43.97
na SMAPE: 51.52
ru SMAPE: 38.48
zh SMAPE: 45.67
SMAPE: 44.77

days_to_consider: 70
de SMAPE: 42.46
en SMAPE: 45.06
es SMAPE: 47.57
fr SMAPE: 43.37
ja SMAPE: 44.17
na SMAPE: 51.66
ru SMAPE: 38.69
zh SMAPE: 46.07
SMAPE: 44.88

days_to_consider: 77
de SMAPE: 42.48
en SMAPE: 45.22
es SMAPE: 47.47
fr SMAPE: 43.42
ja SMAPE: 44.26
na SMAPE: 51.79
ru SMAPE: 38.96
zh SMAPE: 46.30
SMAPE: 44.99

days_to_consider: 84
de SMAPE: 42.56
en SMAPE: 45.38
es SMAPE: 47.39
fr SMAPE: 43.49
ja SMAPE: 44.30
na SMAPE: 51.90
ru SMAPE: 39.25
zh SMAPE: 46.38
SMAPE: 45.08

days_to_consider: 91
de SMAPE: 42.61
en SMAPE: 45.56
es SMAPE: 47.38
fr SMAPE: 43.59
ja SMAPE: 44.40
na SMAPE: 52.07
ru SMAPE: 39.55
zh SMAPE: 46.50
SMAPE: 45.21

days_to_consider: 98
de SMAPE: 42.74
en SMAPE: 45.78
es SMAPE: 47.45
fr SMAPE: 43.70
ja SMAPE: 44.38
na SMAPE: 52.20
ru SMAPE: 39.85
zh SMAPE: 46.51
SMAPE: 45.33
```

#### Median with extended weekends separate
Apply separate medians to Fridays, Saturdays and Sundays
Apply imputation after median is computed.
```
days_to_consider: 14
de SMAPE: 46.86
en SMAPE: 46.79
es SMAPE: 68.08
fr SMAPE: 46.86
ja SMAPE: 46.34
na SMAPE: 53.37
ru SMAPE: 39.59
zh SMAPE: 44.47
SMAPE: 49.05

days_to_consider: 21
de SMAPE: 44.67
en SMAPE: 45.49
es SMAPE: 64.77
fr SMAPE: 45.02
ja SMAPE: 44.80
na SMAPE: 52.47
ru SMAPE: 38.74
zh SMAPE: 44.24
SMAPE: 47.53

days_to_consider: 28
de SMAPE: 43.21
en SMAPE: 44.38
es SMAPE: 57.01
fr SMAPE: 44.01
ja SMAPE: 44.48
na SMAPE: 52.00
ru SMAPE: 38.38
zh SMAPE: 44.24
SMAPE: 45.96

days_to_consider: 35
de SMAPE: 42.64
en SMAPE: 44.14
es SMAPE: 50.96
fr SMAPE: 43.40
ja SMAPE: 44.34
na SMAPE: 51.47
ru SMAPE: 38.44
zh SMAPE: 44.52
SMAPE: 44.99

days_to_consider: 42
de SMAPE: 42.44
en SMAPE: 44.34
es SMAPE: 49.23
fr SMAPE: 43.44
ja SMAPE: 44.08
na SMAPE: 51.37
ru SMAPE: 38.58
zh SMAPE: 44.68
SMAPE: 44.77

days_to_consider: 49
de SMAPE: 42.50
en SMAPE: 44.53
es SMAPE: 48.34
fr SMAPE: 43.84
ja SMAPE: 43.98
na SMAPE: 51.30
ru SMAPE: 38.75
zh SMAPE: 45.14
SMAPE: 44.80

days_to_consider: 56
de SMAPE: 42.50
en SMAPE: 44.63
es SMAPE: 47.94
fr SMAPE: 43.63
ja SMAPE: 43.95
na SMAPE: 51.26
ru SMAPE: 38.79
zh SMAPE: 45.47
SMAPE: 44.77

days_to_consider: 63
de SMAPE: 42.51
en SMAPE: 44.81
es SMAPE: 47.84
fr SMAPE: 43.57
ja SMAPE: 44.09
na SMAPE: 51.31
ru SMAPE: 38.87
zh SMAPE: 45.77
SMAPE: 44.85

days_to_consider: 70
de SMAPE: 42.58
en SMAPE: 44.97
es SMAPE: 47.72
fr SMAPE: 43.53
ja SMAPE: 44.29
na SMAPE: 51.41
ru SMAPE: 39.07
zh SMAPE: 46.10
SMAPE: 44.96

days_to_consider: 77
de SMAPE: 42.59
en SMAPE: 45.17
es SMAPE: 47.84
fr SMAPE: 43.56
ja SMAPE: 44.40
na SMAPE: 51.54
ru SMAPE: 39.35
zh SMAPE: 46.41
SMAPE: 45.11

days_to_consider: 84
de SMAPE: 42.64
en SMAPE: 45.32
es SMAPE: 47.86
fr SMAPE: 43.65
ja SMAPE: 44.45
na SMAPE: 51.65
ru SMAPE: 39.63
zh SMAPE: 46.52
SMAPE: 45.22

days_to_consider: 91
de SMAPE: 42.68
en SMAPE: 45.51
es SMAPE: 47.89
fr SMAPE: 43.76
ja SMAPE: 44.55
na SMAPE: 51.83
ru SMAPE: 39.94
zh SMAPE: 46.66
SMAPE: 45.35

days_to_consider: 98
de SMAPE: 42.82
en SMAPE: 45.73
es SMAPE: 48.02
fr SMAPE: 43.90
ja SMAPE: 44.55
na SMAPE: 52.02
ru SMAPE: 40.27
zh SMAPE: 46.69
SMAPE: 45.50
```