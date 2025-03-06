# Lab 1
## On branch master with CMakeLists.txt
### To compile executable file with array of double in build directory
1. `cmake .. -D double:BOOL=ON # compile file with array of double`
### OR
2. `cmake .. # compile file with array of float`
3. `make`

### Result with double 
### 6.28319e-07 # <- step 
### 3.68912e-10 # <- sum

### Result with float
### 6.28319e-07 # <- step 
### -0.213894   # <- sum

# Lab 2
## SYS INFO
### PROC
`Architecture:           x86_64`<br>
`CPU op-mode(s):         32-bit, 64-bit`<br>
`Address sizes:          46 bits physical, 48 bits virtual`<br>
`Byte Order:             Little Endian`<br>
`CPU(s):                 80`<br>
`On-line CPU(s) list:    0-79`<br>
`Vendor ID:              GenuineIntel`<br>
`Model name:             Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz`<br>
`CPU family:             6`<br>
`Model:                  85`<br>
`Thread(s) per core:     2`<br>
`Core(s) per socket:     20`<br>
`Socket(s):              2`<br>
`Stepping:               7`<br>
`CPU max MHz:            3900.0000`<br>
`CPU min MHz:            1000.0000`<br>
`BogoMIPS:               5000.00`
### SERVER
`ProLiant XL270d Gen10`
### NUMA
`available: 2 nodes (0-1)`<br>
`node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59`<br>
`node 0 size: 385636 MB`<br>
`node 0 free: 242756 MB`<br>
`node 1 cpus: 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79`<br>
`node 1 size: 387008 MB`<br>
`node 1 free: 260900 MB`<br>
`node distances:`<br>
`node   0   1 `<br>
`  0:  10  21 `<br>
`  1:  21  10 `
## RESULTS
| M=N  | T1         | S2        | S4     |S7     |S8     |S16    |S20    |S40    |
|------|------------|-----------|--------|-------|-------|-------|-------|-------|
| 20000| 0.624741   | 1.87079   | 3.64727|6.24131|7.06638|15.5192|19.5824|21.9282|
| 40000| 2.56161    | 1.94669   |3.88276 |6.6319 |7.17922|12.149 |12.1201|21.1387|

