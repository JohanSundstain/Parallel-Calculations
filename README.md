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
