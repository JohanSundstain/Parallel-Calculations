# Lab 1
## On branch main with Makefile
### To compile executable file with array of double
1. `make MODE=double # compile file with array of double`
2. `make # compile file with array of float`
   
### Other options
1. `make clean # delete object files from build directory`
2. `make delete # delete build directory`
3. `make run # run executable file`

### Result with double 
6.28319e-07 # <- step 
3.68912e-10 # <- sum

### Result with float
6.28319e-07 # <- step 
-0.213894   # <- sum
