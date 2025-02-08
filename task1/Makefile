# $@ - target name
# %< - first name

CXX = g++
SRC =  main.cpp
BUILD_DIR = build
SRC_OBJECT = $(BUILD_DIR)/main.o
TARGET = $(BUILD_DIR)/main


ifeq ($(MODE),double)
	MACRO = DOUBLE
else
	MACRO = FLOAT
endif

$(TARGET): $(SRC_OBJECT)
	$(CXX) $< -o $@ 

$(SRC_OBJECT): 
	mkdir -p $(BUILD_DIR)
	$(CXX) -c $(SRC) -o $@ -D $(MACRO)

clean:
	rm -f $(BUILD_DIR)/*.o

delete:
	rm -r $(BUILD_DIR)

run:
	./$(TARGET)
