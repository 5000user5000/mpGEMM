CXX        := g++
CXXFLAGS   := -std=c++17 -O2 -Wall -I./src -march=native -fopenmp

PYBIND11_INC := $(shell python3 -m pybind11 --includes)
PYEXT := $(shell python3-config --extension-suffix)


# ---- MKL toggle ----
ifeq ($(USE_MKL),1)
	MKL_INC := /usr/include/mkl
	MKL_LIBDIR := /usr/lib/x86_64-linux-gnu
	CXXFLAGS += -DUSE_MKL -I$(MKL_INC)
	LDLIBS   += -Wl,--no-as-needed -L$(MKL_LIBDIR) -lmkl_rt -lpthread -lm -ldl
endif
# --------------------

# ---- OpenMP toggle ----
LDLIBS   += -lgomp
# --------------------


SRC_DIR    := src
TEST_DIR   := tests
BUILD_DIR  := build

# main test
TEST_SRC       := $(TEST_DIR)/run_benchmark.cpp
TARGET_MAIN    := $(BUILD_DIR)/run_benchmark

# correctness suite
CORR_SRC       := $(TEST_DIR)/test_correctness.cpp
TARGET_CORR    := $(BUILD_DIR)/test_correctness

# matrix ops test
MATRIX_OPS_SRC := $(TEST_DIR)/test_matrix_ops.cpp
TARGET_MATRIX_OPS := $(BUILD_DIR)/test_matrix_ops

HEADERS    := \
    $(SRC_DIR)/layout_policies.hpp \
    $(SRC_DIR)/storage_policies.hpp \
    $(SRC_DIR)/matrix.hpp \
    $(SRC_DIR)/matrix_ops.hpp \
    $(SRC_DIR)/lut_utils.hpp \
	$(SRC_DIR)/post_processing.hpp

.PHONY: all run test clean pytest matrix_ops matrix_ops_float matrix_ops_lut

all: $(BUILD_DIR) $(TARGET_MAIN) $(TARGET_CORR) $(TARGET_MATRIX_OPS) mpgemm$(PYEXT)

# ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# build main
$(TARGET_MAIN): $(TEST_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TARGET_MAIN) $(LDFLAGS) $(LDLIBS)

# build correctness suite
$(TARGET_CORR):  $(CORR_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CORR_SRC) -o $(TARGET_CORR) $(LDFLAGS) $(LDLIBS)

# build matrix ops test
$(TARGET_MATRIX_OPS): $(MATRIX_OPS_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -pthread $(MATRIX_OPS_SRC) -o $(TARGET_MATRIX_OPS) $(LDFLAGS) $(LDLIBS)

# build pybind11 module
mpgemm$(PYEXT):  src/bindings.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INC) -fPIC -shared src/bindings.cpp -o $@ $(LDFLAGS) $(LDLIBS)

# run pytest
pytest: all
	PYTHONPATH=. python3 -m pytest -q tests/test_api.py

run: all
	./$(TARGET_MAIN)

test: $(TARGET_CORR)
	./$(TARGET_CORR)

matrix_ops: $(TARGET_MATRIX_OPS)
	@echo "Running float version..."
	@./$(TARGET_MATRIX_OPS) float
	@echo "\nRunning LUT version..."
	@./$(TARGET_MATRIX_OPS) lut

# 方便的命令
matrix_ops_float: $(TARGET_MATRIX_OPS)
	./$(TARGET_MATRIX_OPS) float

matrix_ops_lut: $(TARGET_MATRIX_OPS)
	./$(TARGET_MATRIX_OPS) lut

clean:
	rm -rf $(BUILD_DIR)
	rm -f mpgemm$(PYEXT)