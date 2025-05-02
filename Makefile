CXX        := g++
CXXFLAGS   := -std=c++17 -O2 -Wall -I./src -march=native

# ---- MKL toggle ----
ifeq ($(USE_MKL),1)
    CXXFLAGS += -DUSE_MKL -I$(MKLROOT)/include
    LDFLAGS  += -L$(MKLROOT)/lib -Wl,--no-as-needed
    LDLIBS   += -lmkl_rt
endif
# --------------------


SRC_DIR    := src
TEST_DIR   := tests
BUILD_DIR  := build

# main test
TEST_SRC       := $(TEST_DIR)/main.cpp
TARGET_MAIN    := $(BUILD_DIR)/main

# correctness suite
CORR_SRC       := $(TEST_DIR)/test_correctness.cpp   # fix filename
TARGET_CORR    := $(BUILD_DIR)/test_correctness       # fix binary name

HEADERS    := \
    $(SRC_DIR)/layout_policies.hpp \
    $(SRC_DIR)/storage_policies.hpp \
    $(SRC_DIR)/matrix.hpp \
    $(SRC_DIR)/matrix_ops.hpp \
    $(SRC_DIR)/lut_utils.hpp

.PHONY: all run test clean

all: $(BUILD_DIR) $(TARGET_MAIN) $(TARGET_CORR)

# ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# build main
$(TARGET_MAIN): $(TEST_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TARGET_MAIN) $(LDFLAGS) $(LDLIBS)

# build correctness suite
$(TARGET_CORR): $(CORR_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CORR_SRC) -o $(TARGET_CORR) $(LDFLAGS) $(LDLIBS)

run: all
	./$(TARGET_MAIN)

test: $(TARGET_CORR)
	./$(TARGET_CORR)

clean:
	rm -rf $(BUILD_DIR)