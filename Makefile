CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build

TARGET := $(BUILD_DIR)/main
SRC := $(SRC_DIR)/matrix.cpp $(TEST_DIR)/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
