CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
DATA_DIR := ../data

# main test file
TARGET := $(BUILD_DIR)/main
SRC := $(SRC_DIR)/matrix.cpp  $(SRC_DIR)/matrix_packed.cpp  $(TEST_DIR)/main.cpp

# correctness test file
TEST_TARGET := $(BUILD_DIR)/test_correctness
TEST_SRC := $(SRC_DIR)/matrix.cpp $(SRC_DIR)/matrix_packed.cpp $(TEST_DIR)/test_correctness.cpp


all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

$(TEST_TARGET): $(TEST_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) -lpthread

# 建立數據檔資料夾（../data），如果尚未存在的話
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

run: $(TARGET)
	./$(TARGET)

# 執行正確性測試，並確保 ../data 資料夾存在
test: $(TEST_TARGET) $(DATA_DIR)
	./$(TEST_TARGET)

clean:
	rm -rf $(BUILD_DIR)
