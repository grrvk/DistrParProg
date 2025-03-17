CXX = clang++
CXXFLAGS = -std=c++17 -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp `pkg-config --cflags opencv4`
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp `pkg-config --libs opencv4`

TARGET = sharpen
OBJ = sharpen.o

NUMBER_OF_THREADS = 1
MAX_IMAGES = 100
INPUT_DIR = ../input
OUTPUT_DIR = ../output

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

sharpen.o: sharpen.cpp
	$(CXX) $(CXXFLAGS) -c sharpen.cpp

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET) "$(INPUT_DIR)" "$(OUTPUT_DIR)" $(NUMBER_OF_THREADS) $(MAX_IMAGES)
