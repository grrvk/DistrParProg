CXX = clang++
CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

TARGET = sharpen
OBJ = sharpen.o

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
	./$(TARGET) "$(INPUT_DIR)" "$(OUTPUT_DIR)" $(MAX_IMAGES)
