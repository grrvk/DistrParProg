CXX = clang++
CXXFLAGS = -std=c++17 -I/opt/homebrew/Cellar/open-mpi/5.0.7/include `pkg-config --cflags opencv4`
LDFLAGS = -L/opt/homebrew/Cellar/open-mpi/5.0.7/lib -lmpi `pkg-config --libs opencv4`

TARGET = sharpen
OBJ = sharpen.o

NUMBER_OF_PROCESSES = 2
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
	mpirun -np $(NUMBER_OF_PROCESSES) ./$(TARGET) "$(INPUT_DIR)" "$(OUTPUT_DIR)" $(MAX_IMAGES)
