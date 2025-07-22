CC = nvcc
FLAGS=-forward-unknown-to-host-compiler -arch=sm_90  -DNDEBUG -O3 -march=native -std=c++11

SRCS = main.cpp CUDAStream.cu
HEADERS = Stream.h CUDAStream.h
TARGET = babelstream_cuda

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)