CC = nvcc
FLAGS=-O3 -std=c++17

SRCS = main.cpp CUDAStream.cu
HEADERS = Stream.h CUDAStream.h
TARGET = babelstream_cuda

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)