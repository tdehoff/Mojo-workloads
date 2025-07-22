CC = nvcc
FLAGS = -O3 -use_fast_math -arch=sm_90

SRCS = hartree-fock.cu
TARGET = hartree-fock-cuda

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

