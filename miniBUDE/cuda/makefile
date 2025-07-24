CC = nvcc -x cu

# Include fast-math
FLAGS=-std=c++17 -forward-unknown-to-host-compiler -DMODEL=cuda -extended-lambda -use_fast_math -restrict -arch=sm_90
# Without fast-math
# FLAGS=-std=c++17 -forward-unknown-to-host-compiler -DMODEL=cuda -extended-lambda -restrict -arch=sm_90

TARGET = bude_cuda

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)