CC = hipcc
FLAGS = -ffast-math -O3 --offload-arch=gfx942

SRCS = hartree-fock.cc
TARGET = hartree-fock-hip

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

