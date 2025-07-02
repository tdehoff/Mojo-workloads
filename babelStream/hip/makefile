CC = hipcc
FLAGS=-O3 -std=c++17

SRCS = main.cpp HIPStream.cpp
HEADERS = Stream.h HIPStream.h
TARGET = babelstream_hip

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(FLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)