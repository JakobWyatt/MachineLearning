all: libml.a

libml.a: math.o nn.o
	ar rcs bin/linux/libml.a bin/linux/math.o bin/linux/nn.o

nn.o:
	g++ -g -std=c++17 -c src/nn.cpp -o bin/linux/nn.o

math.o:
	g++ -g -std=c++17 -c src/math.cpp -o bin/linux/math.o

clean:
	rm -rf ./bin/linux/*