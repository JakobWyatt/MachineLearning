CXX=g++
CPPFLAGS=-g -std=c++17 -c $(shell root-config --cflags)

SRCDIR=./src/
OBJDIR=./bin/linux/

SRCS=/math.cpp /nn.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: clean libml.a

libml.a: $(subst /,,$(OBJS))
	ar rcs $(OBJDIR)libdnn.a $(subst /,$(OBJDIR),$(OBJS))

nn.o:
	$(CXX) $(CPPFLAGS) $(SRCDIR)nn.cpp -o $(OBJDIR)nn.o

math.o:
	$(CXX) $(CPPFLAGS) $(SRCDIR)math.cpp -o $(OBJDIR)math.o

clean:
	rm -rf $(OBJDIR)*