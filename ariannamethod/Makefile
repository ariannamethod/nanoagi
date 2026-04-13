# ariannamethod/Makefile — build libnotorch for nanoagi
# "fuck torch"

CC = cc
CFLAGS = -O2 -Wall -std=c11 -fPIC

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  LIB = libnotorch.dylib
  SHARED = -shared -dynamiclib
endif
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
  LIB = libnotorch.so
  SHARED = -shared
endif

.PHONY: all clean

all: $(LIB)

$(LIB): notorch.c notorch.h
	$(CC) $(CFLAGS) $(SHARED) $(BLAS_FLAGS) -o $(LIB) notorch.c -lm
	@echo "Built $(LIB)"

clean:
	rm -f $(LIB) *.o
