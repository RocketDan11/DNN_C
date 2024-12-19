C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c conv/*.c *.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h conv/*.h *.h)
OBJ = ${C_SOURCES:.c=.o}
CFLAGS = -c
LDFLAGS = -lm

MAIN = main
CC = /usr/bin/gcc
LINKER = /usr/bin/ld

run: ${MAIN}
	./${MAIN}

main: ${OBJ}
	${CC} $^ -o $@ ${LDFLAGS}

# Generic rules
%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} $< -o $@

clean:
	rm matrix/*.o neural/*.o util/*.o conv/*.o *.o ${MAIN}
