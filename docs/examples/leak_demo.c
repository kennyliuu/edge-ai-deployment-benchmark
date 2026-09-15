/* Tiny intentional leak for Valgrind practice on Jetson / host. */
#include <stdlib.h>
#include <stdio.h>

int main(void) {
    for (int i = 0; i < 100; i++) {
        char *p = malloc(1024);
        if (!p) return 1;
        p[0] = (char)i;
        /* leak on purpose */
    }
    fprintf(stderr, "leaked 100 KiB intentionally\n");
    return 0;
}
