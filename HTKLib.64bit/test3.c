#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void test(float val) {
    if (val)
        printf("orz + %e\n", val);
}

int main() {
    float a = 1.0;
    float b = 0.0;
    float c = -1.0;

    test(a);
    test(b);
    test(c);

    return 0;
}

