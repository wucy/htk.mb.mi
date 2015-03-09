#include <stdio.h>

void main(int argc, char *argv[]) {
    FILE *fp;
    FILE *fpsv;
    int ch;

    ch = ' ';
    fp = fopen(argv[1], "r");
    fpsv = fopen(argv[1], "r");
    
    ch = fgetc(fpsv);
    ch = fgetc(fp);
    printf(" 1st char %c\n", ch);
    ch = fgetc(fpsv);
    ch = fgetc(fp);
    printf(" 2nd char %c\n", ch);
    ch = fgetc(fpsv);
    ch = fgetc(fp);
    rewind(fp);

    ch = fgetc(fp);
    printf(" 1st char %c\n", ch);
    ch = fgetc(fpsv);
    printf(" 1st char of fpsv %c\n", ch);
    ch = fgetc(fp);
    printf(" 1st char %c\n", ch);

    fclose(fp);
    fclose(fpsv);
}

