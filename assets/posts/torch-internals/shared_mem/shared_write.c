#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

extern char *data;
extern void die(int code, char *msg);

static pid_t reader_pid;

void do_job()
{
    scanf("%s", data);
    kill(reader_pid, SIGUSR1);
}

void setup(int argc, char *argv[])
{
    if (argc > 1) {
        printf("%s\n", argv[1]);
        reader_pid = atoi(argv[1]);
    } else {
        die(11, "No reader PID specified\n");
    }
}
