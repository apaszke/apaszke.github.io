#include <stdio.h>
#include <unistd.h>
#include <signal.h>

extern char *data;
extern void die(int code, char *msg);

void do_job()
{
    usleep(100000000);
}

static void print_data()
{
    printf("%s\n", data);
}

void setup(int argc, char *argv[])
{
    printf("My PID: %d\n", getpid());
    
    if (signal(SIGUSR1, print_data) == SIG_ERR)
        die(10, "Failed to register signal handler\n");
}
