#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>

#define SHARED_OBJ_NAME "/shared_mem_test"
#define SHARED_OBJ_SIZE 255


char *data;
int reader_pid;

extern void do_job();
extern void setup();

void die(int code, char* msg)
{
    printf("ERROR: %s\n", msg);
    exit(code);
}

int main(int argc, char *argv[])
{
    int fd;

    setup(argc, argv);

    if ((fd = shm_open(SHARED_OBJ_NAME, O_RDWR|O_CREAT, 0777)) == -1)
        die(1, "Failed to open shared object");

    struct stat shared_stats;
    if (fstat(fd, &shared_stats) == -1)
        die(2, "Failed to check shared object size");

    if (shared_stats.st_size == 0) {
        if (ftruncate(fd, SHARED_OBJ_SIZE) == -1)
            printf("ftruncate failed with code %d\n", errno);
        if (fstat(fd, &shared_stats) == -1)
            die(2, "Failed to check shared object size");
    }

    printf("Shared object size: %lld\n", shared_stats.st_size);

    data = mmap(NULL, SHARED_OBJ_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (data == MAP_FAILED)
        die(3, "Failed to map shared object");

    while (1) {
        do_job(data, reader_pid);
        if (!strcmp(data, "q"))
            break;
    }

    if (munmap(data, 255))
        die(4, "munmap failed");

    // Writing process will close the object first so ENOENT is fine
    if (shm_unlink(SHARED_OBJ_NAME) == -1 && errno != ENOENT)
        die(5, "shm_unlink failed");

    return 0;
}
