#!/bin/bash

docker exec --user docker_mssplace -it ${USER}_mssplace \
    /bin/bash -c "cd /home/docker_mssplace; echo ${USER}_mssplace container; echo ; /bin/bash"
