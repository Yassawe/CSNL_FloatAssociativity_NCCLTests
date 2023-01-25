echo "singlethreaded"

build/scripts/allreduce_multithread


echo "multithreaded"

mpirun -np 4 --allow-run-as-root build/scripts/allreduce_multithread


