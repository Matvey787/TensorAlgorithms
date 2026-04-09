START_INDEX=3
END_INDEX=20
for i in $(eval echo {$START_INDEX..$END_INDEX})
    do
        /usr/bin/python3 /home/matvey/work/BitonicSort2/tests/randomizer.py tests/e2e/test${i}.dat -100000 100000 $((i * 1301));
    done