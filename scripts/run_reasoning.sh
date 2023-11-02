if [ $# -ge 4 ]; then
    THREADS=$4
else
    THREADS=8
fi

if [ "$5" ]; then
    PLACEHOLDER="--$5"
fi

helm-run --conf-paths $1 --suite $2 --max-eval-instances $3 -n $THREADS $PLACEHOLDER